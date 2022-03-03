"""
"""

from tqdm import tqdm
import torch

from vp_suite.encoders_decoders import DCGAN64_Encoder, DCGAN64_Decoder, VGG64_Encoder, VGG64_Decoder, VGGSp_Encoder, VGGSp_Decoder
from vp_suite.model_blocks import LSTM, GaussianLSTM
from vp_suite.base.base_model import VideoPredictionModel


class SVG(VideoPredictionModel):
    """
    Deterministic SVG model from Denton & Fergus, https://arxiv.org/abs/1802.07687. 2018

    Parameters
    ----------
    in_channels: integer
        number of color channels in the images
    nf: integer
        Base number of convolutional kernels for the encoder and decoder
    hidden_dim: integer
        Dimensionality of the LSTMS
    num_layers: integer
        Number of LSTM layers
    encoder_arch: string {DCGAN, VGG}
        Type of architecture used for the encoder/decoder
    learned_prior: bool
        If True, GaussianLSTM is used to learn an approximate posterior q(z_{t}:x_{1:t}) of the
        underlying data distribution. This posterior is matched to a learned prior Gaussian
        distribution p(z_{t}:x_{1:t-1}).
    num_gaussian_layers: integer
        Number of Gaussian LSTMs used for the learned prior and approximate posterior
    latent_hidden_dim: integer
        Hidden dimensionality of the Gaussian LSTMs
    latent_dim: integer
        Dimensionality of the sampled latent vectors (and also corresponding mean and log-var)
    """

    NAME = "SVG"
    REQUIRED_ARGS = ["img_shape", "action_size", "tensor_value_range", "in_channels",
                     "nf", "hidden_dim", "num_layers", "learned_prior", "encoder_arch"]

    def __init__(self, device, **model_kwargs):
        """ Module initializer """
        super(SVG, self).__init__(device, **model_kwargs)
        if self.encoder_arch not in ["DCGAN", "VGG", "VGGSp"]:
            raise ValueError("SVG only supports [DCGAN, VGG, VGGSp] for the encoder_arch")
        self.kl_mult = model_kwargs.pop("kl_mult", 1e-4)

        in_dim = self.in_dim if not self.learned_prior else self.in_dim + self.latent_dim
        if self.encoder_arch == "DCGAN":
            self.encoder = DCGAN64_Encoder(nc=self.in_channels, nf=self.nf, dim=self.in_dim)
            self.decoder = DCGAN64_Decoder(nc=self.in_channels, nf=self.nf, dim=self.in_dim)
        elif self.encoder_arch == "VGG":
            self.encoder = VGG64_Encoder(nc=self.in_channels, nf=self.nf, dim=self.in_dim)
            self.decoder = VGG64_Decoder(nc=self.in_channels, nf=self.nf, dim=self.in_dim)
        elif self.encoder_arch == "VGGSp":
            self.encoder = VGGSp_Encoder(nc=self.in_channels, nf=self.nf, dim=self.in_dim)
            self.decoder = VGGSp_Decoder(nc=self.in_channels, nf=self.nf, dim=self.in_dim)
        self.predictor = LSTM(
                input_dim=in_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.in_dim,
                num_layers=self.num_layers
            )
        if self.learned_prior:
            self.prior_network = GaussianLSTM(
                input_dim=self.in_dim,
                hidden_dim=self.latent_hidden_dim,
                output_dim=self.latent_dim,
                num_layers=self.latent_num_layers
            )
            self.posterior_network = GaussianLSTM(
                input_dim=self.in_dim,
                hidden_dim=self.latent_hidden_dim,
                output_dim=self.latent_dim,
                num_layers=self.latent_num_layers
            )

    def forward(self, input_tensor, pred_frames=1, teacher_force=False, **kwargs):
        """
        Forward pass through the SVG model
        Parameters:
        -----
        x: torch Tensor
            Batch of sequences to feed to the model. Shape is (B, Frames, C, H, W)
        context: integer
            number of seed frames to give as context to the model
        pred_frames: integer
            number of frames to predict. #frames=pred_frames are predicted autoregressively
        teacher_force: boolean
            If True, real frame is given as input during autoregressive prediction mode
        Returns:
        --------
        predictions: torch Tensor
            TODO:
        out_dict: dictionary
            dict containing the means and variances for the prior and
            posterior distributions, which are needed for the KL-Loss
        """
        (B, num_frames, _, _, _), device = input_tensor.shape, input_tensor.device
        pred_state, prior_state, posterior_state = self._init_hidden(batch_size=B, device=device)
        self.predictor.hidden_state = pred_state
        if self.learned_prior:
            self.prior_network.hidden_state = prior_state
            self.posterior_network.hidden_state = posterior_state

        inputs, targets = input_tensor[:, :].float(), input_tensor[:, 1:].float()
        next_input = inputs[:, 0]  # first frame

        context = num_frames - pred_frames
        preds, mus_post, logvars_post, mus_prior, logvars_prior = [], [], [], [], []
        for t in range(0, context + pred_frames - 1):
            # encoding images
            target_feats, _ = self.encoder(targets[:, t])
            if (t < context):
                feats, skips = self.encoder(next_input)
            else:
                feats, _ = self.encoder(next_input)
            # predicting latent and learning distribution
            if (self.learned_prior):
                z_post, (mu_post, logvar_post) = self.posterior_network(target_feats)
                z_prior, (mu_prior, logvar_prior) = self.prior_network(feats)
                latent = z_post if (t < context - 1 or self.training) else z_prior
                feats = torch.cat([feats, latent], -1)

            # predicting future features and decoding next frame
            pred_feats = self.predictor(feats)
            pred_output, _ = self.decoder([pred_feats, skips])

            if t >= context - 1 or teacher_force:
                preds.append(pred_output)
                # print(f"{t} save")
                if self.learned_prior:
                    mus_post.append(mu_post)
                    logvars_post.append(logvar_post)
                    mus_prior.append(mu_prior)
                    logvars_prior.append(logvar_prior)

            # feeding GT in context or teacher-forced mode, autoregressive o/w
            next_input = inputs[:, t+1] if (t < context-1 or teacher_force) else pred_output

        preds = torch.stack(preds, dim=1)
        kl_loss = self.kl_loss(mus_prior, logvars_prior, mus_post, logvars_post)
        return preds, {"kl_loss": kl_loss * self.kl_mult}

    def train_iter(self, config, loader, optimizer, loss_provider, epoch):
        """
        SVG's training iteration employs the teacher forcing approach. The model always receives the ground
        truth inputs, and predicts all frames, including during the seed stage.
        Additionally, it uses the objective function: MSE + KL-Diverge
        Args:
            config (dict): The configuration dict of the current training run (combines model, dataset and run config)
            loader (DataLoader): Training data is sampled from this loader.
            optimizer (Optimizer): The optimizer to use for weight update calculations.
            loss_provider (PredictionLossProvider): An instance of the :class:`LossProvider` class
            epoch (int): The current epoch.
        """
        self.train()
        loop = tqdm(loader)
        pred_frames = config["pred_frames"]
        for batch_idx, data in enumerate(loop):
            # preparing data
            input, targets, actions = self.unpack_data(data, config)
            imgs = torch.cat((input, targets), dim=1)
            predictions, model_losses = self(imgs, pred_frames=pred_frames, teacher_force=False)
            # loss on all frames: context + predicted
            # targets = imgs[:, 1:]
            _, total_loss = loss_provider.get_losses(predictions, targets)
            if model_losses is not None:
                for value in model_losses.values():
                    total_loss += value
            # bwd
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # bookkeeping
            loop.set_postfix(loss=total_loss.item())
        return

    def eval_iter(self, config, loader, loss_provider):
        r"""
        SVG's evaluation iteration employs the teacher forcing approach.
        Additionally, it uses the objective function: MSE + KL-Diverge
        Args:
            config (dict): The configuration dict of the current validation run (combines model, dataset and run config)
            loader (DataLoader): Validation data is sampled from this loader.
            loss_provider (PredictionLossProvider): An instance of the :class:`LossProvider` class for flexible loss calculation.
        Returns: A dictionary containing the averages value for each loss type specified for usage,
        as well as the value for the 'indicator' loss (the loss used for determining overall model improvement).
        """
        self.eval()
        loop = tqdm(loader)
        all_losses = []
        indicator_losses = []

        pred_frames = config["pred_frames"]
        with torch.no_grad():
            for batch_idx, data in enumerate(loop):
                # preparing data
                input, targets, actions = self.unpack_data(data, config)
                imgs = torch.cat((input, targets), dim=1)
                predictions, model_losses = self(imgs, pred_frames=pred_frames, teacher_force=False)

                # metrics
                loss_values, _ = loss_provider.get_losses(predictions, targets)
                all_losses.append(loss_values)
                indicator_losses.append(loss_values[config["val_rec_criterion"]])
        indicator_loss = torch.stack(indicator_losses).mean()
        all_losses = {
            k: torch.stack([loss_values[k] for loss_values in all_losses]).mean().item() for k in all_losses[0].keys()
        }
        self.train()

        return all_losses, indicator_loss

    def _init_hidden(self, batch_size, device):
        """ Basic logic for initializing hidden states. It's overriden in Hierarch mode l"""
        predictor_state = self.predictor._init_hidden(batch_size=batch_size, device=device)
        if(self.learned_prior):
            prior_state = self.prior_network._init_hidden(batch_size=batch_size, device=device)
            posterior_state = self.posterior_network._init_hidden(batch_size=batch_size, device=device)
        else:
            prior_state, posterior_state = None, None
        return predictor_state, prior_state, posterior_state

    def kl_loss(self, mu1, logvar1, mu2, logvar2):
        """ Computing the KL-Divergence between two Gaussian distributions """
        if self.learned_prior:
            mu1 = torch.stack(mu1, dim=1)
            logvar1 = torch.stack(logvar1, dim=1)
            mu2 = torch.stack(mu2, dim=1)
            logvar2 = torch.stack(logvar2, dim=1)
            sigma1 = logvar1.mul(0.5).exp()
            sigma2 = logvar2.mul(0.5).exp()
            kld = torch.log(sigma2/sigma1) + (torch.exp(logvar1) + (mu1 - mu2)**2)/(2*torch.exp(logvar2)) - 1/2
            kld = kld.sum(dim=-1).mean(dim=-1).mean(dim=-1)
        else:
            kld = torch.tensor(0.)
        return kld

#

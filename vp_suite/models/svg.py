"""
TODO (Angel): Make swapping encoder and decoder more flexible
"""

import torch

from vp_suite.encoders_decoders import DCGAN64_Encoder, DCGAN64_Decoder, VGG64_Encoder, VGG64_Decoder
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
        if self.encoder_arch not in ["DCGAN", "VGG"]:
            raise ValueError(f"SVG only supports [DCGAN, VGG] for the encoder_arch")
        super(SVG, self).__init__(device, **model_kwargs)

        if self.encoder_arch == "DCGAN":
            self.encoder = DCGAN64_Encoder(nc=self.in_channels, nf=self.nf, dim=self.in_dim)
            self.decoder = DCGAN64_Decoder(nc=self.in_channels, nf=self.nf, dim=self.in_dim)
        elif self.encoder_arch == "VGG":
            self.encoder = VGG64_Encoder(nc=self.in_channels, nf=self.nf, dim=self.in_dim)
            self.decoder = VGG64_Decoder(nc=self.in_channels, nf=self.nf, dim=self.in_dim)
        in_dim = self.nf * 8 if not self.learned_prior else self.nf * 8 + self.latent_dim
        self.predictor = LSTM(
                input_dim=in_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.nf * 8,
                num_layers=self.num_layers
            )
        if self.learned_prior:
            self.prior_network = GaussianLSTM(
                input_dim=self.nf * 8,
                hidden_dim=self.latent_hidden_dim,
                output_dim=self.latent_dim,
                num_layers=self.num_layers
            )
            self.posterior_network = GaussianLSTM(
                input_dim=self.nf * 8,
                hidden_dim=self.latent_hidden_dim,
                output_dim=self.latent_dim,
                num_layers=self.num_layers
            )

    def forward(self, input_tensor, context=10, pred_frames=10, teacher_force=False, **kwargs):
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
        if(context + pred_frames - 1 > num_frames):
            raise ValueError(f"""The number of frames in the sequence ({num_frames} must not be
                             smaller than context + pred_frames ({context + pred_frames })""")

        inputs, targets = input_tensor[:, :].float(), input_tensor[:, 1:].float()
        next_input = inputs[:, 0]  # first frame

        for t in range(0, context + pred_frames - 1):
            # encoding images
            target_feats, _ = self.encoder(targets[:, t]) if (t < num_frames-1) else (None, None)
            if (t < context):
                feats, skips = self.encoder(next_input)
            else:
                feats, _ = self.encoder(next_input)

            # predicting latent and learning distribution
            preds, mus_post, logvars_post, mus_prior, logvars_prior = [], [], [], [], []
            if (self.learned_prior):
                if t < num_frames-1:
                    z_post, (mu_post, logvar_post) = self.posterior_network(target_feats)
                else:
                    z_post, (mu_post, logvar_post) = None, None, None
                z_prior, (mu_prior, logvar_prior) = self.prior_network(feats)
                latent = z_post if (t < context-1 or self.training) else z_prior
                feats = torch.cat([feats, latent], 1)
                mus_post.append(mu_post)
                logvars_post.append(logvar_post)
                mus_prior.append(mu_prior)
                logvars_prior.append(logvar_prior)

            # predicting future features and decoding next frame
            pred_feats = self.predictor(feats)
            pred_output, _ = self.decoder([pred_feats, skips])
            preds.append(pred_output)

            # feeding GT in context or teacher-forced mode, autoregressive o/w
            next_input = inputs[:, t+1] if (t < context-1 or teacher_force) else pred_output

        preds = torch.stack(preds, dim=1)
        return preds, None

    def _init_hidden(self, batch_size, device):
        """ Basic logic for initializing hidden states. It's overriden in Hierarch mode l"""
        predictor_state = self.predictor._init_hidden(batch_size=batch_size, device=device)
        if(self.learned_prior):
            prior_state = self.prior_network._init_hidden(batch_size=batch_size, device=device)
            posterior_state = self.posterior_network._init_hidden(batch_size=batch_size, device=device)
        else:
            prior_state, posterior_state = None, None
        return predictor_state, prior_state, posterior_state

#

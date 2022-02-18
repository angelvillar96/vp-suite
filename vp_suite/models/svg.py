"""
TODO (Angel): Make swapping encoder and decoder more flexible
"""

import torch

from vp_suite.encoders_decoders import DCGAN64_Encoder, DCGAN64_Decoder
from vp_suite.encoders_decoders import LSTM, GaussianLSTM
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
    REQUIRED_ARGS = ["in_dim", "nf", "hidden_dim", "num_layers"]

    def __init__(self, device, in_channels=1, nf=64, hidden_dim=256, num_layers=2, learned_prior=False,
                 num_gaussian_layers=1, latent_hidden_dim=64, latent_dim=10, **model_kwargs):
        """ Module initializer """
        super(SVG, self).__init__(device, in_channels, nf, hidden_dim, num_layers,
                                  num_gaussian_layers, learned_prior, **model_kwargs)
        self.in_channels = in_channels
        self.learned_prior = learned_prior

        self.encoder = DCGAN64_Encoder(nc=in_channels, nf=nf)
        self.decoder = DCGAN64_Decoder(nc=in_channels, nf=nf)
        self.predictor = LSTM(
                input_dim=nf * 8,
                hidden_dim=hidden_dim,
                output_dim=nf * 8,
                num_layers=num_layers
            )
        if learned_prior:
            self.prior_network = GaussianLSTM(
                input_dim=nf * 8,
                hidden_dim=latent_hidden_dim,
                output_dim=latent_dim,
                num_layers=num_layers
            )
            self.posterior_network = GaussianLSTM(
                input_dim=nf * 8,
                hidden_dim=latent_hidden_dim,
                output_dim=latent_dim,
                num_layers=num_layers
            )
        return

    def forward(self, input_tensor, context=10, pred_frames=10, teacher_force=False):
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
                z_prior, (mu_prior, logvar_prior) = self.prior_networke(feats)
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
        out_dict = {
            "preds": preds,
            "mu_prior": torch.stack(mus_prior, dim=1),
            "logvar_prior": torch.stack(logvars_prior, dim=1),
            "mu_post": torch.stack(mus_post, dim=1),
            "logvar_post": torch.stack(logvars_post, dim=1),
        }
        return preds, out_dict

    def _init_hidden(self, batch_size):
        """ Basic logic for initializing hidden states. It's overriden in Hierarch mode l"""
        device = self.encoder.device
        predictor_state = self.predictor._init_hidden(batch_size=batch_size, device=device)
        if(self.learned_prior):
            prior_state = self.prior.init_hidden(batch_size=batch_size, device=device)
            posterior_state = self.posterior.init_hidden(batch_size=batch_size,device=device)
        else:
            prior_state, posterior_state = None, None
        return predictor_state, prior_state, posterior_state

#

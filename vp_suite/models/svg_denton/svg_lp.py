"""
Original SVG-LearnedPrior.
Implementation from https://github.com/edenton/svg
"""

import torch
import torch.nn as nn
import vp_suite.models.svg_denton.dcgan_64 as model
import vp_suite.models.svg_denton.lstm as lstm_models


class SVGLP(nn.Module):
    """ """

    def __init__(self, in_dim, channels, rnn_dim, rnn_layers, b_size, latent_dim, latent_rnn_dim, latent_layers):
        """ """
        super(SVGLP, self).__init__()
        self.encoder = model.encoder(in_dim, channels)
        self.decoder = model.decoder(in_dim, channels)
        self.predictor = lstm_models.lstm(in_dim + latent_dim, in_dim, rnn_dim, rnn_layers, b_size)
        self.prior_network = lstm_models.gaussian_lstm(in_dim, latent_dim, latent_rnn_dim, latent_layers, b_size)
        self.posterior_network = lstm_models.gaussian_lstm(in_dim, latent_dim, latent_rnn_dim, latent_layers, b_size)

    def forward(self, x, n_pred):
        """ Forward pass during training"""
        if self.training:
            preds = self._train(x, n_pred)
        else:
            preds = self._eval(x, n_pred)
        return preds

    def _train(self, x, n_pred):
        """ Training forward pass """
        t, b, _, _, _ = x.shape
        n_future = n_pred
        n_past = t - n_pred
        self.predictor.hidden = self.predictor.init_hidden()
        self.posterior_network.hidden = self.posterior_network.init_hidden()
        self.prior_network.hidden = self.prior_network.init_hidden()
        preds = []

        for i in range(1, n_past + n_future):
            h = self.encoder(x[i-1])
            h_target = self.encoder(x[i])[0]
            if i < n_past:
                h, skip = h
            else:
                h = h[0]
            z_t, mu, logvar = self.posterior_network(h_target)
            _, mu_p, logvar_p = self.prior_network(h)
            h_pred = self.predictor(torch.cat([h, z_t], 1))
            x_pred = self.decoder([h_pred, skip])
            preds.append(x_pred)

        preds = torch.stack(preds, dim=1)
        return preds

    def _eval(self, x, n_pred):
        """ Training forward pass """
        t, b, _, _, _ = x.shape
        n_future = n_pred
        n_past = t - n_pred
        self.predictor.hidden = self.predictor.init_hidden()
        self.prior_network.hidden = self.prior_network.init_hidden()
        self.posterior_network.hidden = self.posterior_network.init_hidden()
        preds = []

        x_in = x[0]
        for i in range(1, n_past + n_future):
            h = self.encoder(x_in)
            if i < n_past:
                h, skip = h
            else:
                h = h[0]
            if i < n_past:
                h_target = self.encoder(x[i])[0]
                z, _, _ = self.posterior_network(h_target)
                _ = self.prior_network(h)
                _ = self.predictor(torch.cat([h, z], 1))
                x_in = x[i]
                preds.append(x_in)
            else:
                z, _, _ = self.prior_network(h)
                h = self.predictor(torch.cat([h, z], 1))
                x_in = self.decoder([h, skip])
                preds.append(x_in)

        preds = torch.stack(preds, dim=1)
        return preds


#

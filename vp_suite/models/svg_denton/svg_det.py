"""
Original SVG-Deterministic.
Implementation from https://github.com/edenton/svg
"""

import torch
import torch.nn as nn
import vp_suite.models.svg_denton.dcgan_64 as model
import vp_suite.models.svg_denton.lstm as lstm_models


class SVGDet(nn.Module):
    """ """

    def __init__(self, in_dim, channels, rnn_dim, rnn_layers, b_size):
        """ """
        super(SVGDet, self).__init__()
        self.encoder = model.encoder(in_dim, channels)
        self.decoder = model.decoder(in_dim, channels)
        self.predictor = lstm_models.lstm(in_dim, in_dim, rnn_dim, rnn_layers, b_size)

    def forward(self, x, n_pred):
        """ Forward pass during training"""
        t, b, _, _, _ = x.shape
        n_future = n_pred
        n_past = t - n_pred
        self.predictor.hidden = self.predictor.init_hidden()
        preds = []

        for i in range(1, n_past + n_future):
            h = self.encoder(x[i-1])
            if i < n_past:
                h, skip = h
            else:
                h = h[0]
            h_pred = self.predictor(h)
            x_pred = self.decoder([h_pred, skip])
            preds.append(x_pred)

        preds = torch.stack(preds, dim=1)
        return preds

#

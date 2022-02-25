""" """

import torch
from torch import nn

from vp_suite.base.base_model_block import ModelBlock


class LSTM(ModelBlock):
    """
    Implementation of an LSTM block: FC + LSTM + FC + Tan

    Parameters:
    -----
    input_dim: int
        Dimensionality of the input vector. Number of input neurons in first FC
    hidden_dim: int
        Dimensionality of the LSTM state and input
    outputs_dim: int
        Dimensionality of the output vector. Number of output neurons in last FC
    num_layers: integer
        number of cascaded LSTM cells
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, bias=True):
        """ Initializer of the LSTM block """
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.hidden_state = None

        self.embed = nn.Linear(input_dim, hidden_dim)
        self.cell_list = nn.ModuleList([
                nn.LSTMCell(hidden_dim, hidden_dim, bias) for i in range(self.num_layers)
            ])
        self.output = nn.Sequential(
                    nn.Linear(hidden_dim, output_dim),
                    nn.Tanh()
                )
        return

    def forward(self, input_tensor):
        """
        Forward pass through Gaussian LSTM

        Parameters
        ----------
        input_tensor: torch Tensor
            2-D Tensor either of shape (b d)

        Returns
        -------
        z: torch Tensor
            Latent vector sampled from the approximate posterior
        (mu, logvar): tuple (torch Tensor, torch Tensor)
            Mean and log-covariance of the Gaussain approximate posterior
        """
        b, d = input_tensor.shape
        # self.hidden_state = self.hidden_state if self.hidden_state is not None else self._init_hidden(batch_size=b)

        embedded = self.embed(input_tensor.view(-1, self.input_dim))
        h_in = embedded

        for layer_idx, lstm in enumerate(self.cell_list):
            self.hidden_state[layer_idx] = lstm(h_in, self.hidden_state[layer_idx])
            h_in = self.hidden_state[layer_idx][0]

        output_tensor = self.output(h_in)
        return output_tensor

    def _init_hidden(self, batch_size=1, device=None):
        """ Initializing hidden state vectors with zeros """
        init_states = []
        device = device if device is not None else self.embed.device
        for i in range(self.num_layers):
            init_states.append(
                (
                    torch.zeros(batch_size, self.hidden_dim, device=device),
                    torch.zeros(batch_size, self.hidden_dim, device=device)
                )
            )
        return init_states


class GaussianLSTM(ModelBlock):
    """
    LSTM that models the underlying prior distribution of the latent variables.
    It predicts the mean and covariance of the prior distribution, from which we
    then sample latent vectors.

    Parameters
    ----------
    input_dim: int
        Dimensionality of the input vector. Number of input neurons in first FC
    hidden_dim: int
        Dimensionality of the LSTM state
    output_size: int
        Dimensionality of the output vector. Number of output neurons in last FC
    num_layers: int
        Number of consecutive LSTM-Cells
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, batch_first=True, bias=True):
        """ """
        super(GaussianLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.hidden_state = None

        self.embed = nn.Linear(input_dim, hidden_dim)
        self.cell_list = nn.ModuleList([
                nn.LSTMCell(hidden_dim, hidden_dim, bias) for i in range(self.num_layers)
            ])
        self.mu_net = nn.Linear(hidden_dim, output_dim)
        self.logvar_net = nn.Linear(hidden_dim, output_dim)
        return

    def forward(self, input_tensor):
        """
        Forward pass through Gaussian LSTM
        Parameters
        ----------
        input_tensor: torch Tensor
            2-D Tensor either of shape (b, d)

        Returns
        -------
        z: torch Tensor
            Latent vector sampled from the approximate posterior
        (mu, logvar): tuple (torch Tensor, torch Tensor)
            Mean and log-covariance of the Gaussain approximate posterior
        """
        b, d = input_tensor.shape
        # self.hidden_state = self.hidden_state if self.hidden_state is not None else self._init_hidden(batch_size=b)

        embedded = self.embed(input_tensor.view(-1, self.input_dim))
        h_in = embedded
        for layer_idx, lstm in enumerate(self.cell_list):
            self.hidden_state[layer_idx] = lstm(h_in, self.hidden_state[layer_idx])
            h_in = self.hidden_state[layer_idx][0]

        mu = self.mu_net(h_in)
        logvar = self.logvar_net(h_in)
        z = self.reparameterize(mu, logvar)
        return z, (mu, logvar)

    def reparameterize(self, mu, logvar):
        """ Reparameterization trick """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z

    def _init_hidden(self, batch_size=1, device=None):
        """ Initializing hidden state vectors with zeros """
        init_states = []
        device = device if device is not None else self.embed.device
        for i in range(self.num_layers):
            init_states.append(
                (
                    torch.zeros(batch_size, self.hidden_dim, device=device),
                    torch.zeros(batch_size, self.hidden_dim, device=device)
                )
            )
        return init_states

#

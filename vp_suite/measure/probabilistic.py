r"""Module for probabilistic measures.

APPLIES TO ALL LOSSES:
- expected data type: torch.tensor (torch.float)
"""

import torch

from vp_suite.base.base_measure import BaseMeasure


class KLLoss(BaseMeasure):
    r"""
    KL-Divergence loss function
    """
    NAME = "KL-Divergence (KL)"

    def __init__(self, device):
        super(KLLoss, self).__init__(device)

    def criterion(self, mu1, logvar1, mu2, logvar2):
        """ Computing the KL-Divergence between two Gaussian distributions """
        sigma1 = logvar1.mul(0.5).exp()
        sigma2 = logvar2.mul(0.5).exp()
        kld = torch.log(sigma2/sigma1) + (torch.exp(logvar1) + (mu1 - mu2)**2)/(2*torch.exp(logvar2)) - 1/2
        return kld

    def forward(self, mu1, logvar1, mu2, logvar2):
        """ Computing the KL-Divergence between two Gaussian distributions """
        value = self.criterion(mu1, logvar1, mu2, logvar2)
        return value.sum(dim=(-1)).mean()

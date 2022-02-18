"""
DCGAN-64 Encoder and Decoder.
 - Downsamples the image from 64x64 -> 1x1 while increasing the number of channels
"""

import torch
import torch.nn as nn

BN_TRACK_STATS = False


class dcgan_conv(nn.Module):
    def __init__(self, nin, nout):
        super().__init__()
        self.main = nn.Sequential(
                nn.Conv2d(nin, nout, 4, 2, 1),
                nn.BatchNorm2d(nout, track_running_stats=BN_TRACK_STATS),
                nn.LeakyReLU(0.2, inplace=True),
                )

    def forward(self, input):
        return self.main(input)


class dcgan_upconv(nn.Module):
    def __init__(self, nin, nout):
        super().__init__()
        self.main = nn.Sequential(
                nn.ConvTranspose2d(nin, nout, 4, 2, 1),
                nn.BatchNorm2d(nout, track_running_stats=BN_TRACK_STATS),
                nn.LeakyReLU(0.2, inplace=True),
                )

    def forward(self, input):
        return self.main(input)


class DCGAN_Encoder(nn.Module):
    """
    DCGAN-like encoder. Uses 4 Conv-blocks that downsample the image by a factor of 2
    while increasing the number of channels by 2

    Parameters:
    -----------
    nc: int
        number of channels in the input image
    nf: int
        Base number of convolutional kernels. It's multiplied by 2 at each Conv. Block
    """

    def __init__(self, dim, nc=1, nf=64):
        """ """
        super().__init__()
        self.spatial_dims = [(32, 32), (16, 16), (8, 8), (4, 4), (1, 1)]
        self.dim = dim
        # input is (nc) x 64 x 64
        self.c1 = dcgan_conv(nc, nf)
        # state size. (nf) x 32 x 32
        self.c2 = dcgan_conv(nf, nf * 2)
        # state size. (nf*2) x 16 x 16
        self.c3 = dcgan_conv(nf * 2, nf * 4)
        # state size. (nf*4) x 8 x 8
        self.c4 = dcgan_conv(nf * 4, nf * 8)
        # state size. (nf*8) x 4 x 4
        self.c5 = nn.Sequential(
                nn.Conv2d(nf * 8, dim, 4, 1, 0),
                nn.BatchNorm2d(dim, track_running_stats=BN_TRACK_STATS),
                nn.Tanh()
                )

    def forward(self, input):
        """ """
        h1 = self.c1(input)
        h2 = self.c2(h1)
        h3 = self.c3(h2)
        h4 = self.c4(h3)
        h5 = self.c5(h4)
        return h5.view(-1, self.dim), [h1, h2, h3, h4]


class DCGAN_Decoder(nn.Module):
    """
    DCGAN-like decoder. Uses 4 TransposedConv-blocks that upsample the image by a factor of 2
    while decreasing the number of channels by 2.
    Residual connections comming from the encoder are concatenated with the decoded features
    after each decoder block.

    Parameters:
    -----------
    nc: int
        number of channels in the output image
    nf: int
        Base number of convolutional kernels.
    """

    def __init__(self, dim, nc=1, nf=64):
        super().__init__()
        self.dim = dim
        self.upc1 = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(dim, nf * 8, 4, 1, 0),
                nn.BatchNorm2d(nf * 8, track_running_stats=BN_TRACK_STATS),
                nn.LeakyReLU(0.2, inplace=True)
                )
        # state size. (nf*8) x 4 x 4
        self.upc2 = dcgan_upconv(nf * 8 * 2, nf * 4)
        # state size. (nf*4) x 8 x 8
        self.upc3 = dcgan_upconv(nf * 4 * 2, nf * 2)
        # state size. (nf*2) x 16 x 16
        self.upc4 = dcgan_upconv(nf * 2 * 2, nf)
        # state size. (nf) x 32 x 32
        self.upc5 = nn.Sequential(
                nn.ConvTranspose2d(nf * 2, nc, 4, 2, 1),
                nn.Sigmoid()
                # state size. (nc) x 64 x 64
                )

    def forward(self, input):
        vec, skip = input
        d1 = self.upc1(vec.view(-1, self.dim, 1, 1))
        d2 = self.upc2(torch.cat([d1, skip[3]], 1))
        d3 = self.upc3(torch.cat([d2, skip[2]], 1))
        d4 = self.upc4(torch.cat([d3, skip[1]], 1))
        output = self.upc5(torch.cat([d4, skip[0]], 1))
        return output, [d1, d2, d3, d4]

#

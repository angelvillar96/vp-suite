"""
VGG-64 Encoder and Decoder.
 - Downsamples the image from 64x64 -> 1x1 while increasing the number of channels
"""


import torch
import torch.nn as nn

BN_TRACK_STATS = False


class vgg_layer(nn.Module):
    def __init__(self, nin, nout):
        super(vgg_layer, self).__init__()
        self.main = nn.Sequential(
                nn.Conv2d(nin, nout, 3, 1, 1),
                nn.BatchNorm2d(nout, track_running_stats=BN_TRACK_STATS),
                nn.LeakyReLU(0.2, inplace=True)
                )

    def forward(self, input):
        return self.main(input)


class VGG_Encoder(nn.Module):
    def __init__(self, nc=1, nf=64, dim=128):
        super(VGG_Encoder, self).__init__()
        self.dim = dim
        # 64 x 64
        self.c1 = nn.Sequential(
                vgg_layer(nc, nf),
                vgg_layer(nf, nf),
                )
        # 32 x 32
        self.c2 = nn.Sequential(
                vgg_layer(nf, nf * 2),
                vgg_layer(nf * 2, nf * 2),
                )
        # 16 x 16
        self.c3 = nn.Sequential(
                vgg_layer(nf * 2, nf * 4),
                vgg_layer(nf * 4, nf * 4),
                vgg_layer(nf * 4, nf * 4),
                )
        # 8 x 8
        self.c4 = nn.Sequential(
                vgg_layer(nf * 4, nf * 8),
                vgg_layer(nf * 8, nf * 8),
                vgg_layer(nf * 8, nf * 8),
                )
        # 4 x 4
        self.c5 = nn.Sequential(
                nn.Conv2d(nf * 8, dim, 4, 1, 0),
                nn.BatchNorm2d(dim, track_running_stats=BN_TRACK_STATS),
                nn.Tanh()
                )
        self.mp = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def get_spatial_dims(self, img_size, level=-1):
        assert level == -1
        return (1, 1)

    def forward(self, input):
        h1 = self.c1(input)  # 64 -> 32
        h2 = self.c2(self.mp(h1))  # 32 -> 16
        h3 = self.c3(self.mp(h2))  # 16 -> 8
        h4 = self.c4(self.mp(h3))  # 8 -> 4
        h5 = self.c5(self.mp(h4))  # 4 -> 1
        return h5.view(-1, self.dim), [h1, h2, h3, h4]


class VGG_Decoder(nn.Module):
    def __init__(self, nc=1, nf=64, dim=128):
        super(VGG_Decoder, self).__init__()
        self.dim = dim
        # 1 x 1 -> 4 x 4
        self.upc1 = nn.Sequential(
                nn.ConvTranspose2d(dim, nf * 8, 4, 1, 0),
                nn.BatchNorm2d(nf * 8, track_running_stats=BN_TRACK_STATS),
                nn.LeakyReLU(0.2, inplace=True)
                )
        # 8 x 8
        self.upc2 = nn.Sequential(
                vgg_layer(nf * 8 * 2, nf * 8),
                vgg_layer(nf * 8, nf * 8),
                vgg_layer(nf * 8, nf * 4)
                )
        # 16 x 16
        self.upc3 = nn.Sequential(
                vgg_layer(nf * 4 * 2, nf * 4),
                vgg_layer(nf * 4, nf * 4),
                vgg_layer(nf * 4, nf * 2)
                )
        # 32 x 32
        self.upc4 = nn.Sequential(
                vgg_layer(nf * 2 * 2, nf * 2),
                vgg_layer(nf * 2, nf)
                )
        # 64 x 64
        self.upc5 = nn.Sequential(
                vgg_layer(nf * 2, nf),
                nn.ConvTranspose2d(nf, nc, 3, 1, 1),
                nn.Sigmoid()
                )
        self.up = nn.UpsamplingNearest2d(scale_factor=2)

    def forward(self, input):
        vec, skip = input
        d1 = self.upc1(vec.view(-1, self.dim, 1, 1))  # 1 -> 4
        up1 = self.up(d1)  # 4 -> 8
        d2 = self.upc2(torch.cat([up1, skip[3]], 1))  # 8 x 8
        up2 = self.up(d2)  # 8 -> 16
        d3 = self.upc3(torch.cat([up2, skip[2]], 1))  # 16 x 16
        up3 = self.up(d3) # 8 -> 32
        d4 = self.upc4(torch.cat([up3, skip[1]], 1))  # 32 x 32
        up4 = self.up(d4)  # 32 -> 64
        output = self.upc5(torch.cat([up4, skip[0]], 1))  # 64 x 64
        return output, [d1, d2, d3, d4]

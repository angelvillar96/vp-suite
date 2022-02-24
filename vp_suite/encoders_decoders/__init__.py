r"""
This package contains different encoder and decoder modules used by the
video prediction models.
"""

from .DCGAN_64 import DCGAN_Encoder as DCGAN64_Encoder
from .DCGAN_64 import DCGAN_Decoder as DCGAN64_Decoder
from .VGG_64 import VGG_Encoder as VGG64_Encoder
from .VGG_64 import VGG_Decoder as VGG64_Decoder

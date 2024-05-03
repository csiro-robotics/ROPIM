# --------------------------------------------------------
# ROPIM
# Based on https://github.com/microsoft/SimMIM
# Written by Maryam Haghighat
# --------------------------------------------------------

from .vision_transformer import build_vit
from .ropim import build_ropim


def build_model(config, is_pretrain=True):
    if is_pretrain:
        model = build_ropim(config)
    else:
        model = build_vit(config)
    return model

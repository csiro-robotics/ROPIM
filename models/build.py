# --------------------------------------------------------
# SimSIM
# Written by Maryam Haghighat
# --------------------------------------------------------

from .vision_transformer import build_vit
from .simsim import build_simsim


def build_model(config, is_pretrain=True):
    if is_pretrain:
        model = build_simsim(config)
    else:
        model = build_vit(config)
    return model

# --------------------------------------------------------
# ROPIM
# Written by Maryam Haghighat
# --------------------------------------------------------

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from .vision_transformer import VisionTransformer
from functools import partial


class VisionTransformerForROPIM(VisionTransformer):
    def __init__(self, sp=False, **kwargs):
        super().__init__(**kwargs)
        assert self.num_classes == 0
        self.sp = sp

    def _trunc_normal_(self, tensor, mean=0., std=1.):
        trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)

    def forward(self, x, sp_sketch_invsketch):
        x = self.patch_embed(x)
        B, L, _ = x.shape

        if self.sp:
            sp_sketch_invsketch=sp_sketch_invsketch.type(x.type())
            x = torch.matmul(x.permute(0, 2, 1), sp_sketch_invsketch)  # Matrix x sketched, x_sk
            x = x.permute(0, 2, 1)  # x_hat

        cls_tokens = self.cls_token.expand(B, -1, -1)  #  cls_tokens impl from Phil Wang
        x = torch.cat((cls_tokens, x), dim=1)

        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None

        for i, blk in enumerate(self.blocks):
            x = blk(x, rel_pos_bias=rel_pos_bias)

        x = self.norm(x)

        x = x[:, 1:]
        B, L, C = x.shape
        H = W = int(L ** 0.5)
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        return x


class ROPIM(nn.Module):
    def __init__(self, encoder, encoder_stride):
        super().__init__()
        self.encoder = encoder
        self.encoder_stride = encoder_stride

        self.decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=self.encoder.num_features,
                out_channels=self.encoder_stride ** 2 * 3, kernel_size=1),
            nn.PixelShuffle(self.encoder_stride),
        )

        self.in_chans = self.encoder.in_chans
        self.patch_size = self.encoder.patch_size

    def forward(self, x, sp_sketch_invsketch):
        z = self.encoder(x, sp_sketch_invsketch)
        x_rec = self.decoder(z)
        return x_rec


    @torch.jit.ignore
    def no_weight_decay(self):
        if hasattr(self.encoder, 'no_weight_decay'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay()}
        return {}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        if hasattr(self.encoder, 'no_weight_decay_keywords'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay_keywords()}
        return {}


def build_ropim(config):
    encoder = VisionTransformerForROPIM(
    sp=config.DATA.sp,
    img_size=config.DATA.IMG_SIZE,
    patch_size=config.MODEL.VIT.PATCH_SIZE,
    in_chans=config.MODEL.VIT.IN_CHANS,
    num_classes=0,
    embed_dim=config.MODEL.VIT.EMBED_DIM,
    depth=config.MODEL.VIT.DEPTH,
    num_heads=config.MODEL.VIT.NUM_HEADS,
    mlp_ratio=config.MODEL.VIT.MLP_RATIO,
    qkv_bias=config.MODEL.VIT.QKV_BIAS,
    drop_rate=config.MODEL.DROP_RATE,
    drop_path_rate=config.MODEL.DROP_PATH_RATE,
    norm_layer=partial(nn.LayerNorm, eps=1e-6),
    init_values=config.MODEL.VIT.INIT_VALUES,
    use_abs_pos_emb=config.MODEL.VIT.USE_APE,
    use_rel_pos_bias=config.MODEL.VIT.USE_RPB,
    use_shared_rel_pos_bias=config.MODEL.VIT.USE_SHARED_RPB,
    use_mean_pooling=config.MODEL.VIT.USE_MEAN_POOLING)
    encoder_stride = config.MODEL.VIT.PATCH_SIZE

    model = ROPIM(encoder=encoder, encoder_stride=encoder_stride)

    return model

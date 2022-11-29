from .data_simsim import build_loader_simsim
from .data_finetune import build_loader_finetune

def build_loader(config, logger, is_pretrain):
    if is_pretrain:
        return build_loader_simsim(config, logger)
    else:
        return build_loader_finetune(config, logger)
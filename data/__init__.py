from .data_ropim import build_loader_ropim
from .data_finetune import build_loader_finetune

def build_loader(config, logger, is_pretrain):
    if is_pretrain:
        return build_loader_ropim(config, logger)
    else:
        return build_loader_finetune(config, logger)
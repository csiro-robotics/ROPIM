# --------------------------------------------------------
# SimSIM
# Written by Maryam Haghighat
# --------------------------------------------------------


import os
import datetime
import time
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from timm.utils import AverageMeter
from config import get_config
from models import build_model
from data import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, save_checkpoint, get_grad_norm, auto_resume_helper, init_distributed_mode
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision import transforms as T


def str2bool(v):
  return v.lower() in ('true', '1')


def rev_PixelShuffle(x, r):
    B, C, H, W = x.shape
    x = x.reshape(B, C, H//r, r, W//r, r)
    x = x.permute(0, 1, 3, 5, 2, 4)
    x = x.reshape(B, C*(r**2), H//r, W//r)
    return x.contiguous()


def parse_option():
    parser = argparse.ArgumentParser('SimSIM pre-training', add_help=False)
    parser.add_argument('--cfg', type=str,  metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')

    parser.add_argument('--tag', help='tag of experiment')
    # Sketching
    parser.add_argument('--Spatial_Sketching_Threshold', type=float, default=0)
    parser.add_argument('--sp', type= str2bool, default=True, help='apply spatial sketching')
    parser.add_argument('--eye_sp_sketch', type= str2bool, default=False)
    parser.add_argument('--amp-opt-level', type=str, default='O0', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    # distributed training
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--world_size', default=4, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_on_itp', action='store_true')
    args = parser.parse_args()
    config = get_config(args)
    return args, config


def main(config):
    data_loader_train = build_loader(config, logger, is_pretrain=True)

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config, is_pretrain=True)
    model.cuda()
    logger.info(str(model))

    optimizer = build_optimizer(config, model, logger, is_pretrain=True)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False)
    model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model_without_ddp, 'flops'):
        flops = model_without_ddp.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT, logger)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, logger)

    logger.info("Start training")
    start_time = time.time()
    if dist.get_rank() == 0 and not os.path.exists(f'imgs/{config.TAG}'):
        os.makedirs(f'imgs/{config.TAG}')

    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(config, model, data_loader_train, optimizer, epoch, lr_scheduler)
        if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            save_checkpoint(config, epoch, model_without_ddp, 0., optimizer, lr_scheduler, logger)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def train_one_epoch(config, model, data_loader, optimizer, epoch, lr_scheduler):
    model.train()

    optimizer.zero_grad()
    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()

    start = time.time()
    end = time.time()
    for idx, (img_org, sp_sketch_invsketch, _) in enumerate(data_loader):
        img_org = img_org.cuda(non_blocking=True)
        if config.DATA.sp:
            sp_sketch_invsketch = sp_sketch_invsketch.cuda(non_blocking=True)
            if config.DATA.eye_sp_sketch:
                eye_mat = torch.eye(sp_sketch_invsketch.shape[1]).cuda(non_blocking=True)
                sp_sketch_invsketch = eye_mat - sp_sketch_invsketch

        img_rec = model(img_org, sp_sketch_invsketch)
        model_loss = img_org - img_rec

        model_loss = rev_PixelShuffle(model_loss, config.MODEL.VIT.PATCH_SIZE)
        eye_mat = torch.eye(sp_sketch_invsketch.shape[1]).cuda(non_blocking=True)
        C_Sketch_invSketch = eye_mat - sp_sketch_invsketch


        # Division option 1
        model_loss = torch.matmul(model_loss.flatten(2), C_Sketch_invSketch) / (
                C_Sketch_invSketch.sum((1, 2)).abs().view(-1, 1, 1) + 1e-5)
        model_loss = model_loss.abs().mean()*200 #/ config.DATA.BATCH_SIZE / config.MODEL.VIT.IN_CHANS

        # Division option 2
        # model_loss = torch.matmul(model_loss.flatten(2), C_Sketch_invSketch)
        # model_loss = model_loss.abs().mean() * 40000 / (
        #         C_Sketch_invSketch.mean().abs() + 1e-5)
        # / config.DATA.BATCH_SIZE / config.MODEL.VIT.IN_CHANS
        # torch.numel(C_Sketch_invSketch[0])

        # Division option 3
        # C_Sketch_invSketch.sum().abs() + 1e-5)

        # Try DC mat
        # DC_mat=torch.matmul(torch.ones(model_loss.flatten(2).size()).cuda(non_blocking=True), C_Sketch_invSketch).mean((1,2))
        # model_loss = torch.matmul(model_loss.flatten(2), C_Sketch_invSketch) / (
        #         DC_mat.view(-1,1,1) + 1e-5)  # (C_Sketch_invSketch.sum((1,2)).abs()+1e-5)
        ##########################

        optimizer.zero_grad()
        model_loss.backward()
        if config.TRAIN.CLIP_GRAD:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
        else:
            grad_norm = get_grad_norm(model.parameters())
        optimizer.step()
        lr_scheduler.step_update(epoch * num_steps + idx)
        torch.cuda.synchronize()
        loss_meter.update(model_loss.item(), img_org.size(0))
        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()
        torch.distributed.barrier()

        # save images for further investigation
        # if dist.get_rank() == 0 and idx == num_steps - 1:
        #     mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
        #     std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
        #     unnormalize = T.Normalize((-mean / std), (1.0 / std))
        #
        #     a1 = img_org[0, :, :, :].detach().clone()
        #     save_image(torch.clamp(unnormalize(a1), 0, 1), f'imgs/{config.TAG}/org_img_it{idx}_epoch{epoch}.png')
        #     a2 = img_rec[0, :, :, :].detach().clone()
        #     save_image(torch.clamp(unnormalize(a2), 0, 1), f'imgs/{config.TAG}/img_rec_it{idx}_epoch{epoch}.png')


    if idx % config.PRINT_FREQ == 0 or idx == num_steps - 1:
        lr = optimizer.param_groups[0]['lr']
        memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
        etas = batch_time.avg * (num_steps - idx)
        logger.info(
            f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
            f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
            f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
            f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
            f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
            f'mem {memory_used:.0f}MB \t')

    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


# if __name__ == '__main__':
#     _, config = parse_option()
#
#     if config.AMP_OPT_LEVEL != "O0":
#         assert amp is not None, "amp not installed!"
#
#     if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
#         rank = int(os.environ["RANK"])
#         world_size = int(os.environ['WORLD_SIZE'])
#         print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
#     else:
#         rank = -1
#         world_size = -1
#     torch.cuda.set_device(config.LOCAL_RANK)
#     # os.environ["NCCL_ASYNC_ERROR_HANDLING"]=1 timeout=datetime.timedelta(seconds=3600),
#     torch.distributed.init_process_group(backend='nccl', init_method='env://',  world_size=world_size, rank=rank)
#     torch.distributed.barrier()
#
#     seed = config.SEED + dist.get_rank()
#     torch.manual_seed(seed)
#     np.random.seed(seed)
#     cudnn.benchmark = True


if __name__ == '__main__':
    _, config = parse_option()
    init_distributed_mode(config)
    device = torch.device('cuda')

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # Apply a linear learning rate rule based on total batch size, Note: not optimal
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0

    # Scale the learning rate with accumulation steps
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    if dist.get_rank() == 0 and not os.path.exists(config.OUTPUT):
        os.makedirs(config.OUTPUT)

    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())

    main(config)

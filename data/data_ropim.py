# --------------------------------------------------------
# ROPIM
# Written by Maryam Haghighat
# --------------------------------------------------------

import random
import numpy as np

import torch
import torch.distributed as dist
import torchvision.transforms as T
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data._utils.collate import default_collate
from torchvision.datasets import ImageFolder
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
###### SKETCH ########

class SketchGenerator:
    def __init__(self,  input_size, threshold):
        assert input_size  != 0
        self.n_size_in = int(input_size)
        self.threshold=threshold
    def create_s_dense(self, hi, si, c):
        d = hi.size(0)
        out = torch.zeros((d, c), device=hi.device)  # in*out
        out[torch.arange(0, d), hi.type(torch.LongTensor)] = si
        return out

    def choose_h_sk_mat(self, nSketchDimC, nFeatDimC, device):
        nRep = int(np.ceil(nFeatDimC / nSketchDimC))
        rand_array = torch.zeros(nFeatDimC, device=device)
        for i in range(nRep):
            rand_array_i = torch.randperm(int(nSketchDimC))
            rand_array[i * nSketchDimC:(i + 1) * nSketchDimC] = rand_array_i

        return rand_array

    def choose_s_sk_mat(self, nSketchDimC, nFeatDimC, device):
        nRep = int(np.ceil(nFeatDimC / nSketchDimC))
        rand_array = [-1, 1] * nRep
        random.shuffle(rand_array)
        return torch.tensor(rand_array, dtype=torch.float32, device=device)

    def create_sketch_mat(self, dim, sketch_dim, device):
        h1 = self.choose_h_sk_mat(sketch_dim, dim, device)
        s1 = self.choose_s_sk_mat(2, dim, device)
        sdense1 = self.create_s_dense(h1, s1, sketch_dim)
        return sdense1

    def __call__(self):
        if not self.threshold:
            return None

        if torch.rand(1) > float(self.threshold):
            self.sketching_ratio = .25 #.07 
        else:
            self.sketching_ratio = .14 # .25

        self.n_size_out = int(np.ceil(self.n_size_in * self.sketching_ratio))
        sketch_mat = self.create_sketch_mat(self.n_size_in , self.n_size_out, "cpu")
        sketch_invsketch =torch.matmul(sketch_mat, sketch_mat.t() * self.sketching_ratio)

        ## ADDED to try
        # DC_mat = torch.matmul(torch.ones(int(16*16*3), self.n_size_in).float(), sketch_invsketch.float())
        # sketch_invsketch = sketch_invsketch / DC_mat.mean()
        #####
        return sketch_invsketch


class ROPIMTransform:
    def __init__(self, config):
        self.transform_img = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.RandomResizedCrop(config.DATA.IMG_SIZE, scale=(0.67, 1.), ratio=(3. / 4., 4. / 3.)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=torch.tensor(IMAGENET_DEFAULT_MEAN),std=torch.tensor(IMAGENET_DEFAULT_STD)),
        ])
 
        if config.MODEL.TYPE == 'swin':
            model_patch_size=config.MODEL.SWIN.PATCH_SIZE
        elif config.MODEL.TYPE == 'vit':
            model_patch_size=config.MODEL.VIT.PATCH_SIZE
        else:
            raise NotImplementedError

        self.SpacialSketchGenerator = SketchGenerator(
            input_size=(config.DATA.IMG_SIZE/model_patch_size)**2, threshold=config.DATA.Spatial_Sketching_Threshold)


    def __call__(self, img):
        img = self.transform_img(img)
        sp_sketch_invsketch = self.SpacialSketchGenerator()
        return img, sp_sketch_invsketch


def collate_fn(batch):
    if not isinstance(batch[0][0], tuple):
        return default_collate(batch)
    else:
        batch_num = len(batch)
        ret = []
        for item_idx in range(len(batch[0][0])):
            if batch[0][0][item_idx] is None:
                ret.append(None)
            else:
                ret.append(default_collate([batch[i][0][item_idx] for i in range(batch_num)]))
        ret.append(default_collate([batch[i][1] for i in range(batch_num)]))
        return ret


def build_loader_ropim(config, logger):
    transform = ROPIMTransform(config)
    logger.info(f'Pre-train data transform:\n{transform}')

    dataset = ImageFolder(config.DATA.DATA_PATH, transform)
    logger.info(f'Build dataset: train images = {len(dataset)}')
    
    sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True)
    dataloader = DataLoader(dataset, config.DATA.BATCH_SIZE, sampler=sampler, num_workers=config.DATA.NUM_WORKERS, pin_memory=True, drop_last=True, collate_fn=collate_fn)
    
    return dataloader
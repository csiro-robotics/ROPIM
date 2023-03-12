# --------------------------------------------------------
# SimSIM
# Written by Maryam Haghighat
# --------------------------------------------------------

import os
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import Mixup
from timm.data import create_transform
from timm.data.transforms import _pil_interp
import json
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import  verify_str_arg
import PIL.Image

def build_loader_finetune(config, logger):
    config.defrost()
    dataset_train, config.MODEL.NUM_CLASSES = build_dataset(is_train=True, config=config, logger=logger)
    config.freeze()
    dataset_val, _ = build_dataset(is_train=False, config=config, logger=logger)
    logger.info(f"Build dataset: train images = {len(dataset_train)}, val images = {len(dataset_val)}")

    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    sampler_train = DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    sampler_val = DistributedSampler(
        dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False
    )

    data_loader_train = DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
    )

    data_loader_val = DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False,
    )

    # setup mixup / cutmix
    mixup_fn = None
    mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
            prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
            label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES)

    return dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn

def build_dataset(is_train, config, logger):
    transform = build_transform(is_train, config)
    logger.info(f'Fine-tune data transform, is_train={is_train}:\n{transform}')

    if config.DATA.DATASET == 'imagenet':
        prefix = 'train' if is_train else 'validation'
        root = os.path.join(config.DATA.DATA_PATH, prefix)
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000  

    elif config.DATA.DATASET == 'imagenet100':
        prefix = 'train' if is_train else 'val'
        root = os.path.join(config.DATA.DATA_PATH, prefix)
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 100  
    elif config.DATA.DATASET == 'CUB':
        prefix = 'train' if is_train else 'val'
        root = os.path.join(config.DATA.DATA_PATH, prefix)
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 200
    elif config.DATA.DATASET == 'flowers102':
        prefix = 'train' if is_train else 'val'
        dataset = datasets.Flowers102(root=config.DATA.DATA_PATH, download=False, transform=transform)
        nb_classes = 102 
    elif config.DATA.DATASET == 'inat17':
        if is_train:
            split='train'
        else:
            split='val'
        dataset=INat2017(root='/datasets/work/mlaifsp-st-d61/source/iNaturalist', split=split, transform=transform)

        nb_classes=13    

    elif config.DATA.DATASET == 'cifar100':
        dataset = datasets.CIFAR100(root=config.DATA.DATA_PATH,  train=is_train, download=False, transform=transform)
        nb_classes=100

    elif config.DATA.DATASET == 'cifar10':
        dataset = datasets.CIFAR100(root=config.DATA.DATA_PATH,  train=is_train, download=False, transform=transform)
        nb_classes=10

    else:
        raise NotImplementedError("We only support ImageNet Now.")

    return dataset, nb_classes



def build_dataset_CIFAR(is_train, config, logger):
    transform = build_transform(is_train, config)
    logger.info(f'Fine-tune data transform, is_train={is_train}:\n{transform}')
    dataset = datasets.CIFAR10(root='./data',  train=is_train, download=False, transform=transform)
    nb_classes=10
    return dataset, nb_classes


def build_transform(is_train, config):
    resize_im = config.DATA.IMG_SIZE > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=config.DATA.IMG_SIZE,
            is_training=True,
            color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
            auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != 'none' else None,
            re_prob=config.AUG.REPROB,
            re_mode=config.AUG.REMODE,
            re_count=config.AUG.RECOUNT,
            interpolation=config.DATA.INTERPOLATION,
        )

        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(config.DATA.IMG_SIZE, padding=4)
        return transform

    t = []
    if resize_im:
        if config.TEST.CROP:
            size = int((256 / 224) * config.DATA.IMG_SIZE)
            t.append(
                transforms.Resize(size, interpolation=_pil_interp(config.DATA.INTERPOLATION)),
                # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(config.DATA.IMG_SIZE))
        else:
            t.append(
                transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                                  interpolation=_pil_interp(config.DATA.INTERPOLATION))
            )

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)



class INat2017(VisionDataset):

    base_folder = 'train_val_images/'
    file_list = {
        'imgs': ('https://storage.googleapis.com/asia_inat_data/train_val/train_val_images.tar.gz',
                 'train_val_images.tar.gz',
                 '7c784ea5e424efaec655bd392f87301f'),
        'annos': ('https://storage.googleapis.com/asia_inat_data/train_val/train_val2017.zip',
                  'train_val2017.zip',
                  '444c835f6459867ad69fcb36478786e7')
    }

    def __init__(self, root, split='train', transform=None, target_transform=None, download=False):
        super(INat2017, self).__init__(root, transform=transform, target_transform=target_transform)
        self.loader = default_loader
        self.split = verify_str_arg(split, "split", ("train", "val",))

        anno_filename = '/datasets/work/mlaifsp-st-d61/source/iNaturalist/train_val2017/' + split + '2017.json'
        with open(os.path.join(self.root, anno_filename), 'r') as fp:
            all_annos = json.load(fp)

        self.annos = all_annos['annotations']
        self.images = all_annos['images']

        self.cat_ids = {'Plantae': 0, 'Insecta': 1, 'Aves': 2, 'Reptilia': 3, 'Mammalia': 4, 'Fungi':5, 'Amphibia': 6,
                        'Mollusca': 7, 'Animalia': 8, 'Arachnida': 9, 'Actinopterygii': 10, 'Chromista': 11, 'Protozoa': 12}

    def __getitem__(self, index):
        path = os.path.join(self.root, self.images[index]['file_name'])

        split_name = self.images[index]['file_name'].split('/')
        target = self.cat_ids[split_name[1]]

        image = PIL.Image.open(path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

    def __len__(self):
        return len(self.images)
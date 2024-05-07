# ROPIM: Pre-training with Random Orthogonal Projection Image Modeling

[ICLR 2024 Spotlight] The official repository of Self-Supervised Learning methods "ROPIM", [Pre-training with Random Orthogonal Projection Image Modeling](https://openreview.net/forum?id=z4Hcegjzph)

ROPIM is a self-supervised learning technique based on count sketching, which  reduces local semantic information under the bounded noise variance. While  Masked Image Modelling (MIM) introduces Binary noise, ROPIM proposes a _continous_ masking strategy. 
Continuous masking allows for larger number of masking patterns compared to binary masking.
![alt text](figures/ROPIM.png)


## Citation

If you find our work useful for your research, please consider giving a star :star: and citation :beer::

```bibtex
@inproceedings{
haghighat2024ROPIM,
title={Pre-training with Random Orthogonal Projection Image Modeling},
author={Maryam Haghighat and Peyman Moghadam and Shaheer Mohamed and Piotr Koniusz},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=z4Hcegjzph}
}
```
## Usage

Setup conda environment and install required packages:
```bash
# Create environment
conda create -n ropim python=3.8 -y
conda activate ropim

# Install requirements
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge

# Clone ROPIM repo
git clone https://github.com/csiro-robotics/ROPIM
cd ROPIM

# Install other requirements
pip install -r requirements.txt
```
### Pre-training with ROPIM
For pre-training models with `ROPIM`, run:
                   
```bash
python3 -m torch.distributed.launch --nnodes <num-of-nodes> --nproc_per_node <num-of-gpus-per-node> --node_rank <node-rank> --master_addr <hostname> \
main_ropim.py --world_size <total-num-of-gpus> \ 
--cfg <config-file> --data-path <imagenet-path>/train [--batch-size <batch-size-per-gpu> --output <output-directory> --tag <job-tag> --spatial_sketching_threshold <threshol-for-sketching_ratio>]
```
### Fine-tuning pre-trained models
For fine-tuning models pre-trained by `ROPIM`, run:

```bash
python3 -m torch.distributed.launch --nnodes <num-of-nodes> --nproc_per_node <num-of-gpus-per-node> --node_rank <node-rank> --master_addr <hostname> \
main_finetune.py --world_size <total-num-of-gpus> \ 
--cfg <config-file> --data-path <imagenet-path> --pretrained <pretrained-ckpt> [--batch-size <batch-size-per-gpu> --output <output-directory> --tag <job-tag>]
```

## Acknowledgement

This research was funded by the Machine Learning and Artificial Intelligence Future Science Platform (MLAI FSP) and Science Digital at the Commonwealth Scientific and Industrial Research Organisation (CSIRO), Australia.

This code is built using the [timm](https://github.com/huggingface/pytorch-image-models) library, the [BEiT](https://github.com/microsoft/unilm/tree/master/beit) repository and the [SimMIM](https://github.com/microsoft/SimMIM) repository.

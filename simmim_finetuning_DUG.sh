#!/bin/bash
#SBATCH -p csiro_od219033
#SBATCH -C a100x4
#SBATCH --time=23:59:00
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=4
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=END


. "/d/sw/miniconda3/4.8.3/etc/profile.d/conda.sh"
conda activate /data/csiro_od219033/sw/venv/hs
srun sh -c 'python main_finetune.py \
--world_size 8 --batch-size 380 --data-path /data/csiro_od219033/imagenet \
--cfg configs/vit_base/simsim_finetune__vit_base.yaml \
--pretrained output/simsim_pretrain/simsim_pretrain_vit_base_imgnet_sp1SkR.14_LR1.25-5_LossDividedSum1_2_200LossMean_multinode_300e/ckpt_epoch_299.pth \
--tag simsim_finetune_vit_base_imgnet_sp1SkR.14_LR1.25-5_LossDividedSum1_2_200LossMean_multinode_300e_imgnet_100e-corrected'
# threshold 2 => .14
# threshold -1 => .07

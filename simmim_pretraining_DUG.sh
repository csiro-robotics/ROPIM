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
srun sh -c 'python main_simsim.py \
--world_size 8 --batch-size 430 --data-path /data/csiro_od219033/imagenet/train \
--cfg configs/vit_base/simsim_pretrain__vit_base.yaml --Spatial_Sketching_Threshold 2 --sp 'True' \
--eye_sp_sketch 'False' --tag simsim_pretrain_vit_base_imgnet_sp1SkR.14_LR5-5_LossDividedSum1_2_200LossMean_multinode_300e_step_100_200_gamma.5'
# threshold 2 => .14
# threshold -1 => .07

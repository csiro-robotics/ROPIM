#!/bin/bash
#SBATCH --account=OD-219033
#SBATCH --time=10:00:00
#SBATCH --mem=80gb
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --mail-user=maryam.haghighat@csiro.au
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=END

export MASTER_PORT=$(expr 10009 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

source /datasets/work/mlaifsp-st-d61/work/hag036/miniconda3/etc/profile.d/conda.sh
conda activate hs_copy
srun sh -c 'python main_finetune.py \
  --world_size 8 \
  --batch-size 256 \
  --cfg configs/vit_tiny__800ep/simmim_finetune__vit_tiny__img224__800ep.yaml \
  --data-path /datasets/work/mlaifsp-st-d61/source/new/CUB \
  --pretrained /scratch1/hag036/SimMIM_sketching/output/simmim_pretrain/simmim_pretrain__vit_base_imgnet100_sp1SkR.25_LR1-3_LossDividedSum1_2_LossDivNumel/ckpt_epoch_799.pth \
  --tag simmim_finetune__simmim_pretrain__vit_tiny_imgnet100_sp1SkR.25_LR1-3_LossDividedSum1_2_LossDivNumel_ch799_INat17_NoAutoAug_8FTbaseLR_100epochs_5warmup_TotBath2048'

#'simmim_pretrain__vit_base_imgnet100_sp1SkR.25_LR1-3_LossDividedSum1_2_LossDivNumel
#  --pretrained /datasets/work/mlaifsp-st-d61/work/hag036/HSI_mlaifsp/SimMIM_sketching/sketching_all/output/simmim_pretrain/simmim_pretrain__vit_tiny_imgnet100_sp1SkR.25_LR1-3/ckpt_epoch_799.pth \

#  --tag simmim_finetune__vit_tiny_imgnet100_sp1SkR.25_LR1-3_DivdedbySumCketch_ch799_flower102_NoAutoAug_NoJit_NoMixup_4FTbaseLR_warmup5ep_100epochs_newTransform_20warmup'
  #32FTbaseLR works best for imagenet100
#/scratch1/hag036/SimMIM_sketching

#  --pretrained /scratch1/hag036/SimMIM_sketching/output/simmim_pretrain/simmim_pretrain__vit_tiny_imgnet100_sp1SkR.25_LR5-3_b4x128/ckpt_epoch_799.pth \

#  -1=> ratio=.5,     .=> ratio=.25
#  EYE,Th=.25 => ratio=0.4375
#  EYE,Th=.25 => ratio=0.4375

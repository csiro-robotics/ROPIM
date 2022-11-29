#!/bin/bash
#SBATCH --account=OD-219033
#SBATCH --time=1:00:00
#SBATCH --mem=50gb
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:2
#SBATCH --mail-user=maryam.haghighat@csiro.au
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=END
export MASTER_PORT=$(expr 10555 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR


source /datasets/work/mlaifsp-st-d61/work/hag036/miniconda3/etc/profile.d/conda.sh
conda activate hs
srun sh -c 'python main_simsim.py \
    --world_size 4\
    --data-path /scratch1/hag036/imagenet100/train \
    --batch-size 128 \
    --cfg configs/vit_tiny/simsim_pretrain__vit_tiny.yaml \
    --Spatial_Sketching_Threshold 2 \
    --sp 'True' \
    --eye_sp_sketch 'False' \
    --tag simmim_pretrain_ADIOS_vit_tiny_imgnet100_sp1SkR.25_LR1-3_LossDividedSum1_2_LossDivNumel_test'


# Th=-1      => ratio=.5,
# Th=2       => ratio=.25
# Th=.25     => ratio=0.4375
# Th=.5      => ratio=0.375
# Th=.75     => ratio=0.3125

# EYE,Th=.25 => ratio=.5625
# EYE,Th=2   => ratio=0.75
# EYE,Th=.75 => ratio=.6875

#--data-path /scratch1/hag036/imagenet100/train \
#--data-path /datasets/work/mlaifsp-st-d61/source/imagenet/train \

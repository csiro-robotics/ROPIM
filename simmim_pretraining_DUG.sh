#!/bin/bash
#SBATCH -p csiro_od219033
#SBATCH -C a100x4
#SBATCH --time=24:00:00
#SBATCH --nodes=2
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=4
#SBATCH --mail-user=sha457@csiro.au
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=END
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR
export MASTER_PORT=$master_port

. "/d/sw/miniconda3/4.8.3/etc/profile.d/conda.sh"
conda activate /data/csiro_od219033/sw/venv/hs
srun sh -c 'python -m torch.distributed.launch --master_port 12346 --nproc_per_node 4 main_simsim.py \
--world_size 8 --batch-size 380 --data-path /data/csiro_od219033/imagenet/train \
--cfg configs/vit_base/simsim_pretrain__vit_base.yaml --Spatial_Sketching_Threshold -1 --sp 'True' \
--eye_sp_sketch 'False' --tag simsim_pretrain_vit_base_imgnet_sp1SkR.07_LR2.5-5_LossDividedSum1_2_200LossMean_multinode_300e'
# threshold 2 => .14
# threshold -1 => .07

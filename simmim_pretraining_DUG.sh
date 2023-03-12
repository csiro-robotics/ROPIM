#!/bin/bash
#SBATCH -p csiro_od216929
#SBATCH -C a100x4
#SBATCH --time=24:00:00
#SBATCH --nodes=4
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=4 
#SBATCH --mail-user=maryam.haghighat@csiro.au
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=END
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

. "/d/sw/miniconda3/4.8.3/etc/profile.d/conda.sh"
conda activate /data/csiro_od216929/sw/python_virtual_envs/hs
srun sh -c 'python3 -m torch.distributed.launch --nnodes 4 --nproc_per_node 4 \
--master_addr "$MASTER_NAME" --node_rank "$JOB_RANK" main_simsim.py \
--world_size 16 --batch-size 190 --data-path /g/data/wb00/ImageNet/ILSVRC2012/raw-data/train \
--cfg configs/vit_base/simsim_pretrain__vit_base.yaml --Spatial_Sketching_Threshold -1 --sp 'True' \
--eye_sp_sketch 'False' --tag simsim_pretrain_vit_base_imgnet_sp1SkR.07_LR1.25-5_LossDividedSum1_2_200LossMean_multinode_300e'




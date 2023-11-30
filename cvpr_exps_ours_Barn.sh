#!/bin/sh
JOB_START_TIME=$(date)
echo "SLURM_JOB_ID:    ${SLURM_JOB_ID}" 
echo "Running on node: $(hostname)"
echo "Starting on:     ${JOB_START_TIME}" 
### 2022.10.12 debug nice-slam default
module load gcc/8.2.0
module load python_gpu/3.10.4
export PYTHONPATH=$HOME/.local/lib/python3.8/site-packages:$PYTHONPATH
module load eth_proxy
pip install timm

python train_posenet.py configs/Tanks/Barn_ct.yaml

#  sbatch --output=sbatch_log/ours_Barn_3days.out --time=3-0 --gpus=titan_rtx:1 --mem-per-cpu=40g cvpr_exps_ours_Barn.sh


# checked: 36733537
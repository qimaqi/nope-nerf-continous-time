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

python train.py configs/Tanks/Museum.yaml

#  sbatch --output=sbatch_log/baseline_Museum_3days.out --time=3-0 --gpus=rtx_2080_ti:1 --mem-per-cpu=40g cvpr_exps_baseline_Museum.sh


# checked: 36636603
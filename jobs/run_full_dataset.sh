#!/bin/bash
#$ -N full_dataset
#$ -q gpu
#$ -pe sharedmem 1
#$ -l h_vmem=40G
#$ -l h_rt=02:00:00
#$ -l gpu=1
#$ -o /exports/eddie/scratch/s2887048/logs/full_dataset.o$JOB_ID
#$ -e /exports/eddie/scratch/s2887048/logs/full_dataset.e$JOB_ID
#$ -cwd

# Load environment
source /etc/profile.d/modules.sh
module load anaconda/2024.02

export PROJECT_SCRATCH=/exports/eddie/scratch/s2887048
export HF_HOME=$PROJECT_SCRATCH/hf_cache
export TRANSFORMERS_CACHE=$PROJECT_SCRATCH/hf_cache

source activate $PROJECT_SCRATCH/envs/attribution

cd $HOME/llm-hypothesis-attribution

echo "=== Full dataset run started at $(date) ==="
echo "Host: $(hostname)"
nvidia-smi
echo ""

time python -m scripts.run_full_dataset

echo ""
echo "=== Full dataset run finished at $(date) ==="

#!/bin/bash
#$ -N full_llama8b
#$ -q gpu
#$ -pe sharedmem 1
#$ -l h_vmem=60G
#$ -l h_rt=08:00:00
#$ -l gpu=1
#$ -o /exports/eddie/scratch/s2887048/logs/full_llama8b.o$JOB_ID
#$ -e /exports/eddie/scratch/s2887048/logs/full_llama8b.e$JOB_ID
#$ -cwd

source /etc/profile.d/modules.sh
module load anaconda/2024.02

export PROJECT_SCRATCH=/exports/eddie/scratch/s2887048
export HF_HOME=$PROJECT_SCRATCH/hf_cache
export TRANSFORMERS_CACHE=$PROJECT_SCRATCH/hf_cache

source activate $PROJECT_SCRATCH/envs/attribution

cd $HOME/llm-hypothesis-attribution

echo "=== Llama-3.1-8B run started at $(date) ==="
echo "Host: $(hostname)"
nvidia-smi
echo ""

time python -m scripts.run_full_dataset --model llama8b

echo ""
echo "=== Llama-3.1-8B run finished at $(date) ==="

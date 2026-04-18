#!/bin/bash
#$ -N benchmark_both
#$ -q gpu
#$ -pe sharedmem 1
#$ -l h_vmem=40G
#$ -l h_rt=01:00:00
#$ -l gpu=1
#$ -o /exports/eddie/scratch/s2887048/logs/benchmark_both.o$JOB_ID
#$ -e /exports/eddie/scratch/s2887048/logs/benchmark_both.e$JOB_ID
#$ -cwd

# Load environment
source /etc/profile.d/modules.sh
module load anaconda/2024.02

export PROJECT_SCRATCH=/exports/eddie/scratch/s2887048
export HF_HOME=$PROJECT_SCRATCH/hf_cache
export TRANSFORMERS_CACHE=$PROJECT_SCRATCH/hf_cache

source activate $PROJECT_SCRATCH/envs/attribution

# SSH agent for any git operations
eval "$(ssh-agent -s)" > /dev/null 2>&1
ssh-add ~/.ssh/id_ed25519 > /dev/null 2>&1

cd $HOME/llm-hypothesis-attribution

echo "=== Job started at $(date) ==="
echo "Host: $(hostname)"
nvidia-smi
echo ""

time python -m scripts.run_benchmark_both

echo ""
echo "=== Job finished at $(date) ==="

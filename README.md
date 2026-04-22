# What Drives LLM-Generated Hypotheses? A Large-Scale Attribution Analysis of Scientific Abstract Sections

Large language models are increasingly used to generate scientific hypotheses, yet which parts of scientific text actually drive these outputs remains unknown. This repository contains the code, data, and analysis for the first large-scale attribution study measuring how rhetorical sections of scientific abstracts — Background, Objective, Method, Result, and Other — influence LLM-generated hypotheses.

**Central finding:** Result sections almost never drive hypothesis generation (0.4–0.7% top-ranked), while Objective sections dominate (36–38%). This hierarchy is stable across three model scales (1.1B to 8B parameters) and two attribution methods (Feature Ablation and Shapley Value Sampling), which agree in 93–95% of cases.

## Table of Contents

- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Setup](#setup)
  - [Local Setup](#local-setup)
  - [HPC Setup (Eddie / Slurm / SGE)](#hpc-setup)
- [Reproducing the Experiments](#reproducing-the-experiments)
  - [Step 1: Verify the Dataset Loads](#step-1-verify-the-dataset-loads)
  - [Step 2: Single-Abstract Smoke Test](#step-2-single-abstract-smoke-test)
  - [Step 3: 10-Abstract Benchmark](#step-3-10-abstract-benchmark)
  - [Step 4: Full Dataset Run](#step-4-full-dataset-run)
  - [Step 5: Analysis and Figures](#step-5-analysis-and-figures)
- [Understanding the Pipeline](#understanding-the-pipeline)
  - [Data Loading](#data-loading)
  - [Hypothesis Generation](#hypothesis-generation)
  - [Value Function](#value-function)
  - [Feature Ablation](#feature-ablation)
  - [Shapley Value Sampling](#shapley-value-sampling)
- [Outputs](#outputs)
- [Models Used](#models-used)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

## Repository Structure

```
llm-hypothesis-attribution/
│
├── src/                          # Core library code
│   ├── data/
│   │   └── load_csabstruct.py    # Dataset loading and parsing
│   ├── models/
│   │   └── load_model.py         # Unified model loading wrapper
│   ├── generation/
│   │   └── prompts.py            # Prompt templates
│   ├── attribution/
│   │   ├── value_function.py     # Log-probability computation
│   │   ├── feature_ablation.py   # Feature Ablation implementation
│   │   └── shapley.py            # Shapley Value Sampling implementation
│   └── utils/
│       └── io.py                 # I/O utilities for saving results
│
├── scripts/                      # Runnable entry points
│   ├── run_single_abstract.py    # Single-abstract smoke test
│   ├── run_benchmark_10.py       # 10-abstract benchmark (FA only)
│   ├── run_benchmark_both.py     # 10-abstract benchmark (FA + Shapley)
│   ├── run_full_dataset.py       # Full dataset run (parameterised by model)
│   └── run_analysis.py           # Analysis, figures, and tables
│
├── jobs/                         # HPC job submission scripts (SGE)
│   ├── test_gpu.sh
│   ├── run_phi3.sh
│   ├── run_llama8b.sh
│   └── run_analysis.sh
│
├── results/                      # Summary outputs (tracked in git)
│   ├── full_tinyllama/
│   │   ├── summary.csv
│   │   └── run_metadata.json
│   ├── full_phi3/
│   │   ├── summary.csv
│   │   └── run_metadata.json
│   ├── full_llama8b/
│   │   ├── summary.csv
│   │   └── run_metadata.json
│   └── figures/                  # Generated figures (PDF + PNG)
│
├── paper.tex                     # Paper manuscript
├── requirements.txt
└── README.md
```

## Requirements

- **Python** 3.11 (tested; 3.10 should also work)
- **GPU:** Any CUDA-compatible GPU with at least 8GB VRAM for TinyLlama, 16GB for Phi-3, 24GB for Llama-3.1-8B. All experiments were run on NVIDIA A100 80GB, but consumer GPUs (RTX 3090, 4090) will work — just slower.
- **CPU-only:** The pipeline runs on CPU as well, but expect 10–20 minutes per abstract instead of 2–3 seconds. Useful for debugging, not for full runs.
- **Disk space:** ~15GB for all three model weights (cached by HuggingFace), plus ~1.5GB for raw JSON outputs across all models.

### Python Dependencies

```
torch>=2.1.0,<2.5.0
transformers>=4.36.0,<5.0.0
datasets>=2.16.0,<3.0.0
captum>=0.7.0
accelerate>=0.26.0
scipy>=1.11.0
matplotlib>=3.8.0
numpy>=1.24.0,<2.0.0
pyyaml>=6.0
```

## Setup

### Local Setup

This is the recommended path for getting started. You can run the entire pipeline on a laptop (with a GPU) or a single workstation.

**1. Clone the repository:**

```bash
git clone https://github.com/Adnan1729/llm-hypothesis-attribution.git
cd llm-hypothesis-attribution
```

**2. Create a Python 3.11 virtual environment:**

```bash
# On Linux/Mac
python3.11 -m venv venv
source venv/bin/activate

# On Windows (Git Bash)
py -3.11 -m venv venv
source venv/Scripts/activate
```

If you don't have Python 3.11, install it from [python.org](https://www.python.org/downloads/release/python-3119/) or via your package manager.

**3. Install dependencies:**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**4. Verify the installation:**

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "from transformers import AutoModelForCausalLM; print('Transformers OK')"
```

You should see your PyTorch version and whether CUDA is available. The pipeline works without CUDA (just slowly).

**5. (For Llama-3.1-8B only) Accept the Meta license and authenticate:**

Llama-3.1-8B requires you to accept Meta's license on HuggingFace before downloading:

1. Go to [huggingface.co/meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
2. Click "Accept License" and wait for approval (usually instant)
3. Create a read-access token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
4. Authenticate locally:

```bash
huggingface-cli login
# Paste your token when prompted
```

TinyLlama and Phi-3 do not require authentication.

### HPC Setup

If you have access to an HPC cluster with GPUs (e.g., university compute resources), the full dataset run completes in under 2 hours. These instructions assume an SGE-based scheduler (like University of Edinburgh's Eddie), but adapt easily to Slurm.

**1. Clone the repo to your home directory:**

```bash
cd ~
git clone https://github.com/Adnan1729/llm-hypothesis-attribution.git
cd llm-hypothesis-attribution
```

**2. Set up a scratch workspace for large files:**

Model weights and raw outputs should go on a high-capacity scratch filesystem, not your home directory. Adjust paths below to match your cluster's layout.

```bash
# Example for Edinburgh Eddie — replace with your cluster's scratch path
export PROJECT_SCRATCH=/exports/eddie/scratch/$USER
mkdir -p $PROJECT_SCRATCH/envs
mkdir -p $PROJECT_SCRATCH/hf_cache
mkdir -p $PROJECT_SCRATCH/outputs
mkdir -p $PROJECT_SCRATCH/logs
```

**3. Add environment variables to your shell profile:**

```bash
cat >> ~/.bashrc << 'EOF'
export PROJECT_SCRATCH=/exports/eddie/scratch/$USER
export HF_HOME=$PROJECT_SCRATCH/hf_cache
EOF
source ~/.bashrc
```

**4. Create a conda environment in scratch:**

```bash
# Load your cluster's conda/anaconda module
module load anaconda    # exact name varies by cluster

# Create env in scratch (not home — envs are multi-GB)
conda create -p $PROJECT_SCRATCH/envs/attribution python=3.11 -y
conda activate $PROJECT_SCRATCH/envs/attribution
pip install -r requirements.txt
```

**5. Test on a GPU node (interactive session):**

```bash
# SGE example — adjust queue name and resource flags for your cluster
qlogin -q gpu -pe sharedmem 1 -l h_vmem=40G -l h_rt=02:00:00 -l gpu=1

# Once on the GPU node:
module load anaconda
conda activate $PROJECT_SCRATCH/envs/attribution
cd ~/llm-hypothesis-attribution
nvidia-smi                              # confirm GPU is visible
python -m scripts.run_single_abstract   # should show Device: cuda:0
```

**6. Submit batch jobs:**

See the `jobs/` directory for example SGE job scripts. The key structure is:

```bash
#!/bin/bash
#$ -N full_dataset
#$ -q gpu
#$ -pe sharedmem 1
#$ -l h_vmem=40G
#$ -l h_rt=02:00:00
#$ -l gpu=1

source /etc/profile.d/modules.sh
module load anaconda
export PROJECT_SCRATCH=/exports/eddie/scratch/$USER
export HF_HOME=$PROJECT_SCRATCH/hf_cache
source activate $PROJECT_SCRATCH/envs/attribution
cd $HOME/llm-hypothesis-attribution

python -m scripts.run_full_dataset --model tinyllama
```

For Slurm-based clusters, the equivalent would use `#SBATCH` directives and `sbatch` for submission.

## Reproducing the Experiments

The pipeline is designed to be run incrementally. Each step validates the previous one before scaling up.

### Step 1: Verify the Dataset Loads

```bash
python -m src.data.load_csabstruct
```

**Expected output:**

```
Loaded 226 abstracts from test split
First abstract (test_0):
  Num sentences: 6
  Label distribution: [('background', 2), ('objective', 3), ('other', 1)]
  First 2 sentences: ['While deep convolutional neural networks ...', ...]
```

This downloads the CSABSTRUCT dataset from HuggingFace (~1MB) and prints a summary of the first abstract. The dataset contains 2,189 computer science abstracts with gold-standard sentence-level rhetorical role labels (background, objective, method, result, other).

### Step 2: Single-Abstract Smoke Test

```bash
python -m scripts.run_single_abstract
```

**What this does:** Loads one abstract, generates a hypothesis with TinyLlama, runs Feature Ablation across all sections, and prints the attribution scores.

**What to look for:**
- The generated hypothesis should be a coherent restatement of the paper's main claim.
- The full-context log-probability should be much higher (less negative) than the empty-context log-probability. A delta of 50+ log-prob units is typical.
- Attribution scores should be positive for most sections, with a clear hierarchy.

**First run downloads the model weights (~2.2GB for TinyLlama).** Subsequent runs are fast.

**Approximate time:** 2–3 minutes on GPU, 10–20 minutes on CPU.

### Step 3: 10-Abstract Benchmark

```bash
python -m scripts.run_benchmark_both
```

**What this does:** Runs both Feature Ablation and Shapley Value Sampling on 10 abstracts that have at least 4 of the 5 rhetorical labels. For each abstract, it generates one hypothesis, computes attributions with both methods, saves results as JSON, and prints a side-by-side comparison showing whether the two methods agree on the most influential section.

**What to look for:**
- Both methods should agree on the top section for most abstracts (our result: 9/10).
- Shapley values are typically higher in magnitude than Feature Ablation scores.
- Timing per abstract should be 2–10 seconds on GPU.

**Approximate time:** 3–5 minutes on GPU (including model load).

### Step 4: Full Dataset Run

```bash
# TinyLlama (default)
python -m scripts.run_full_dataset

# Phi-3-mini
python -m scripts.run_full_dataset --model phi3

# Llama-3.1-8B
python -m scripts.run_full_dataset --model llama8b
```

**What this does:** Processes all 2,153 qualifying abstracts from CSABSTRUCT (train + validation + test splits combined) with both attribution methods. For each abstract, it generates a hypothesis, runs Feature Ablation and Shapley Value Sampling, and saves comprehensive JSON per abstract plus a summary CSV.

**Output files:**

| File | Size | Content |
|------|------|---------|
| `summary.csv` | ~400KB | One row per abstract: all attribution scores, section word counts, timing, method agreement |
| `run_metadata.json` | ~1KB | Experiment config, aggregate statistics, total compute time |
| `feature_ablation/*.json` | ~270MB total | Complete per-abstract records with full text, labels, section info |
| `shapley/*.json` | ~270MB total | Same, plus all marginal contribution samples |

**Approximate times (NVIDIA A100 80GB):**

| Model | Time | Abstracts/second |
|-------|------|-----------------|
| TinyLlama-1.1B | 63 min | 0.57 |
| Phi-3-mini-4k | 53 min | 0.68 |
| Llama-3.1-8B | 91 min | 0.39 |

On consumer GPUs (RTX 3090/4090), expect roughly 2–3x these times. On CPU, the full run is impractical (~days).

### Step 5: Analysis and Figures

```bash
# Set thread limits if running on a login node with restricted resources
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

python -m scripts.run_analysis
```

**What this does:** Auto-detects all completed model runs, loads their summary CSVs, and produces:

- **7 figure types** (PDF + PNG): mean attribution bars, top-section frequency, FA vs Shapley scatter, violin distributions, length confound scatter, cross-model comparison, cross-model heatmap
- **2 LaTeX tables** ready to paste into a paper
- **Statistical tests**: Spearman/Kendall correlations, Friedman tests with p-values

The script works with any subset of models — if only TinyLlama has run, it produces single-model figures and skips cross-model comparisons.

**Customisation:**

```bash
# Analyse specific models only
python -m scripts.run_analysis --models tinyllama phi3

# Save figures to a custom directory
python -m scripts.run_analysis --output ./my_figures
```

## Understanding the Pipeline

### Data Loading

`src/data/load_csabstruct.py` loads the dataset from HuggingFace (`allenai/csabstruct`) and converts integer label IDs to human-readable names. Each abstract is represented as an `Abstract` dataclass containing the original sentences, their gold rhetorical labels, and a sections dictionary grouping sentences by label.

### Hypothesis Generation

For each abstract, we format the full text into a fixed prompt template and generate a hypothesis using greedy decoding (temperature=0). The prompt is:

> *"Read this scientific paper abstract and identify its main hypothesis. Be specific and concise."*

The same hypothesis is used as the fixed target for both attribution methods, ensuring they are compared on identical inputs.

### Value Function

The value function (Equation 1 in the paper) computes the sum of log-probabilities over the hypothesis tokens, conditioned on a given context:

```
v(context) = Σ log P(h_t | context, h_{<t})
```

The hypothesis tokens are fixed. Only the context changes between evaluations. A higher value means the context makes the hypothesis more likely.

### Feature Ablation

For each section, we remove its sentences from the abstract and measure how much the hypothesis log-probability drops:

```
attribution(section) = v(full_abstract) - v(abstract_without_section)
```

A larger drop means the section contributed more to generating the hypothesis. This requires n+1 forward passes (one baseline, one per section).

### Shapley Value Sampling

Shapley values estimate each section's contribution by averaging its marginal contribution across all possible orderings of sections. We use Monte Carlo sampling with 100 random permutations.

**Key optimisation:** With 5 sections, there are at most 2^5 = 32 unique coalitions (subsets of sections). We cache the value function result for each coalition, so 100 permutations × 5 sections = 500 evaluations reduce to at most 32 forward passes. This makes Shapley only 2.5x slower than Feature Ablation, not 100x.

## Outputs

Pre-computed results for all three models are included in `results/`. Each model directory contains:

- `summary.csv` — the primary analysis-ready file with one row per abstract
- `run_metadata.json` — experiment configuration and aggregate statistics

Raw per-abstract JSON files (~540MB per model) are not included in this repository. They are available on [Zenodo DOI: to be added].

### Summary CSV Columns

| Column | Description |
|--------|-------------|
| `abstract_id` | Unique identifier (split_index format) |
| `model` | Model key (tinyllama, phi3, llama8b) |
| `num_sentences` | Number of sentences in the abstract |
| `num_sections` | Number of distinct rhetorical sections |
| `sections_present` | Comma-separated list of section labels |
| `hypothesis_length` | Word count of generated hypothesis |
| `fa_baseline_lp` | Log-probability of hypothesis given full abstract |
| `fa_top_section` | Section with highest Feature Ablation score |
| `sh_top_section` | Section with highest Shapley value |
| `top_section_agree` | Whether FA and Shapley agree on top section |
| `fa_{section}` | Feature Ablation score for each section |
| `sh_{section}` | Shapley value for each section |
| `words_{section}` | Word count of each section |

## Models Used

| Model | Parameters | HuggingFace ID | License |
|-------|-----------|----------------|---------|
| TinyLlama-1.1B-Chat | 1.1B | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | Apache 2.0 |
| Phi-3-mini-4k-instruct | 3.8B | `microsoft/Phi-3-mini-4k-instruct` | MIT |
| Llama-3.1-8B-Instruct | 8B | `meta-llama/Llama-3.1-8B-Instruct` | Llama 3.1 Community |

All models were run in float16 precision on NVIDIA A100 80GB GPUs with greedy decoding (temperature=0, seed=42).

## Citation

```bibtex
@article{mahmud2026what,
  title={What Drives LLM-Generated Hypotheses? A Large-Scale Attribution 
         Analysis of Scientific Abstract Sections},
  author={Mahmud, Adnan and Abdel Rehim, Abbi and Reader, Gabriel and King, Ross},
  year={2026}
}
```

## Acknowledgements

We thank the **University of Edinburgh** for providing access to the Eddie Mark 3 HPC cluster (NVIDIA A100 80GB GPUs), which was used for all experiments reported in this work.

We also thank **AllenAI** for releasing the CSABSTRUCT dataset, which made this research possible.

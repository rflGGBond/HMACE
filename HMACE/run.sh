#!/bin/bash

# Configuration Parameters
GRAPHS="BA3000 WS3000 congress-Twitter"
TOTAL_BUDGET="110"
NUM_COMMUNITIES=16
MAX_GEN=20
T_COMM=4
MC_RUNS=10000
REPEATS=10

# LLM Configuration
# Options: local, openai
LLM_PROVIDER="local" 
# LLM_MODEL="claude-3-7-sonnet" 
# LLM_MODEL="Qwen2.5-7B-Instruct"
LLM_MODEL="Meta-Llama-3.1-8B-Instruct"
API_KEY="${OPENAI_API_KEY:-}"
BASE_URL="${OPENAI_BASE_URL:-}"
MODEL_ROOT="../../models"

if [ "$LLM_PROVIDER" = "openai" ] && [ -z "$API_KEY" ]; then
    echo "Error: OPENAI_API_KEY is required when LLM_PROVIDER=openai." >&2
    echo "Set it before running, for example: export OPENAI_API_KEY='<your-api-key>'" >&2
    echo "For OpenRouter or other compatible APIs, also set OPENAI_BASE_URL if needed." >&2
    exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# # Suppress tokenizers warning when forking
export TOKENIZERS_PARALLELISM=false

# Restrict to two empty GPUs (e.g. 1 and 2 or 1 and 3)
export CUDA_VISIBLE_DEVICES="2,3"

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "Running HMACE with:"
echo "GRAPHS=$GRAPHS"
echo "TOTAL_BUDGET=$TOTAL_BUDGET"
echo "NUM_COMMUNITIES=$NUM_COMMUNITIES"
echo "MAX_GEN=$MAX_GEN"
echo "T_COMM=$T_COMM"
echo "LLM_PROVIDER=$LLM_PROVIDER"
echo "LLM_MODEL=$LLM_MODEL"
if [ "$LLM_PROVIDER" = "openai" ]; then
    echo "OPENAI_API_KEY=<set>"
    echo "OPENAI_BASE_URL=${BASE_URL:-<default>}"
fi
echo "-----------------------------------"

# Run the python script
# We run 'run.py' directly since we are in HMACE directory, 
# but run.py handles imports by adding parent to sys.path
python3 run.py \
    --graphs $GRAPHS \
    --total_budget $TOTAL_BUDGET \
    --num_communities "$NUM_COMMUNITIES" \
    --max_gen "$MAX_GEN" \
    --t_comm "$T_COMM" \
    --mc_runs "$MC_RUNS" \
    --repeats "$REPEATS" \
    --llm_provider "$LLM_PROVIDER" \
    --llm_model "$LLM_MODEL" \
    --api_key "$API_KEY" \
    --base_url "$BASE_URL" \
    --model_root "$MODEL_ROOT"

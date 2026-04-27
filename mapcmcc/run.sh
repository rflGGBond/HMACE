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
# API_KEY="sk-WawUuKWJpbYJguipBd721182BfAa48D594A6Fc57839242F0" 
API_KEY="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"
# API_KEY="sk-524c07fb8b534c359fe3d2ce8cdc39c8"
# BASE_URL="https://aihubmix.com/v1"
BASE_URL="https://openrouter.ai/api/v1"
# BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_ROOT="../../models"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# # Suppress tokenizers warning when forking
export TOKENIZERS_PARALLELISM=false

# Restrict to two empty GPUs (e.g. 1 and 2 or 1 and 3)
export CUDA_VISIBLE_DEVICES="2,3"

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "Running MAPCMCC with:"
echo "GRAPHS=$GRAPHS"
echo "TOTAL_BUDGET=$TOTAL_BUDGET"
echo "NUM_COMMUNITIES=$NUM_COMMUNITIES"
echo "MAX_GEN=$MAX_GEN"
echo "T_COMM=$T_COMM"
echo "LLM_PROVIDER=$LLM_PROVIDER"
echo "LLM_MODEL=$LLM_MODEL"
echo "-----------------------------------"

# Run the python script
# We run 'run.py' directly since we are in mapcmcc directory, 
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
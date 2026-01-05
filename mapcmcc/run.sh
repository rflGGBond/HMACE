#!/bin/bash

# Configuration Parameters
GRAPHS="email-Eu-core"
TOTAL_BUDGET="20 110 200"
NUM_COMMUNITIES=16
MAX_GEN=20
T_COMM=4
MC_RUNS=100

# LLM Configuration
# Options: local, openai
LLM_PROVIDER="local" 
# LLM_MODEL="qwen2.5-7b-instruct" 
LLM_MODEL="Qwen2.5-7B-Instruct"
API_KEY="sk-524c07fb8b534c359fe3d2ce8cdc39c8" 
MODEL_ROOT="../../models"

# Suppress tokenizers warning when forking
export TOKENIZERS_PARALLELISM=false

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
    --llm_provider "$LLM_PROVIDER" \
    --llm_model "$LLM_MODEL" \
    --api_key "$API_KEY" \
    --model_root "$MODEL_ROOT"
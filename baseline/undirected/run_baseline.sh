#!/bin/bash

# Default parameters
K_VALUES="20 110 200"
REPEATS=1
MC_RUNS=100

# Default graphs
# Comment out graphs you don't want to run
UNDIRECTED_GRAPHS=""
# UNDIRECTED_GRAPHS="$UNDIRECTED_GRAPHS facebook"
# UNDIRECTED_GRAPHS="$UNDIRECTED_GRAPHS HR"
UNDIRECTED_GRAPHS="$UNDIRECTED_GRAPHS BA3000"
UNDIRECTED_GRAPHS="$UNDIRECTED_GRAPHS ER3000"
UNDIRECTED_GRAPHS="$UNDIRECTED_GRAPHS RG3000"
UNDIRECTED_GRAPHS="$UNDIRECTED_GRAPHS WS3000"

echo "Running Undirected Baseline Experiments"
echo "K Values: $K_VALUES"
echo "Repeats: $REPEATS"
echo "MC Runs: $MC_RUNS"
echo "Graphs: $UNDIRECTED_GRAPHS"
echo "--------------------------------"

if [ -n "$UNDIRECTED_GRAPHS" ]; then
    # Random
    echo "Running Random Baseline..."
    python3 random_select.py --k $K_VALUES --repeats $REPEATS --graphs $UNDIRECTED_GRAPHS --mc_runs $MC_RUNS
    
    # Max-Degree
    echo "Running Max-Degree Baseline..."
    python3 max_degree.py --k $K_VALUES --repeats $REPEATS --graphs $UNDIRECTED_GRAPHS --mc_runs $MC_RUNS
    
    # # CMIA-H
    # echo "Running CMIA-H Baseline..."
    # python3 cmia_h.py --k $K_VALUES --repeats $REPEATS --graphs $UNDIRECTED_GRAPHS --mc_runs $MC_RUNS

    # IBMM
    echo "Running IBMM Baseline..."
    python3 ibmm.py --k $K_VALUES --repeats $REPEATS --graphs $UNDIRECTED_GRAPHS --mc_runs $MC_RUNS

    # DDSE
    echo "Running DDSE Baseline..."
    python3 ddse.py --k $K_VALUES --repeats $REPEATS --graphs $UNDIRECTED_GRAPHS --mc_runs $MC_RUNS

    # SEA-PEA
    echo "Running SEA-PEA Baseline..."
    python3 sea_pea.py --k $K_VALUES --repeats $REPEATS --graphs $UNDIRECTED_GRAPHS --mc_runs $MC_RUNS

    # CELF
    echo "Running CELF Baseline..."
    python3 celf.py --k $K_VALUES --repeats $REPEATS --graphs $UNDIRECTED_GRAPHS --mc_runs $MC_RUNS
else
    echo "No graphs selected."
fi

echo "Undirected baseline experiments finished."

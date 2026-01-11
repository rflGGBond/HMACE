#!/bin/bash

# Default parameters
K_VALUES="20 110 200"
REPEATS=10
MC_RUNS=100

# Default graphs
# Comment out graphs you don't want to run
DIRECTED_GRAPHS=""
# DIRECTED_GRAPHS="$DIRECTED_GRAPHS email-Eu-core"
# DIRECTED_GRAPHS="$DIRECTED_GRAPHS p2p-Gnutella31"
# DIRECTED_GRAPHS="$DIRECTED_GRAPHS Email-EuAll"
DIRECTED_GRAPHS="$DIRECTED_GRAPHS soc-Epinions1"

echo "Running Directed Baseline Experiments"
echo "K Values: $K_VALUES"
echo "Repeats: $REPEATS"
echo "MC Runs: $MC_RUNS"
echo "Graphs: $DIRECTED_GRAPHS"
echo "--------------------------------"

if [ -n "$DIRECTED_GRAPHS" ]; then
    # # Random
    # echo "Running Random Baseline..."
    # python3 random_select.py --k $K_VALUES --repeats $REPEATS --graphs $DIRECTED_GRAPHS --mc_runs $MC_RUNS
    
    # # Max-Degree
    # echo "Running Max-Degree Baseline..."
    # python3 max_degree.py --k $K_VALUES --repeats $REPEATS --graphs $DIRECTED_GRAPHS --mc_runs $MC_RUNS
    
    # # CMIA-H
    # echo "Running CMIA-H Baseline..."
    # python3 cmia_h_gpu.py --k $K_VALUES --repeats $REPEATS --graphs $DIRECTED_GRAPHS --mc_runs $MC_RUNS

    # # IBMM
    # echo "Running IBMM Baseline..."
    # python3 ibmm.py --k $K_VALUES --repeats $REPEATS --graphs $DIRECTED_GRAPHS --mc_runs $MC_RUNS

    # DDSE
    echo "Running DDSE Baseline..."
    python3 ddse.py --k $K_VALUES --repeats $REPEATS --graphs $DIRECTED_GRAPHS --mc_runs $MC_RUNS

    # # SEA-PEA
    # echo "Running SEA-PEA Baseline..."
    # python3 sea_pea.py --k $K_VALUES --repeats $REPEATS --graphs $DIRECTED_GRAPHS --mc_runs $MC_RUNS

    # CELF
    echo "Running CELF Baseline..."
    python3 celf.py --k $K_VALUES --repeats $REPEATS --graphs $DIRECTED_GRAPHS --mc_runs $MC_RUNS
else
    echo "No graphs selected."
fi

echo "Directed baseline experiments finished."

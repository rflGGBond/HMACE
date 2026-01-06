#!/bin/bash

# Default parameters
K_VALUES="20 110 200"
REPEATS=1
MC_RUNS=100

# Default graphs for each script
# Comment out graphs you don't want to run
DIRECTED_GRAPHS=""
# DIRECTED_GRAPHS="$DIRECTED_GRAPHS email-Eu-core"
# DIRECTED_GRAPHS="$DIRECTED_GRAPHS Email-EuAll"
DIRECTED_GRAPHS="$DIRECTED_GRAPHS p2p-Gnutella31"
# DIRECTED_GRAPHS="$DIRECTED_GRAPHS soc-Epinions1"

UNDIRECTED_GRAPHS=""
# UNDIRECTED_GRAPHS="$UNDIRECTED_GRAPHS facebook"
UNDIRECTED_GRAPHS="$UNDIRECTED_GRAPHS HR"
# UNDIRECTED_GRAPHS="$UNDIRECTED_GRAPHS BA3000"
# UNDIRECTED_GRAPHS="$UNDIRECTED_GRAPHS ER3000"
# UNDIRECTED_GRAPHS="$UNDIRECTED_GRAPHS RG3000"
# UNDIRECTED_GRAPHS="$UNDIRECTED_GRAPHS WS3000"

# You can override these variables by setting environment variables
# Example: K_VALUES="50 100" REPEATS=5 ./run_baseline.sh

echo "Running Baseline Experiments"
echo "K Values: $K_VALUES"
echo "Repeats: $REPEATS"
echo "MC Runs: $MC_RUNS"
echo "--------------------------------"

# Run Directed Graphs
if [ -n "$DIRECTED_GRAPHS" ]; then
    echo "Starting Directed Graph Simulations..."
    echo "Graphs: $DIRECTED_GRAPHS"
    python3 run_directed.py --k $K_VALUES --repeats $REPEATS --graphs $DIRECTED_GRAPHS --mc_runs $MC_RUNS
    if [ $? -eq 0 ]; then
        echo "Directed simulations completed successfully."
    else
        echo "Directed simulations failed."
        exit 1
    fi
else
    echo "Skipping Directed Graph Simulations (No graphs selected)."
fi
echo "--------------------------------"

# Run Undirected Graphs
if [ -n "$UNDIRECTED_GRAPHS" ]; then
    echo "Starting Undirected Graph Simulations..."
    echo "Graphs: $UNDIRECTED_GRAPHS"
    python3 run_undirected.py --k $K_VALUES --repeats $REPEATS --graphs $UNDIRECTED_GRAPHS --mc_runs $MC_RUNS
    if [ $? -eq 0 ]; then
        echo "Undirected simulations completed successfully."
    else
        echo "Undirected simulations failed."
        exit 1
    fi
else
    echo "Skipping Undirected Graph Simulations (No graphs selected)."
fi

echo "All baseline experiments finished."

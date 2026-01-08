import os
import sys
import random
import time
import copy
import math
import networkx as nx
import matplotlib.pyplot as plt
import argparse
from datetime import datetime
import heapq

# Add path to import select_SN
sys.path.append('../')
try:
    from select_SN import select_SN
except ImportError:
    pass

class Logger(object):
    def __init__(self, stream=sys.stdout):
        output_dir = "../../results/logs/CELF/" 
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        current_date = datetime.now().strftime("%Y%m%d%H%M%S")
        log_name = f"log_{current_date}_celf_directed.txt"
        filename = os.path.join(output_dir, log_name)

        self.terminal = stream
        self.log = open(filename, 'a+')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def run_diffusion_model(G, S_P, S_N, model='COICM'):
    pos_activated = set(S_P)
    neg_activated = set(S_N)
    pos_frontier = set(S_P)
    neg_frontier = set(S_N)
    
    while pos_frontier or neg_frontier:
        next_pos_frontier = set()
        next_neg_frontier = set()
        potential_pos_activations = set()
        potential_neg_activations = set()
        
        for u in pos_frontier:
            for v in G.neighbors(u):
                if v not in pos_activated and v not in neg_activated:
                    weight = G[u][v]['weight']
                    prob = 1.0 if model == 'MCICM' else weight
                    if random.random() < prob:
                        potential_pos_activations.add(v)
        
        for u in neg_frontier:
            for v in G.neighbors(u):
                if v not in pos_activated and v not in neg_activated:
                    weight = G[u][v]['weight']
                    if random.random() < weight:
                        potential_neg_activations.add(v)
        
        for v in potential_pos_activations:
            pos_activated.add(v)
            next_pos_frontier.add(v)
            
        for v in potential_neg_activations:
            if v not in potential_pos_activations:
                neg_activated.add(v)
                next_neg_frontier.add(v)
        
        pos_frontier = next_pos_frontier
        neg_frontier = next_neg_frontier
        
    return len(neg_activated)

def monte_carlo_evaluation(G, S_P, S_N, model='COICM', runs=100):
    total_neg_activated = 0
    for _ in range(runs):
        total_neg_activated += run_diffusion_model(G, S_P, S_N, model)
    return total_neg_activated / runs

# --- CELF Implementation ---

def celf(G, k, SN, mc_runs_celf=100):
    """
    CELF (Cost-Effective Lazy Forward selection) algorithm.
    Objective: Maximize blocking effect (Minimize Negative Activation).
    Gain(u) = MC(S) - MC(S + {u})
    """
    start_time = time.time()
    
    # 1. Initial Marginal Gain Calculation
    # Calculate gain for each node individually
    # Baseline: No blocking seeds
    base_neg = monte_carlo_evaluation(G, [], SN, model='COICM', runs=mc_runs_celf)
    
    candidates = [n for n in G.nodes() if n not in SN]
    
    # Optimization: Only consider nodes reachable from SN or close to SN?
    # For speed, we might limit candidates to top degree nodes first?
    # But CELF is supposed to handle it.
    # To make it feasible, let's limit initial pool if graph is large.
    if len(candidates) > 500:
        # Pre-select top 500 by degree as pool
        candidates = sorted(candidates, key=lambda x: G.out_degree(x), reverse=True)[:500]
    
    gains = []
    print(f"  Calculating initial gains for {len(candidates)} candidates...")
    
    for i, u in enumerate(candidates):
        # Gain = Reduction in negative influence
        # MC([u])
        curr_neg = monte_carlo_evaluation(G, [u], SN, model='COICM', runs=mc_runs_celf)
        gain = base_neg - curr_neg
        # Use negative heap for max-priority queue
        heapq.heappush(gains, (-gain, u))
        
        if i % 50 == 0:
            print(f"    Processed {i}/{len(candidates)}")
            
    # 2. CELF Loop
    S = []
    current_neg = base_neg
    
    print("  Starting CELF loop...")
    while len(S) < k:
        matched = False
        while not matched:
            # Pop best candidate
            if not gains:
                break
                
            neg_gain, u = heapq.heappop(gains)
            gain = -neg_gain
            
            # Re-evaluate marginal gain
            # Gain(u) = MC(S) - MC(S + {u})
            # We already know MC(S) -> current_neg
            
            # Optimization: If S is empty, the gain is valid (from init)
            if len(S) == 0:
                matched = True
                S.append(u)
                current_neg = current_neg - gain # Approx update
                # Better to re-eval current_neg exactly or trust gain?
                # Let's re-eval to avoid drift
                current_neg = monte_carlo_evaluation(G, S, SN, model='COICM', runs=mc_runs_celf)
                print(f"    Selected {u}, Gain: {gain:.4f}, Current Neg: {current_neg:.4f}")
            else:
                # Check if u is already in S (shouldn't be)
                if u in S: continue
                
                # Check against top of heap (next best)
                if not gains:
                    S.append(u)
                    matched = True
                    break
                    
                # Peek next best old gain
                next_best_neg_gain, _ = gains[0]
                next_best_old_gain = -next_best_neg_gain
                
                # We need to recompute u's gain
                new_neg_u = monte_carlo_evaluation(G, S + [u], SN, model='COICM', runs=mc_runs_celf)
                new_gain = current_neg - new_neg_u
                
                # If new_gain >= next_best_old_gain, u is optimal!
                if new_gain >= next_best_old_gain:
                    matched = True
                    S.append(u)
                    current_neg = new_neg_u
                    print(f"    Selected {u}, New Gain: {new_gain:.4f}, Current Neg: {current_neg:.4f}")
                else:
                    # Push back with updated gain
                    heapq.heappush(gains, (-new_gain, u))
                    # Loop continues to pop next best
                    
    return S

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, nargs="+", default=[20, 110, 200])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--graphs", type=str, nargs="+", default=["email-Eu-core"])
    parser.add_argument("--mc_runs", type=int, default=100)
    args = parser.parse_args()

    SEED = 42
    random.seed(SEED)
    print(f"Random seed set to {SEED}")

    sys.stdout = Logger(sys.stdout)

    SN_size = 50
    SN_dic = {}
    graphs = args.graphs
    for g in graphs:
        SN_dic[g] = select_SN(g, SN_size)

    for file_name in graphs:
        G = nx.DiGraph()
        with open(f'../../graph/{file_name}.txt') as f:
            for line in f:
                n, m, w = line.split()
                n = int(n)
                m = int(m)
                w = float(w)
                G.add_edge(n, m, weight=w)

        nodes = list(G.nodes)
        SN = copy.deepcopy(SN_dic[file_name])
        
        k_values = args.k
        avg_neg_nodes_COICM = []
        avg_neg_nodes_MCICM = []

        # CELF is slow, so we reduce repeats if needed, but here we follow args
        # Also reduce internal MC runs for CELF optimization step to be faster?
        # args.mc_runs is for final eval. Let's use smaller for internal CELF.
        mc_runs_celf = 20 # Trade-off for speed
        
        for k in k_values:
            repeats = args.repeats
            current_k_coicm = []
            current_k_mcicm = []

            for r in range(repeats):
                current_seed = 42 + k + r * 1000
                random.seed(current_seed)
                try:
                    import numpy as np
                    np.random.seed(current_seed)
                except ImportError:
                    pass
                print(f"\nCELF: {file_name}, k={k}, run={r+1}/{repeats}")
                
                start_time = time.time()
                bestS = celf(G, k, SN, mc_runs_celf=mc_runs_celf)
                end_time = time.time()
                print(f"Time taken: {end_time - start_time:.2f}s")
                # print("Selected Seeds:", bestS)

                # Evaluate COICM
                print(f"Running Monte Carlo Evaluation (COICM)...")
                res_coicm = monte_carlo_evaluation(G, bestS, SN, model='COICM', runs=args.mc_runs)
                print(f"Negatively Activated Nodes (COICM): {res_coicm:.0f}")
                current_k_coicm.append(int(round(res_coicm)))

                # Evaluate MCICM
                print(f"Running Monte Carlo Evaluation (MCICM)...")
                res_mcicm = monte_carlo_evaluation(G, bestS, SN, model='MCICM', runs=args.mc_runs)
                print(f"Negatively Activated Nodes (MCICM): {res_mcicm:.0f}")
                current_k_mcicm.append(int(round(res_mcicm)))   

            avg_neg_nodes_COICM.append(int(round(sum(current_k_coicm) / len(current_k_coicm))))
            avg_neg_nodes_MCICM.append(int(round(sum(current_k_mcicm) / len(current_k_mcicm))))

        # Plot COICM
        try:
            output_fig_dir_coicm = f"../../results/COICM/CELF/repeats{args.repeats}_runs{args.mc_runs}"
            if not os.path.exists(output_fig_dir_coicm):
                os.makedirs(output_fig_dir_coicm)
            
            plt.figure(figsize=(6, 6))  
            plt.plot(k_values, avg_neg_nodes_COICM, marker='o', linestyle='--', label=file_name, color='salmon')
            for x, y in zip(k_values, avg_neg_nodes_COICM):
                plt.text(x, y, f'{y:.0f}', ha='center', va='bottom')
            plt.title(f'COICM CELF {file_name}')
            plt.xlabel('k')
            plt.ylabel('Negatively Activated Nodes')
            plt.xticks(k_values)
            plt.tight_layout()
            plt.savefig(os.path.join(output_fig_dir_coicm, f'COICM_{file_name}.png'))
            plt.close()
            print(f"Saved plot to {os.path.join(output_fig_dir_coicm, f'COICM_{file_name}.png')}")
        except Exception as e:
            print(f"Error plotting COICM: {e}")

        # Plot MCICM
        try:
            output_fig_dir_mcicm = f"../../results/MCICM/CELF/repeats{args.repeats}_runs{args.mc_runs}"
            if not os.path.exists(output_fig_dir_mcicm):
                os.makedirs(output_fig_dir_mcicm)
            
            plt.figure(figsize=(6, 6))
            plt.plot(k_values, avg_neg_nodes_MCICM, marker='o', linestyle='--', label=file_name, color='skyblue')
            for x, y in zip(k_values, avg_neg_nodes_MCICM):
                plt.text(x, y, f'{y:.0f}', ha='center', va='bottom')
            plt.title(f'MCICM CELF {file_name}')
            plt.xlabel('k')
            plt.ylabel('Negatively Activated Nodes')
            plt.xticks(k_values)
            plt.tight_layout()
            plt.savefig(os.path.join(output_fig_dir_mcicm, f'MCICM_{file_name}.png'))
            plt.close()
            print(f"Saved plot to {os.path.join(output_fig_dir_mcicm, f'MCICM_{file_name}.png')}")
        except Exception as e:
            print(f"Error plotting MCICM: {e}")

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
from collections import defaultdict

# Add path to import select_SN
sys.path.append('../')
try:
    from select_SN import select_SN
except ImportError:
    pass

class Logger(object):
    def __init__(self, stream=sys.stdout):
        output_dir = "../../results/logs/DDSE/" 
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        current_date = datetime.now().strftime("%Y%m%d%H%M%S")
        log_name = f"log_{current_date}_ddse_directed.txt"
        filename = os.path.join(output_dir, log_name)

        self.terminal = stream
        self.log = open(filename, 'a+')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def run_diffusion_model(G, S_P, S_N, model='COICM'):
    """
    Simulate the diffusion process where positive and negative information propagate simultaneously.
    """
    pos_activated = set(S_P)
    neg_activated = set(S_N)
    
    pos_frontier = set(S_P)
    neg_frontier = set(S_N)
    
    while pos_frontier or neg_frontier:
        next_pos_frontier = set()
        next_neg_frontier = set()
        
        # Potential activations for this step
        potential_pos_activations = set()
        potential_neg_activations = set()
        
        # 1. Determine all potential positive activations
        for u in pos_frontier:
            for v in G.neighbors(u):
                if v not in pos_activated and v not in neg_activated:
                    weight = G[u][v]['weight']
                    prob = 1.0 if model == 'MCICM' else weight
                    if random.random() < prob:
                        potential_pos_activations.add(v)
        
        # 2. Determine all potential negative activations
        for u in neg_frontier:
            for v in G.neighbors(u):
                if v not in pos_activated and v not in neg_activated:
                    weight = G[u][v]['weight']
                    if random.random() < weight:
                        potential_neg_activations.add(v)
        
        # 3. Resolve conflicts (Positive Priority)
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

# --- DDSE Implementation ---

def fitness_dpadv(seed, G, SN, relevant_nodes, hop=3):
    """
    Calculate DPADV (Degree-Penalty-Based Activation Degree Value).
    Measures expected negative influence spread (to be minimized).
    """
    effect_fc = 0
    ZP_fc = []
    ZN_fc = []
    ZP_fc.append(list(seed))
    ZN_fc.append(list(SN))
    
    for h in range(1, hop + 1):
        ZP_fc.append([])
        ZN_fc.append([])
        
    pP_fc = defaultdict(lambda: 0)
    apP_fc = defaultdict(lambda: 0)
    pN_fc = defaultdict(lambda: 0)
    apN_fc = defaultdict(lambda: 0)
    
    # Initialize probabilities
    for v in seed:
        pP_fc[v, 0] = 1
        for h in range(hop + 1):
            apP_fc[v, h] = 1
    for v in SN:
        pN_fc[v, 0] = 1
        for h in range(hop + 1):
            apN_fc[v, h] = 1
            
    # Iterative propagation
    for h in range(hop):
        temppP_fc = defaultdict(lambda: 1)
        temppN_fc = defaultdict(lambda: 1)

        # Positive Activation
        for v in ZP_fc[h]:
            # Directed: successors
            W_fc = list(G.successors(v))
            ZP_fc[h + 1] += W_fc
            for w in W_fc:
                temppP_fc[w] *= (1 - pP_fc[v, h] * G[v][w]['weight'])
        ZP_fc[h + 1] = list(set(ZP_fc[h + 1]))
        
        for v in ZP_fc[h + 1]:
            pP_fc[v, h + 1] = (1 - temppP_fc[v]) * (1 - apN_fc[v, h]) * (1 - apP_fc[v, h])
            for tau_f in range(h + 1, hop + 1):
                apP_fc[v, tau_f] = apP_fc[v, h] + pP_fc[v, h + 1]

        # Negative Activation
        for v in ZN_fc[h]:
            # Directed: successors
            W_fc = list(G.successors(v))
            ZN_fc[h + 1] += W_fc
            for w in W_fc:
                temppN_fc[w] *= (1 - pN_fc[v, h] * G[v][w]['weight'])
        ZN_fc[h + 1] = list(set(ZN_fc[h + 1]))
        
        for v in ZN_fc[h + 1]:
            pN_fc[v, h + 1] = temppP_fc[v] * (1 - temppN_fc[v]) * (1 - apN_fc[v, h]) * (1 - apP_fc[v, h])
            for tau_f in range(h + 1, hop + 1):
                apN_fc[v, tau_f] = apN_fc[v, h] + pN_fc[v, h + 1]
                
    # Sum over relevant nodes
    touched_nodes = set(apN_fc.keys())
    
    for u in touched_nodes:
        effect_fc += apN_fc[u, hop]
        
    return effect_fc

def generate_individual(sorted_nodes, k, i_idx, n_pop):
    """
    Generate i-th element of a vector using DDS logic.
    """
    individual = set()
    
    for j in range(k):
        window = int(k * (j + 5)) 
        if window > len(sorted_nodes):
            window = len(sorted_nodes)
        
        added = False
        for _ in range(10):
            idx = random.randint(0, window - 1)
            node = sorted_nodes[idx]
            if node not in individual:
                individual.add(node)
                added = True
                break
        
        if not added:
            for idx in range(len(sorted_nodes)):
                node = sorted_nodes[idx]
                if node not in individual:
                    individual.add(node)
                    break
                    
    return list(individual)

def ddse(G, k, SN, n_pop=20, g_max=10, mutation_prob=0.3, crossover_prob=0.5):
    """
    DDSE Algorithm with DPADV fitness.
    """
    nodes = list(G.nodes())
    sorted_nodes = sorted(nodes, key=lambda x: G.out_degree(x), reverse=True)
    
    relevant_nodes = nodes
    hop = 3
    
    # 1. Initialization
    population = [] 
    for _ in range(n_pop):
        ind = generate_individual(sorted_nodes, k, 0, n_pop)
        population.append(ind)
        
    # Main Loop
    for g in range(g_max):
        print(f"  Gen {g+1}/{g_max}")
        
        # 2. Mutation
        mutated_pop = []
        for i in range(n_pop):
            original = population[i]
            mutant = list(original)
            current_set = set(mutant)
            
            for j in range(k):
                if random.random() < mutation_prob:
                    window = int(k * (j + 5))
                    if window > len(sorted_nodes): window = len(sorted_nodes)
                    new_node = sorted_nodes[random.randint(0, window-1)]
                    
                    if new_node not in current_set:
                        mutant[j] = new_node
                        current_set.add(new_node)
            
            mutated_pop.append(mutant)
            
        # 3. Crossover
        crossover_pop = []
        for i in range(n_pop):
            parent = population[i]
            mutant = mutated_pop[i]
            child = []
            child_set = set()
            
            for j in range(k):
                if random.random() < crossover_prob:
                    val = mutant[j]
                else:
                    val = parent[j]
                    
                if val in child_set:
                    if val == mutant[j]: alt = parent[j]
                    else: alt = mutant[j]
                    
                    if alt not in child_set:
                        val = alt
                    else:
                         window = int(k * (j + 5))
                         while True:
                             idx = random.randint(0, min(window, len(sorted_nodes)-1))
                             cand = sorted_nodes[idx]
                             if cand not in child_set:
                                 val = cand
                                 break
                
                child.append(val)
                child_set.add(val)
            
            crossover_pop.append(child)
            
        # 4. Selection
        new_population = []
        for i in range(n_pop):
            orig = population[i]
            cross = crossover_pop[i]
            
            # Minimize DPADV
            fit_orig = fitness_dpadv(set(orig), G, SN, relevant_nodes, hop)
            fit_cross = fitness_dpadv(set(cross), G, SN, relevant_nodes, hop)
            
            if fit_cross < fit_orig:
                new_population.append(cross)
            else:
                new_population.append(orig)
                
        population = new_population
        
    # Final Selection
    best_ind = None
    best_fit = float('inf')
    
    for ind in population:
        fit = fitness_dpadv(set(ind), G, SN, relevant_nodes, hop)
        if fit < best_fit:
            best_fit = fit
            best_ind = ind
            
    # Local Search
    best_set = set(best_ind)
    improved = True
    while improved:
        improved = False
        current_list = list(best_set)
        
        for i in range(k):
            u = current_list[i]
            for v in G.neighbors(u):
                if v not in best_set:
                    new_set = best_set.copy()
                    new_set.remove(u)
                    new_set.add(v)
                    
                    new_fit = fitness_dpadv(new_set, G, SN, relevant_nodes, hop)
                    if new_fit < best_fit:
                        best_fit = new_fit
                        best_set = new_set
                        best_ind = list(best_set)
                        improved = True
                        break 
            if improved: break
            
    return list(best_set)


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
                print(f"\nDDSE: {file_name}, k={k}, run={r+1}/{repeats}")
                
                start_time = time.time()
                # Pass SN to DDSE
                bestS = ddse(G, k, SN)
                end_time = time.time()
                print(f"Time taken: {end_time - start_time:.2f}s")
                # print("Selected Seeds:", bestS)

                # Evaluate COICM
                print(f"Running Monte Carlo Evaluation (COICM)...")
                res_coicm = monte_carlo_evaluation(G, bestS, SN, model='COICM', runs=args.mc_runs)
                print(f"Negatively Activated Nodes (COICM): {res_coicm:.0f}")
                current_k_coicm.append(res_coicm)

                # Evaluate MCICM
                print(f"Running Monte Carlo Evaluation (MCICM)...")
                res_mcicm = monte_carlo_evaluation(G, bestS, SN, model='MCICM', runs=args.mc_runs)
                print(f"Negatively Activated Nodes (MCICM): {res_mcicm:.0f}")
                current_k_mcicm.append(res_mcicm)

            avg_neg_nodes_COICM.append(sum(current_k_coicm) / len(current_k_coicm))
            avg_neg_nodes_MCICM.append(sum(current_k_mcicm) / len(current_k_mcicm))

        # Plot COICM
        try:
            output_fig_dir_coicm = f"../../results/COICM/DDSE/repeats{args.repeats}_runs{args.mc_runs}"
            if not os.path.exists(output_fig_dir_coicm):
                os.makedirs(output_fig_dir_coicm)
            
            plt.figure(figsize=(6, 6))
            plt.plot(k_values, avg_neg_nodes_COICM, marker='o', linestyle='--', label=file_name, color='salmon')
            for x, y in zip(k_values, avg_neg_nodes_COICM):
                plt.text(x, y, f'{y:.0f}', ha='center', va='bottom')
            plt.title(f'COICM DDSE {file_name}')
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
            output_fig_dir_mcicm = f"../../results/MCICM/DDSE/repeats{args.repeats}_runs{args.mc_runs}"
            if not os.path.exists(output_fig_dir_mcicm):
                os.makedirs(output_fig_dir_mcicm)
            
            plt.figure(figsize=(6, 6))
            plt.plot(k_values, avg_neg_nodes_MCICM, marker='o', linestyle='--', label=file_name, color='skyblue')
            for x, y in zip(k_values, avg_neg_nodes_MCICM):
                plt.text(x, y, f'{y:.0f}', ha='center', va='bottom')
            plt.title(f'MCICM DDSE {file_name}')
            plt.xlabel('k')
            plt.ylabel('Negatively Activated Nodes')
            plt.xticks(k_values)
            plt.tight_layout()
            plt.savefig(os.path.join(output_fig_dir_mcicm, f'MCICM_{file_name}.png'))
            plt.close()
            print(f"Saved plot to {os.path.join(output_fig_dir_mcicm, f'MCICM_{file_name}.png')}")
        except Exception as e:
            print(f"Error plotting MCICM: {e}")

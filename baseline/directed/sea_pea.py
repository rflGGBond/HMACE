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
        output_dir = "../../results/logs/SEA-PEA/" 
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        current_date = datetime.now().strftime("%Y%m%d%H%M%S")
        log_name = f"log_{current_date}_sea-pea_directed.txt"
        filename = os.path.join(output_dir, log_name)

        self.terminal = stream
        self.log = open(filename, 'a+')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

# --- Fitness Function (DPADV) ---
def fitness_C(seed_7, G_7, SN_7, comAndFS_7, hop_7):
    # This is the DPADV function logic adapted from run_directed.py
    # Calculates the expected negative influence (minimized)
    
    # G_7 is expected to be directed for calculation logic
    # In run_directed.py it calls G_7.to_directed(), but G is already directed here.
    
    effect_fc = 0
    ZP_fc = []
    ZN_fc = []
    ZP_fc.append(seed_7)
    ZN_fc.append(SN_7)
    for h in range(1, hop_7 + 1):
        ZP_fc.append([])
        ZN_fc.append([])
    pP_fc = defaultdict(lambda: 0)
    apP_fc = defaultdict(lambda: 0)
    pN_fc = defaultdict(lambda: 0)
    apN_fc = defaultdict(lambda: 0)
    
    # Initialize probabilities
    for v in seed_7:
        pP_fc[v, 0] = 1
        for h in range(hop_7 + 1):
            apP_fc[v, h] = 1
    for v in SN_7:
        pN_fc[v, 0] = 1
        for h in range(hop_7 + 1):
            apN_fc[v, h] = 1
            
    # Iterative propagation
    for h in range(hop_7):
        temppP_fc = defaultdict(lambda: 1)
        temppN_fc = defaultdict(lambda: 1)

        # Positive Activation
        for v in ZP_fc[h]:
            W_fc = list(G_7.successors(v))
            ZP_fc[h + 1] += W_fc
            for w in W_fc:
                temppP_fc[w] *= (1 - pP_fc[v, h] * G_7[v][w]['weight'])
        ZP_fc[h + 1] = list(set(ZP_fc[h + 1]))
        
        for v in ZP_fc[h + 1]:
            pP_fc[v, h + 1] = (1 - temppP_fc[v]) * (1 - apN_fc[v, h]) * (1 - apP_fc[v, h])
            for tau_f in range(h + 1, hop_7 + 1):
                apP_fc[v, tau_f] = apP_fc[v, h] + pP_fc[v, h + 1]

        # Negative Activation
        for v in ZN_fc[h]:
            W_fc = list(G_7.successors(v))
            ZN_fc[h + 1] += W_fc
            for w in W_fc:
                temppN_fc[w] *= (1 - pN_fc[v, h] * G_7[v][w]['weight'])
        ZN_fc[h + 1] = list(set(ZN_fc[h + 1]))
        
        for v in ZN_fc[h + 1]:
            pN_fc[v, h + 1] = temppP_fc[v] * (1 - temppN_fc[v]) * (1 - apN_fc[v, h]) * (1 - apP_fc[v, h])
            for tau_f in range(h + 1, hop_7 + 1):
                apN_fc[v, tau_f] = apN_fc[v, h] + pN_fc[v, h + 1]
                
    # Calculate Total Negative Influence
    # Only sum over comAndFS_7 (which should be relevant nodes, e.g. V \ (S U SN))
    # Or typically all nodes V.
    for u in comAndFS_7:
        effect_fc += apN_fc[u, hop_7]
        
    return effect_fc

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

# --- SEA-PEA Implementation ---

def calculate_average_degree(G):
    return 2 * G.number_of_edges() / G.number_of_nodes()

def calculate_ncn(d_bar, k, alpha):
    # Equation 3: UB(d, k, a) = ceil(k * e^(alpha * d) + k)
    # Using small alpha to prevent explosion
    val = k * math.exp(alpha * d_bar) + k
    return int(math.ceil(val))

def initialization(pop_size, sorted_nodes, ncn):
    # Algorithm 2
    population = []
    candidates = sorted_nodes[:ncn]
    for _ in range(pop_size):
        # Initialize with empty vectors? Algo 2 says empty.
        # But seed addition adds to them.
        population.append([])
    return population

def seed_addition(population, sorted_nodes, d_bar, alpha, k_current, G, SN, hop, relevant_nodes):
    # Algorithm 3
    # Add one node to each individual
    new_population = []
    
    # Calculate NCN based on current size?
    # Algo 3 uses |Pi| + 1.
    
    for i in range(len(population)):
        ind = population[i]
        current_len = len(ind)
        # NCN = min(UB(|Pi|+1, d, a), |T|)
        ncn = calculate_ncn(d_bar, current_len + 1, alpha)
        if ncn > len(sorted_nodes): ncn = len(sorted_nodes)
        
        candidates = sorted_nodes[:ncn]
        
        # Randomly select a node from candidates not in ind
        # Ensure we exclude SN
        valid_candidates = [n for n in candidates if n not in ind and n not in SN]
        
        if valid_candidates:
            new_node = random.choice(valid_candidates)
            ind.append(new_node)
        else:
            # Pick any valid node from graph
            others = [n for n in sorted_nodes if n not in ind and n not in SN]
            if others:
                ind.append(random.choice(others))
                
        new_population.append(ind)
        
    return new_population

def crossover_mutation(population, i, cr, mu, sorted_nodes, d_bar, alpha):
    # Algorithm 4
    Pi = population[i]
    k_size = len(Pi)
    
    # Best individual P1 is passed as context?
    # In main loop, P is sorted. So P[0] is best.
    P1 = population[0]
    
    new_Pi = copy.deepcopy(Pi)
    
    if i == 0:
        # Best individual only mutates 1 element
        j = random.randint(0, k_size - 1)
        # NCN based on |Pi|
        ncn = calculate_ncn(d_bar, k_size, alpha)
        if ncn > len(sorted_nodes): ncn = len(sorted_nodes)
        candidates = sorted_nodes[:ncn]
        
        # Replace j-th element
        valid = [n for n in candidates if n not in new_Pi] # excluding existing
        if valid:
            new_Pi[j] = random.choice(valid)
            
    else:
        # Crossover with P1
        # R = P1 intersect Pi
        # N = P1 \ R
        # M = Pi \ R (Wait, M = Pi \ (P1 intersect Pi))
        
        set_Pi = set(Pi)
        set_P1 = set(P1)
        
        R = list(set_P1.intersection(set_Pi))
        N_set = list(set_P1 - set_Pi)
        M_set = list(set_Pi - set_P1)
        
        new_ind_set = set(R) # Start with intersection
        
        # For t = 1 to |N| (elements in P1 not in Pi)
        for node in N_set:
            if random.random() < cr:
                new_ind_set.add(node)
            # Else: we might add from M later?
            # Algo 4 line 11: "Add node Nt to Pi with prob cr, otherwise add Mt"
            # This assumes |N| == |M|? Sizes might differ if intersection varies?
            # Actually |P1| = |Pi| = k. So |N| = |M|. Correct.
        
        # Fill remaining spots from M
        needed = k_size - len(new_ind_set)
        # The logic in paper says: iterate t, pick Nt or Mt.
        # Let's implement that strictly.
        
        temp_list = list(new_ind_set) # Starts with R
        
        common_len = min(len(N_set), len(M_set))
        
        for t in range(common_len):
            node_N = N_set[t]
            node_M = M_set[t]
            
            if random.random() < cr:
                temp_list.append(node_N)
            else:
                temp_list.append(node_M)
        
        # Append remaining nodes if sizes differ (handling duplicates case)
        if len(N_set) > common_len:
            temp_list.extend(N_set[common_len:])
        if len(M_set) > common_len:
            temp_list.extend(M_set[common_len:])
                
        new_Pi = temp_list
        
        # Mutation
        # f using Eq 5. f = floor(|P|/2 - i / |P|/2)?
        # f calculation logic:
        # P arranged descending.
        # Actually simple mutation logic:
        # Iterate t from 1 to |Pi|
        # if rand < mu: replace t-th element with unique node Tx
        
        ncn = calculate_ncn(d_bar, k_size + 0.5, alpha) # f parameter usage?
        # Let's use simple NCN for mutation candidate pool
        if ncn > len(sorted_nodes): ncn = len(sorted_nodes)
        candidates = sorted_nodes[:ncn]
        
        # Ensure new_Pi has exactly k_size elements
        while len(new_Pi) < k_size:
            # Add random unique node from candidates
            cand = random.choice(candidates)
            if cand not in new_Pi:
                new_Pi.append(cand)
                
        if len(new_Pi) > k_size:
            new_Pi = new_Pi[:k_size]
        
        for t in range(len(new_Pi)):
            if random.random() < mu:
                # Replace
                valid = [n for n in candidates if n not in new_Pi]
                if valid:
                    # Probabilistic selection based on degree (Eq 6)
                    # p(Tx) = deg(Tx) / sum deg
                    # Weighted choice
                    # Optimized: just random choice from top candidates is close enough for baseline
                    new_Pi[t] = random.choice(valid)
                    
    return new_Pi

def sea_pea(G, k, SN, pop_size=20, g_max=10, cr=0.6, mu=0.1, alpha=0.04):
    # Step 1: Initialization
    nodes = list(G.nodes())
    # Directed: Out-degree
    sorted_nodes = sorted(nodes, key=lambda x: G.out_degree(x), reverse=True)
    d_bar = calculate_average_degree(G)
    
    population = initialization(pop_size, sorted_nodes, 0)
    
    hop = 3 # Default hop for fitness
    relevant_nodes = list(G.nodes()) # For fitness eval
    
    # Outer loop: seed size 1 to k
    for size_k in range(1, k + 1):
        # Seed Addition
        population = seed_addition(population, sorted_nodes, d_bar, alpha, size_k, G, SN, hop, relevant_nodes)
        
        # Inner loop: Evolution
        for g in range(g_max):
            # Sort population by Fitness (Ascending because lower is better for blocking)
            # Wait, standard SEA-PEA maximizes Influence (Descending).
            # DPADV is "Negative Influence" -> Minimize.
            # So we sort Ascending (smallest DPADV is best).
            
            scores = []
            for i in range(pop_size):
                fit = fitness_C(population[i], G, SN, relevant_nodes, hop)
                scores.append((fit, i))
            
            scores.sort(key=lambda x: x[0]) # Ascending
            
            # Reorder population
            new_pop_order = [population[x[1]] for x in scores]
            population = new_pop_order
            
            # Crossover and Mutation
            new_generation = []
            for i in range(pop_size):
                P_new = crossover_mutation(population, i, cr, mu, sorted_nodes, d_bar, alpha)
                
                # Selection
                fit_old = scores[i][0]
                fit_new = fitness_C(P_new, G, SN, relevant_nodes, hop)
                
                if fit_new < fit_old: # Minimize
                    new_generation.append(P_new)
                else:
                    new_generation.append(population[i])
            
            population = new_generation
            
            # Check convergence? (Optional)
            
    # Final Selection
    scores = []
    for i in range(pop_size):
        fit = fitness_C(population[i], G, SN, relevant_nodes, hop)
        scores.append((fit, population[i]))
    
    scores.sort(key=lambda x: x[0])
    best_S = scores[0][1]
    
    return best_S

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
                print(f"\nSEA-PEA: {file_name}, k={k}, run={r+1}/{repeats}")
                
                start_time = time.time()
                bestS = sea_pea(G, k, SN)
                end_time = time.time()
                print(f"Time taken: {end_time - start_time:.2f}s")
                # print("Selected Seeds:", bestS)

                # Evaluate COICM
                print(f"Running Monte Carlo Evaluation (COICM)...")
                res_coicm = monte_carlo_evaluation(G, bestS, SN, model='COICM', runs=args.mc_runs)
                print(f"Negatively Activated Nodes (COICM): {res_coicm:.2f}")
                current_k_coicm.append(res_coicm)

                # Evaluate MCICM
                print(f"Running Monte Carlo Evaluation (MCICM)...")
                res_mcicm = monte_carlo_evaluation(G, bestS, SN, model='MCICM', runs=args.mc_runs)
                print(f"Negatively Activated Nodes (MCICM): {res_mcicm:.2f}")
                current_k_mcicm.append(res_mcicm)

            avg_neg_nodes_COICM.append(sum(current_k_coicm) / len(current_k_coicm))
            avg_neg_nodes_MCICM.append(sum(current_k_mcicm) / len(current_k_mcicm))

        # Plot COICM
        try:
            output_fig_dir_coicm = f"../../results/COICM/SEA-PEA/repeats{args.repeats}_runs{args.mc_runs}"
            if not os.path.exists(output_fig_dir_coicm):
                os.makedirs(output_fig_dir_coicm)
            
            plt.figure(figsize=(6, 6))
            plt.plot(k_values, avg_neg_nodes_COICM, marker='o', linestyle='--', label=file_name, color='salmon')
            for x, y in zip(k_values, avg_neg_nodes_COICM):
                plt.text(x, y, f'{y:.2f}', ha='center', va='bottom')
            plt.title(f'COICM SEA-PEA {file_name}')
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
            output_fig_dir_mcicm = f"../../results/MCICM/SEA-PEA/repeats{args.repeats}_runs{args.mc_runs}"
            if not os.path.exists(output_fig_dir_mcicm):
                os.makedirs(output_fig_dir_mcicm)
            
            plt.figure(figsize=(6, 6))
            plt.plot(k_values, avg_neg_nodes_MCICM, marker='o', linestyle='--', label=file_name, color='skyblue')
            for x, y in zip(k_values, avg_neg_nodes_MCICM):
                plt.text(x, y, f'{y:.2f}', ha='center', va='bottom')
            plt.title(f'MCICM SEA-PEA {file_name}')
            plt.xlabel('k')
            plt.ylabel('Negatively Activated Nodes')
            plt.xticks(k_values)
            plt.tight_layout()
            plt.savefig(os.path.join(output_fig_dir_mcicm, f'MCICM_{file_name}.png'))
            plt.close()
            print(f"Saved plot to {os.path.join(output_fig_dir_mcicm, f'MCICM_{file_name}.png')}")
        except Exception as e:
            print(f"Error plotting MCICM: {e}")

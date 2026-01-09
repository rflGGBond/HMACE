import os
import sys
import random
import time
import copy
import math
import heapq
import networkx as nx
import matplotlib.pyplot as plt
import argparse
from datetime import datetime

# Add path to import select_SN and monte_carlo_evaluation
sys.path.append('../')
try:
    from select_SN import select_SN
except ImportError:
    pass

class Logger(object):
    def __init__(self, stream=sys.stdout):
        output_dir = "../../results/logs/IBMM/" 
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        current_date = datetime.now().strftime("%Y%m%d%H%M%S")
        log_name = f"log_{current_date}_ibmm_directed.txt"
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

# --- IBMM Implementation ---

def generate_modified_rr_set(G, R_set):
    """
    Generate a single modified RR set for IBMM.
    Definition 2:
    1. Pick r uniformly from V \ R.
    2. Sample graph g from G (reverse BFS traversal logic).
    3. RR set contains u if d_g(u, r) <= d_g(R, r).
    """
    nodes = list(G.nodes())
    candidates = [n for n in nodes if n not in R_set]
    if not candidates:
        return set()
    
    r = random.choice(candidates)
    
    # We perform a BFS on the REVERSE graph starting from r
    # But we sample edges as we traverse.
    # Since G is nx.DiGraph, predecessors gives incoming edges.
    
    # Distance map in sampled graph g: u -> distance to r
    dist_to_r = {r: 0}
    queue = [r]
    
    # Track nodes visited in the sampled reverse traversal
    visited = {r}
    
    # We also need to know d_g(R, r)
    min_dist_R = float('inf')
    
    # If r itself is in R (not possible by selection), dist would be 0.
    
    while queue:
        curr = queue.pop(0)
        curr_dist = dist_to_r[curr]
        
        # If we have already found R nodes at closer distance, 
        # do we need to continue?
        # We need to find ALL u such that d(u, r) <= d(R, r).
        # So we can stop exploring if curr_dist >= min_dist_R?
        # Yes, because any further node v -> curr will have dist > curr_dist >= min_dist_R.
        # Wait, if curr_dist == min_dist_R, we still might find other u with same distance.
        # So stop if curr_dist > min_dist_R.
        
        if curr_dist >= min_dist_R and min_dist_R != float('inf'):
             # We can treat nodes at this level, but don't expand further?
             # Actually, if curr_dist == min_dist_R, we process curr, but neighbors will be dist+1.
             # So neighbors will be > min_dist_R.
             pass
        
        # Reverse traversal: look at predecessors (incoming edges in G)
        for u in G.predecessors(curr):
            if u not in visited:
                # Sample edge u -> curr
                weight = G[u][curr]['weight']
                if random.random() < weight:
                    visited.add(u)
                    dist_to_r[u] = curr_dist + 1
                    queue.append(u)
                    
                    if u in R_set:
                        if dist_to_r[u] < min_dist_R:
                            min_dist_R = dist_to_r[u]
    
    # Construct Modified RR Set
    # {u | d_g(u, r) <= d_g(R, r)}
    rr_set = set()
    for u, d in dist_to_r.items():
        if d <= min_dist_R:
            rr_set.add(u)
            
    # Algo 1 Step 5: "If O_i contains at least one node in R"
    # This means min_dist_R must be finite.
    if min_dist_R == float('inf'):
        return set() # Discard
        
    return rr_set

def lower_bound_estimation(G, k, R_set):
    """
    Algorithm 2: LowerBoundEstimation
    Estimates a lower bound for OPT using 2-hop local influence.
    """
    R_list = list(R_set)
    Q1 = set()
    for r in R_list:
        for v in G.neighbors(r):
            if v not in R_set:
                Q1.add(v)
    
    Q2 = set()
    for u in Q1:
        for v in G.neighbors(u):
            if v not in R_set and v not in Q1:
                Q2.add(v)
                
    relevant_nodes = Q1.union(Q2)
    
    # Calculate score1(u) for u in Q1 U Q2
    # Influence from R to u through paths length < 3
    # Length 1: r -> u
    # Length 2: r -> v -> u
    
    score1 = {u: 0.0 for u in relevant_nodes}
    
    # Pre-calculate 1-hop influence from R
    for r in R_list:
        for v in G.neighbors(r):
            if v in relevant_nodes:
                score1[v] += G[r][v]['weight'] # Assuming linear summation as approximation for expectation?
                # Or standard IC prob: 1 - prod(1-p).
                # Since "paths less than 3", usually implies disjoint paths sum for small probs.
                # Let's use summation as it's a heuristic score.

    inf1 = sum(score1.values())
    
    # Calculate score2(v) for v in Q1
    # score2(v) = score1(v) + sum(score1(u) * p_vu) for u in neighbors(v) \ R
    score2 = {v: 0.0 for v in Q1}
    for v in Q1:
        s1 = score1[v]
        sum_neighbors = 0.0
        for u in G.neighbors(v):
            if u not in R_set and u in relevant_nodes: # u should be in Q1 or Q2
                sum_neighbors += score1[u] * G[v][u]['weight']
        score2[v] = s1 + sum_neighbors
        
    # Sort nodes based on score2 and pick top k
    # Only pick from Q1 as candidates? The algo says "for each v in Q1 do ... Sort nodes based on score2".
    # And "S = top k nodes based on score2".
    sorted_candidates = sorted(list(Q1), key=lambda x: score2.get(x, 0.0), reverse=True)
    S_star = set(sorted_candidates[:k])
    
    # Calculate inf2
    # Influence from R to u through paths length < 3 in PRESENCE of S_star
    # If a node in path is in S_star, it blocks.
    
    # Paths blocked:
    # 1. r -> u where u in S_star (u blocked)
    # 2. r -> v -> u where v in S_star or u in S_star
    
    inf2 = 0.0
    for u in relevant_nodes:
        if u in S_star:
            continue # Blocked, influence is 0 (assuming S_star blocks incoming neg influence)
        
        # Influence from R to u
        # 1-hop
        prob_1hop = 0.0
        for r in R_list:
            if u in G[r]: # r -> u exists
                prob_1hop += G[r][u]['weight']
        
        # 2-hop
        prob_2hop = 0.0
        # r -> v -> u
        # We iterate v in in-neighbors of u
        for v in G.predecessors(u):
            if v in S_star: continue # Blocked path
            if v in R_set: continue # Handled in 1-hop
            
            # v must have incoming from R
            # Check v's incoming from R
            for r in R_list:
                if v in G[r]:
                     prob_2hop += G[r][v]['weight'] * G[v][u]['weight']

        inf2 += prob_1hop + prob_2hop

    LB = inf1 - inf2
    return max(LB, 1e-6) # Avoid zero division

def ibmm(G, R_set, k):
    n = G.number_of_nodes()
    # Parameters for N calculation (simplified based on paper logic)
    # We need LB from Algo 2
    LB = lower_bound_estimation(G, k, R_set)
    
    # Equation 16: lambda_m
    epsilon = 0.5 # Default from paper figs
    alpha = math.sqrt(math.log(n) + math.log(2))
    beta = math.sqrt((1 - 1/math.e) * (math.log(n) + math.log(2)))
    
    lambda_m = 2 * n * ((1 - 1/math.e) * alpha + beta)**2 * (epsilon**-2)
    
    N = int(lambda_m / LB)
    
    # Cap N to reasonable limits for runtime if too large
    if N > 2000: 
        print(f"  Estimated N={N} too high, capping at 2000")
        N = 2000
    if N < 100:
        N = 100
        
    print(f"  Estimated N: {N}, LB: {LB:.4f}")
    
    # Generate RR sets
    Theta_R = []
    for _ in range(N):
        rr = generate_modified_rr_set(G, R_set)
        if rr:
            Theta_R.append(rr)
            
    # Greedy Selection (Max Coverage)
    S_star = set()
    # Track coverage: index of RR sets covered
    covered_indices = set()
    
    # Convert list of sets to list of (index, set) for efficiency?
    # Or map element -> list of set_indices
    element_to_sets = {}
    for idx, rr in enumerate(Theta_R):
        for node in rr:
            if node not in R_set: # Candidates V \ R
                if node not in element_to_sets:
                    element_to_sets[node] = set()
                element_to_sets[node].add(idx)
                
    for _ in range(k):
        best_node = None
        max_gain = -1
        
        # Only check candidates that appear in some RR sets
        candidates = list(element_to_sets.keys())
        
        for v in candidates:
            if v in S_star: continue
            
            # Gain = number of NEW sets covered by v
            v_sets = element_to_sets[v]
            gain = len(v_sets - covered_indices)
            
            if gain > max_gain:
                max_gain = gain
                best_node = v
        
        if best_node is not None and max_gain > 0:
            S_star.add(best_node)
            covered_indices.update(element_to_sets[best_node])
        else:
            # If no gain, pick random remaining?
            remaining = [x for x in candidates if x not in S_star]
            if not remaining: 
                # Pick any from V \ R
                others = [x for x in G.nodes() if x not in R_set and x not in S_star]
                if others:
                    S_star.add(random.choice(others))
            else:
                S_star.add(random.choice(remaining))
                
    return list(S_star)


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

        for k in k_values:
            repeats = args.repeats
            current_k_coicm = []

            for r in range(repeats):
                current_seed = 42 + k + r * 1000
                random.seed(current_seed)
                try:
                    import numpy as np
                    np.random.seed(current_seed)
                except ImportError:
                    pass
                print(f"\nIBMM: {file_name}, k={k}, run={r+1}/{repeats}")
                
                start_time = time.time()
                bestS = ibmm(G, SN, k)
                end_time = time.time()
                print(f"Time taken: {end_time - start_time:.2f}s")
                # print("Selected Seeds:", bestS)

                # Evaluate COICM
                print(f"Running Monte Carlo Evaluation (COICM)...")
                res_coicm = monte_carlo_evaluation(G, bestS, SN, model='COICM', runs=args.mc_runs)
                print(f"Negatively Activated Nodes (COICM): {res_coicm:.0f}")
                current_k_coicm.append(res_coicm)
            
            avg_neg_nodes_COICM.append(sum(current_k_coicm) / len(current_k_coicm))

        # Plot COICM
        try:
            output_fig_dir_coicm = f"../../results/COICM/IBMM/repeats{args.repeats}_runs{args.mc_runs}"
            if not os.path.exists(output_fig_dir_coicm):
                os.makedirs(output_fig_dir_coicm)
            
            plt.figure(figsize=(6, 6))
            plt.plot(k_values, avg_neg_nodes_COICM, marker='o', linestyle='--', label=file_name, color='salmon')
            for x, y in zip(k_values, avg_neg_nodes_COICM):
                plt.text(x, y, f'{y:.0f}', ha='center', va='bottom')
            plt.title(f'COICM IBMM {file_name}')
            plt.xlabel('k')
            plt.ylabel('Negatively Activated Nodes')
            plt.xticks(k_values)
            plt.tight_layout()
            plt.savefig(os.path.join(output_fig_dir_coicm, f'COICM_{file_name}.png'))
            plt.close()
            print(f"Saved plot to {os.path.join(output_fig_dir_coicm, f'COICM_{file_name}.png')}")
        except Exception as e:
            print(f"Error plotting COICM: {e}")

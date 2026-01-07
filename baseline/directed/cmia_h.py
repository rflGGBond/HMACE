import os, sys
import time
import copy
import random
import heapq
import math
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime
import argparse

# Add path to import select_SN
sys.path.append('../')
from select_SN import select_SN

class Logger(object):
    def __init__(self, stream=sys.stdout):
        output_dir = "../../results/logs/CMIA-H/" 
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        current_date = datetime.now().strftime("%Y%m%d%H%M%S")
        log_name = f"log_{current_date}_cmia-h_directed.txt"
        filename = os.path.join(output_dir, log_name)

        self.terminal = stream
        self.log = open(filename, 'a+')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def run_diffusion_model(G, S_P, S_N, model='MCICM'):
    """
    Simulate the diffusion process for MCICM.
    (Directed Graph Version)
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
            # G.neighbors(u) in DiGraph returns successors (out-neighbors)
            for v in G.neighbors(u):
                if v not in pos_activated and v not in neg_activated:
                    # In MCICM, positive prob is 1.0
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

def monte_carlo_evaluation(G, S_P, S_N, model='MCICM', runs=100):
    total_neg_activated = 0
    for _ in range(runs):
        total_neg_activated += run_diffusion_model(G, S_P, S_N, model)
    return total_neg_activated / runs

# --- CMIA-H Implementation ---

def get_mia_structure(G, u, theta, is_in=True):
    """
    Construct MIIA (is_in=True) or MIOA (is_in=False) for node u in Directed Graph.
    Returns: 
        nodes_in_mia: Set of nodes in the structure.
        dist: Dictionary of -log(prob) distances.
    """
    # Dijkstra
    dist = {u: 0}
    pq = [(0, u)]
    nodes_in_mia = {u}
    
    # -log(theta) is the max distance
    max_dist = -math.log(theta) if theta > 0 else float('inf')

    while pq:
        d, curr = heapq.heappop(pq)
        
        if d > dist.get(curr, float('inf')):
            continue
        
        # For Directed Graph:
        # MIIA(u): Paths TO u. Traverse incoming edges (predecessors).
        # MIOA(u): Paths FROM u. Traverse outgoing edges (successors/neighbors).
        
        if is_in:
            # Traverse reverse edges: v -> curr
            # NetworkX predecessors gives incoming neighbors
            neighbors = G.predecessors(curr)
        else:
            neighbors = G.neighbors(curr)
        
        for neighbor in neighbors:
            # Weight:
            # If is_in (v->curr): weight is G[neighbor][curr]['weight']
            # If !is_in (curr->v): weight is G[curr][neighbor]['weight']
            
            if is_in:
                weight = G[neighbor][curr]['weight']
            else:
                weight = G[curr][neighbor]['weight']
                
            if weight <= 0: continue
            
            new_dist = d - math.log(weight)
            
            if new_dist <= max_dist:
                if new_dist < dist.get(neighbor, float('inf')):
                    dist[neighbor] = new_dist
                    heapq.heappush(pq, (new_dist, neighbor))
                    nodes_in_mia.add(neighbor)
    
    return nodes_in_mia, dist

def compute_ap(u, S_N, MIIA_nodes, MIIA_dists, d_c, G):
    """
    Compute negative activation probability ap^N(u) recursively (Algorithm 1).
    """
    # Sort nodes by distance descending (leaves of MIIA first)
    sorted_nodes = sorted(list(MIIA_nodes), key=lambda x: MIIA_dists[x], reverse=True)
    
    ap = {}
    
    for v in sorted_nodes:
        if v in S_N:
            ap[v] = 1.0
        else:
            # Neighbors w such that w -> v in MIIA.
            # In MIIA(u), paths go w -> v -> ... -> u.
            # We want w such that w is in MIIA and edge w->v exists.
            
            product = 1.0
            neighbors_in_miia = []
            
            # Incoming edges to v in graph G
            for w in G.predecessors(v):
                if w in MIIA_nodes:
                    w_u_dist = MIIA_dists[w]
                    v_u_dist = MIIA_dists[v]
                    edge_weight_log = -math.log(G[w][v]['weight'])
                    
                    if abs(w_u_dist - (v_u_dist + edge_weight_log)) < 1e-9:
                         neighbors_in_miia.append(w)
            
            if not neighbors_in_miia:
                ap[v] = 0.0
            else:
                for w in neighbors_in_miia:
                    val = ap.get(w, 0.0)
                    product *= (1 - val * G[w][v]['weight'])
                ap[v] = 1.0 - product
    
    return ap, sorted_nodes

def get_hops_in_miia(u, MIIA_nodes, MIIA_dists, G):
    """
    Compute hop distances from all nodes v in MIIA to u (d_N(v, u)).
    """
    hops = {u: 0}
    queue = [u]
    visited = {u}
    
    while queue:
        curr = queue.pop(0)
        current_hops = hops[curr]
        
        # Traverse "up" the MIIA towards sources (v -> curr)
        # We look for predecessors v such that v -> curr is valid MIIA edge
        for v in G.predecessors(curr):
            if v in MIIA_nodes and v not in visited:
                v_dist = MIIA_dists[v]
                curr_dist = MIIA_dists[curr]
                weight_log = -math.log(G[v][curr]['weight'])
                
                if abs(v_dist - (curr_dist + weight_log)) < 1e-9:
                    hops[v] = current_hops + 1
                    visited.add(v)
                    queue.append(v)
                    
    return hops

def compute_dec_inf(u, S_N, MIIA_nodes, MIIA_dists, d_c, G, hops_to_u):
    d_c_u = d_c.get(u, float('inf'))
    
    # Calculate d_N_max: the maximum hop distance from any active negative seed to u.
    # A negative seed is active only if it reaches u strictly faster than any positive seed (hops < d_c[u]).
    valid_hops = [hops_to_u[s] for s in S_N if s in MIIA_nodes and hops_to_u[s] < d_c_u]
    if not valid_hops:
        return 0.0
    
    d_N_max = max(valid_hops)
    
    sorted_nodes = sorted(list(MIIA_nodes), key=lambda x: MIIA_dists[x], reverse=True)
    
    ap = {}
    for v in sorted_nodes:
        if v in S_N:
            if hops_to_u.get(v, float('inf')) <= d_N_max:
                ap[v] = 1.0
            else:
                ap[v] = 0.0
        else:
            product = 1.0
            has_neighbors = False
            for w in G.predecessors(v):
                if w in MIIA_nodes:
                    w_dist = MIIA_dists[w]
                    v_dist = MIIA_dists[v]
                    w_weight = -math.log(G[w][v]['weight'])
                    if abs(w_dist - (v_dist + w_weight)) < 1e-9:
                         has_neighbors = True
                         val = ap.get(w, 0.0)
                         product *= (1 - val * G[w][v]['weight'])
            
            if not has_neighbors:
                ap[v] = 0.0
            else:
                ap[v] = 1.0 - product
                
    return ap.get(u, 0.0)


def cmia_h(G, S_N, k, theta=1/320):
    # 1. Initialization
    S_P = []
    nodes = list(G.nodes())
    d_c = {v: float('inf') for v in nodes} 
    DecInf = {v: 0.0 for v in nodes}
    
    NegS = set()
    for u in S_N:
        mia_nodes, _ = get_mia_structure(G, u, theta, is_in=False) # MIOA
        NegS.update(mia_nodes)
    
    NegS = NegS - set(S_N)
    
    MIIA_cache = {} 
    
    for u in NegS:
        nodes_in, dists = get_mia_structure(G, u, theta, is_in=True) # MIIA
        hops = get_hops_in_miia(u, nodes_in, dists, G)
        MIIA_cache[u] = (nodes_in, dists, hops)
        
        ap_old = compute_dec_inf(u, S_N, nodes_in, dists, d_c, G, hops)
        
        valid_hops = [hops[s] for s in S_N if s in nodes_in]
        if not valid_hops:
            continue
        d_N_max_curr = max(valid_hops)
        
        for v in nodes_in:
            h_v_u = hops.get(v, float('inf'))
            if h_v_u <= d_N_max_curr:
                d_c_temp = {u: h_v_u} 
                ap_new = compute_dec_inf(u, S_N, nodes_in, dists, d_c_temp, G, hops)
                gain = ap_old - ap_new
                DecInf[v] += gain

    # Greedy Loop
    for i in range(k):
        candidates = [n for n in nodes if n not in S_P and n not in S_N]
        if not candidates: break
        
        u_best = max(candidates, key=lambda x: DecInf.get(x, 0))
        S_P.append(u_best)
        
        # Update d_c globally
        bfs_q = [(u_best, 0)]
        dists_from_new_seed = {u_best: 0}
        
        while bfs_q:
            curr, d = bfs_q.pop(0)
            if d > 10: pass
            
            # Traverse OUTGOING edges
            for nbr in G.neighbors(curr):
                if nbr not in dists_from_new_seed:
                    dists_from_new_seed[nbr] = d + 1
                    bfs_q.append((nbr, d + 1))
        
        affected_targets = []
        for t in NegS:
            new_dist = dists_from_new_seed.get(t, float('inf'))
            if new_dist < d_c[t]:
                affected_targets.append(t)
        
        for t in affected_targets:
            old_dc = d_c[t]
            nodes_in, dists, hops = MIIA_cache[t]
            
            ap_old_state = compute_dec_inf(t, S_N, nodes_in, dists, {t: old_dc}, G, hops)
            
            valid_hops_old = [hops[s] for s in S_N if s in nodes_in and hops[s] < old_dc]
            if valid_hops_old:
                d_N_max_old = max(valid_hops_old)
                for v in nodes_in:
                    if hops.get(v, float('inf')) <= d_N_max_old:
                        ap_with_v = compute_dec_inf(t, S_N, nodes_in, dists, {t: hops[v]}, G, hops)
                        gain_old = ap_old_state - ap_with_v
                        DecInf[v] -= gain_old

            d_c[t] = dists_from_new_seed[t]
            
            new_dc = d_c[t]
            ap_new_state = compute_dec_inf(t, S_N, nodes_in, dists, {t: new_dc}, G, hops)
            
            valid_hops_new = [hops[s] for s in S_N if s in nodes_in and hops[s] < new_dc]
            if valid_hops_new:
                d_N_max_new = max(valid_hops_new)
                for v in nodes_in:
                    if hops.get(v, float('inf')) <= d_N_max_new:
                        ap_with_v = compute_dec_inf(t, S_N, nodes_in, dists, {t: hops[v]}, G, hops)
                        gain_new = ap_new_state - ap_with_v
                        DecInf[v] += gain_new

    return S_P


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
        # Use DiGraph for directed graphs
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
        avg_neg_nodes_MCICM = []

        for k in k_values:
            repeats = args.repeats
            current_k_mcicm = []

            for r in range(repeats):
                print(f"\nCMIA-H: {file_name}, k={k}, run={r+1}/{repeats}")
                
                start_time = time.time()
                bestS = cmia_h(G, SN, k)
                end_time = time.time()
                print(f"Time taken: {end_time - start_time:.2f}s")

                # Evaluate MCICM
                print(f"Running Monte Carlo Evaluation (MCICM)...")
                res_mcicm = monte_carlo_evaluation(G, bestS, SN, model='MCICM', runs=args.mc_runs)
                print(f"Average Negatively Activated Nodes (MCICM): {res_mcicm}")
                current_k_mcicm.append(res_mcicm)
            
            avg_neg_nodes_MCICM.append(sum(current_k_mcicm) / len(current_k_mcicm))

        # Plot MCICM
        try:
            output_fig_dir_mcicm = f"../../results/MCICM/CMIA-H/repeats{args.repeats}_runs{args.mc_runs}"
            if not os.path.exists(output_fig_dir_mcicm):
                os.makedirs(output_fig_dir_mcicm)
            
            plt.figure(figsize=(6, 6))
            plt.plot(k_values, avg_neg_nodes_MCICM, marker='o', linestyle='--', label=file_name, color='skyblue')
            for x, y in zip(k_values, avg_neg_nodes_MCICM):
                plt.text(x, y, f'{y:.0f}', ha='center', va='bottom')
            plt.title(f'MCICM CMIA-H {file_name}')
            plt.xlabel('k')
            plt.ylabel('Negatively Activated Nodes')
            plt.xticks(k_values)
            plt.tight_layout()
            plt.savefig(os.path.join(output_fig_dir_mcicm, f'MCICM_{file_name}.png'))
            plt.close()
            print(f"Saved plot to {os.path.join(output_fig_dir_mcicm, f'MCICM_{file_name}.png')}")
        except Exception as e:
            print(f"Error plotting MCICM: {e}")

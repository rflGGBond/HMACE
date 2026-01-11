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
        log_name = f"log_{current_date}_cmia-h_undirected.txt"
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
    Construct MIIA (is_in=True) or MIOA (is_in=False) for node u.
    Returns: 
        nodes_in_mia: Set of nodes in the structure.
        dist: Dictionary of -log(prob) distances.
        predecessors: Dict mapping v -> list of predecessors in MIA.
    """
    # Dijkstra
    dist = {u: 0}
    pq = [(0, u)]
    nodes_in_mia = {u}
    predecessors = {u: []} # For MIIA, predecessors of v are nodes w such that w -> v is in MIIA
                           # For undirected graph, w -> v means edge {w, v}
    
    # -log(theta) is the max distance
    max_dist = -math.log(theta) if theta > 0 else float('inf')

    while pq:
        d, curr = heapq.heappop(pq)
        
        if d > dist.get(curr, float('inf')):
            continue
        
        # For undirected graph, neighbors are both in and out
        neighbors = G.neighbors(curr)
        
        for neighbor in neighbors:
            weight = G[curr][neighbor]['weight']
            if weight <= 0: continue
            
            new_dist = d - math.log(weight)
            
            if new_dist <= max_dist:
                if new_dist < dist.get(neighbor, float('inf')):
                    dist[neighbor] = new_dist
                    heapq.heappush(pq, (new_dist, neighbor))
                    nodes_in_mia.add(neighbor)
                    predecessors[neighbor] = [curr]
                elif new_dist == dist.get(neighbor, float('inf')):
                    if neighbor in predecessors:
                        predecessors[neighbor].append(curr)
                    else:
                        predecessors[neighbor] = [curr]
    
    return nodes_in_mia, dist

def compute_ap(u, S_N, MIIA_nodes, MIIA_dists, d_c, G):
    """
    Compute negative activation probability ap^N(u) recursively (Algorithm 1).
    d_c: dict of shortest path distances from Sp to nodes.
    MIIA_nodes: set of nodes in MIIA(u).
    MIIA_dists: dict of distances from u in MIIA (representing -log prob).
    """
    # This needs to be done in topological order (or distance order)
    # In MIIA(u), influence flows towards u. Nodes further from u (larger MIIA_dist) are processed first.
    
    # Sort nodes by distance descending (leaves of MIIA first)
    sorted_nodes = sorted(list(MIIA_nodes), key=lambda x: MIIA_dists[x], reverse=True)
    
    ap = {}
    
    for v in sorted_nodes:
        # Check condition: if any path length in D is smaller than d_c(u, Sp)
        # Here we simplify: if v is influenced by SN, calculate its ap.
        # But wait, the algorithm says ap^N(u, ...) depends on d_N_max.
        # Let's follow Algo 1 logic adapted for general recursive calc.
        
        if v in S_N:
            ap[v] = 1.0
        else:
            # Nin(v) in MIIA(u) are neighbors w such that w -> v is an edge in MIIA.
            # In our Dijkstra from u, w would be "closer" to u? No, w -> v -> ... -> u.
            # So w is further from u than v.
            # Let's find neighbors w of v such that w is in MIIA_nodes and dist[w] > dist[v] 
            # AND edge {w, v} is valid part of shortest path?
            # Actually, standard MIA defines Nin(v) as incoming neighbors in the arborescence.
            # The arborescence is union of shortest paths.
            # So w -> v exists if dist[w] == dist[v] + weight_distance(w, v).
            
            product = 1.0
            neighbors_in_miia = []
            for w in G.neighbors(v):
                if w in MIIA_nodes:
                    # Check if w -> v is valid flow towards u
                    # Distance from u to w should be larger than u to v?
                    # Yes, influence flows w -> v -> u.
                    # Dijkstra dist(u, w) = dist(u, v) + weight(v, w).
                    # So dist(u, w) > dist(u, v).
                    w_u_dist = MIIA_dists[w]
                    v_u_dist = MIIA_dists[v]
                    edge_weight_log = -math.log(G[w][v]['weight'])
                    
                    # Allow for float tolerance
                    if abs(w_u_dist - (v_u_dist + edge_weight_log)) < 1e-9:
                         neighbors_in_miia.append(w)
            
            if not neighbors_in_miia:
                ap[v] = 0.0
            else:
                for w in neighbors_in_miia:
                    # We need ap[w] here. Since we sort descending, w (further) should be done.
                    val = ap.get(w, 0.0)
                    product *= (1 - val * G[w][v]['weight'])
                ap[v] = 1.0 - product
    
    return ap, sorted_nodes

def get_hops_in_miia(u, MIIA_nodes, MIIA_dists, G):
    """
    Compute hop distances from all nodes v in MIIA to u (d_N(v, u)).
    """
    # Since MIIA is a DAG (or union of paths) flowing to u.
    # We can do BFS backwards from u on the MIIA edges.
    
    hops = {u: 0}
    queue = [u]
    
    visited = {u}
    
    while queue:
        curr = queue.pop(0)
        current_hops = hops[curr]
        
        for v in G.neighbors(curr):
            if v in MIIA_nodes and v not in visited:
                # Check if v -> curr is valid in MIIA
                # dist[v] > dist[curr]
                v_dist = MIIA_dists[v]
                curr_dist = MIIA_dists[curr]
                weight_log = -math.log(G[v][curr]['weight'])
                
                if abs(v_dist - (curr_dist + weight_log)) < 1e-9:
                    hops[v] = current_hops + 1
                    visited.add(v)
                    queue.append(v)
                    
    return hops

def compute_dec_inf(u, S_N, MIIA_nodes, MIIA_dists, d_c, G, hops_to_u):
    """
    Compute DecInf(w, u, Sp) logic?
    Actually we need ap^N(u) considering blocking.
    ap^N(u) = 1 - Product(1 - ap(w)*p_wu)
    Blocking: if d_c[v] <= hops(v, u), v cannot propagate negative influence to u?
    Actually, "only negative seeds in MIIA(u) with d_N(seed, u) < d_c(u, Sp) have negative influence".
    Wait, Eq (1) says:
    DecInf = ap^N(..., d_N_max) - ap^N(..., d_N_max_new)
    
    So we need a function that computes ap^N(u) given a max_hop_limit.
    Only consider paths/nodes where hops_to_u <= max_hop_limit.
    """
    
    # Filter MIIA nodes by hop limit?
    # No, the algo says "ap^N(..., d_N_max(u, Sp))".
    # d_N_max(u, Sp) is largest length in D (path lengths from neg seeds to u) smaller than d_c(u, Sp).
    # If d_c(u, Sp) is infinity, d_N_max is max possible.
    
    d_c_u = d_c.get(u, float('inf'))
    
    # Calculate d_N_max: the maximum hop distance from any active negative seed to u.
    # A negative seed is active only if it reaches u strictly faster than any positive seed (hops < d_c[u]).
    valid_hops = [hops_to_u[s] for s in S_N if s in MIIA_nodes and hops_to_u[s] < d_c_u]
    if not valid_hops:
        return 0.0
    
    d_N_max = max(valid_hops)
    
    # 2. Compute ap(u) restricting to seeds with hops <= d_N_max.
    # Sort nodes by MIIA dist descending
    sorted_nodes = sorted(list(MIIA_nodes), key=lambda x: MIIA_dists[x], reverse=True)
    
    ap = {}
    for v in sorted_nodes:
        if v in S_N:
            # Check restriction
            if hops_to_u.get(v, float('inf')) <= d_N_max:
                ap[v] = 1.0
            else:
                ap[v] = 0.0 # Effectively not a seed for this calculation
                # But it might still transmit? 
                # "negative influence ... is assumed to only propagate in MIOA"
                # If it's not a seed, treat as normal node.
                # But normal nodes start at 0.
                
                # Wait, if v is a negative seed but blocked (hops > d_N_max), 
                # it acts like a non-seed. Can it transmit?
                # "negative influence of any negative seed ... is blocked".
                # So it contributes 0.
                
                # Check if v receives from others?
                # Seeds usually don't receive? 
                # Assuming independent cascade, seeds are always active (prob 1).
                # If blocked, it's 0.
                pass 
                # If v is S_N but blocked, we calculate its ap based on incoming neighbors?
                # Usually seeds are exogenous. ap[v] = 1.
                # If blocked, ap[v] = 0.
        else:
            # Normal calculation
            product = 1.0
            has_neighbors = False
            for w in G.neighbors(v):
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
    d_c = {v: float('inf') for v in nodes} # Shortest path from Sp (hops)
    DecInf = {v: 0.0 for v in nodes}
    
    # Construct MIOA for each u in S_N to find NegS (range of negative influence)
    NegS = set()
    for u in S_N:
        # MIOA(u) is MIIA(u) on reversed edges? 
        # For undirected, same structure.
        # But MIOA means influence FROM u.
        mia_nodes, _ = get_mia_structure(G, u, theta, is_in=False) # Undirected: same
        NegS.update(mia_nodes)
    
    NegS = NegS - set(S_N) # Exclude seeds themselves? Text says "NegS = NegS U (MIOA \ SN)"
    
    # Pre-compute MIIA for all u in NegS
    MIIA_cache = {} # u -> (nodes, dists, hops_to_u)
    
    for u in NegS:
        nodes_in, dists = get_mia_structure(G, u, theta, is_in=True)
        hops = get_hops_in_miia(u, nodes_in, dists, G)
        MIIA_cache[u] = (nodes_in, dists, hops)
        
        # Initial blocked influence (before any positive seeds, d_c = inf)
        # DecInf(v) += ...
        # But wait, initially d_c = inf.
        # ap^N(u, ..., inf) is the base influence.
        # ap^N(u, ..., inf U {v}) is influence if v was seed.
        # Line 12-14: For each node v in PIIS(u, ...):
        # DecInf(v) += ap^N(old) - ap^N(new with v)
        
        # d_N_max(u, Sp=empty) -> max hops of any seed.
        # We need to simulate adding v as seed.
        # This seems expensive to do for all v.
        # Optimization: PIIS(u, ...) restricts v to relevant nodes.
        
        # Initial d_c is inf.
        # d_N_max(u, empty) is the max hop distance of any negative seed reaching u.
        
        ap_old = compute_dec_inf(u, S_N, nodes_in, dists, d_c, G, hops)
        
        # We need to identify v such that adding v reduces this ap.
        # PIIS(u, d_N_max) = {v | hop(v, u) <= d_N_max} ?
        # Yes, line 11: construct PIIS.
        # PIIS are nodes that can reach u within the "cutoff" distance.
        # For these v, if we pick v, d_c(u) becomes hop(v, u).
        # New ap will be based on d_N_max_new which is < hop(v, u).
        
        # Determine d_N_max for current u
        valid_hops = [hops[s] for s in S_N if s in nodes_in]
        if not valid_hops:
            continue
        d_N_max_curr = max(valid_hops)
        
        # PIIS: nodes v where hops(v, u) <= d_N_max_curr
        # We can find these by reversing the hops flow?
        # hops map is v -> u distance.
        # So we iterate v in MIIA.
        
        for v in nodes_in:
            h_v_u = hops.get(v, float('inf'))
            if h_v_u <= d_N_max_curr:
                # Calculate marginal gain if v is added
                # New d_c(u) would be h_v_u (assuming v is closest)
                # We need ap^N with d_c(u) = h_v_u
                
                # Temp d_c for u
                # We pass a d_c dict where d_c[u] = h_v_u
                # But compute_dec_inf takes full d_c.
                # We can mock it.
                d_c_temp = {u: h_v_u} 
                
                ap_new = compute_dec_inf(u, S_N, nodes_in, dists, d_c_temp, G, hops)
                
                gain = ap_old - ap_new
                DecInf[v] += gain

    # Greedy Loop
    for i in range(k):
        # Pick best u (not in S_N and not in S_P)
        candidates = [n for n in nodes if n not in S_P and n not in S_N]
        if not candidates: break
        
        u_best = max(candidates, key=lambda x: DecInf.get(x, 0))
        S_P.append(u_best)
        
        # Update d_c globally
        # BFS from u_best to update d_c for all nodes
        # Since unweighted hops (MCICM positive prop = 1)
        queue = [(u_best, 0)]
        visited_bfs = set()
        
        # We need to update d_c.
        # Standard BFS update.
        # But only need to update if new path is shorter.
        
        # Full BFS update might be slow. PIOS optimization?
        # Line 18: construct PIOS(u).
        # PIOS(u) = nodes v such that d_P(u, v) <= d_N_max(v, Sp).
        # This implies we only explore nodes where we improve blocking.
        
        # Let's do a bounded BFS from u_best.
        
        # We need to identify nodes v whose d_c(v) improves.
        # And for those v, if v is in some NegS's MIIA, we update DecInf of potential seeds.
        
        # This is complex to implement efficiently from scratch perfectly matching the paper.
        # I will implement a simpler update:
        # 1. Update d_c for all reachable nodes (BFS).
        # 2. For any node v whose d_c changed, check if it affects any u in NegS.
        #    If v is in MIIA(target), we need to update DecInf for nodes in PIIS(target).
        
        # Efficient approach using MIIA_cache:
        # Iterate all targets t in NegS.
        # Update d_c(t) based on new seed u_best.
        # If d_c(t) reduced, we update DecInf contributions from t.
        
        # 1. Calculate new distance to all relevant targets.
        # BFS from u_best.
        
        bfs_q = [(u_best, 0)]
        dists_from_new_seed = {u_best: 0}
        
        while bfs_q:
            curr, d = bfs_q.pop(0)
            
            # Optimization: Stop if d is too large?
            # Theta related? No, MCICM is 1.0 prob.
            if d > 10: # Safety break? Or rely on graph size
                pass
            
            for nbr in G.neighbors(curr):
                if nbr not in dists_from_new_seed:
                    dists_from_new_seed[nbr] = d + 1
                    bfs_q.append((nbr, d + 1))
        
        # 2. Identify affected targets
        affected_targets = []
        for t in NegS:
            new_dist = dists_from_new_seed.get(t, float('inf'))
            if new_dist < d_c[t]:
                affected_targets.append(t)
        
        # 3. For each affected target, update DecInf
        for t in affected_targets:
            # Revert old contributions
            # We need to know what d_c[t] WAS.
            old_dc = d_c[t]
            nodes_in, dists, hops = MIIA_cache[t]
            
            ap_old_state = compute_dec_inf(t, S_N, nodes_in, dists, {t: old_dc}, G, hops)
            
            # Subtract old gains for v in PIIS
            # Identify v in PIIS(t) based on old_dc
            # d_N_max_old logic
            valid_hops_old = [hops[s] for s in S_N if s in nodes_in and hops[s] < old_dc]
            if valid_hops_old:
                d_N_max_old = max(valid_hops_old)
                for v in nodes_in:
                    if hops.get(v, float('inf')) <= d_N_max_old:
                        # Calc gain
                        ap_with_v = compute_dec_inf(t, S_N, nodes_in, dists, {t: hops[v]}, G, hops)
                        gain_old = ap_old_state - ap_with_v
                        DecInf[v] -= gain_old

            # Update d_c
            d_c[t] = dists_from_new_seed[t]
            
            # Add new contributions
            new_dc = d_c[t]
            ap_new_state = compute_dec_inf(t, S_N, nodes_in, dists, {t: new_dc}, G, hops)
            
            valid_hops_new = [hops[s] for s in S_N if s in nodes_in and hops[s] < new_dc]
            if valid_hops_new:
                d_N_max_new = max(valid_hops_new)
                for v in nodes_in:
                    if hops.get(v, float('inf')) <= d_N_max_new:
                        # Calc gain
                        ap_with_v = compute_dec_inf(t, S_N, nodes_in, dists, {t: hops[v]}, G, hops)
                        gain_new = ap_new_state - ap_with_v
                        DecInf[v] += gain_new

    return S_P


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, nargs="+", default=[20, 110, 200])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--graphs", type=str, nargs="+", default=["facebook"])
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
        G = nx.Graph()
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
                current_seed = 42 + k + r * 1000
                random.seed(current_seed)
                try:
                    import numpy as np
                    np.random.seed(current_seed)
                except ImportError:
                    pass
                print(f"\nCMIA-H: {file_name}, k={k}, run={r+1}/{repeats}")
                
                start_time = time.time()
                bestS = cmia_h(G, SN, k)
                end_time = time.time()
                print(f"Time taken: {end_time - start_time:.2f}s")
                # print("Selected Seeds:", bestS)

                # Evaluate MCICM
                print(f"Running Monte Carlo Evaluation (MCICM)...")
                res_mcicm = monte_carlo_evaluation(G, bestS, SN, model='MCICM', runs=args.mc_runs)
                print(f"Negatively Activated Nodes (MCICM): {res_mcicm:.2f}")
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
                plt.text(x, y, f'{y:.2f}', ha='center', va='bottom')
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

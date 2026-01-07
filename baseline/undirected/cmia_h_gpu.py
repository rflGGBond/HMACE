import os, sys
import time
import copy
import random
import heapq
import math
import torch
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime
import argparse

# Add path to import select_SN
sys.path.append('../')
from select_SN import select_SN

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

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

def monte_carlo_sparse_batch(num_nodes, indices, values, S_P, S_N, runs=100):
    """
    Monte Carlo Simulation using Sparse Matrix Batching (Block Diagonal).
    Memory Efficient: ~O(runs * E) instead of O(runs * N^2).
    """
    E = values.shape[0]
    
    # 1. Generate randoms and live mask for all runs
    rand = torch.rand((runs, E), device=device)
    live = rand < values.unsqueeze(0) # (runs, E)
    
    # 2. Construct Block Diagonal Indices and Values
    # We want a giant matrix of size (runs*N, runs*N)
    # The adjacency for run r is shifted by r*num_nodes
    
    # Shift amount for each run: [0, N, 2N, ...]
    shifts = torch.arange(runs, device=device) * num_nodes
    # Repeat for each edge: (runs, E)
    shifts = shifts.unsqueeze(1).expand(runs, E)
    
    # Filter only live edges
    # Indices: (2, E)
    row_indices = indices[0].unsqueeze(0).expand(runs, E) # (runs, E)
    col_indices = indices[1].unsqueeze(0).expand(runs, E) # (runs, E)
    
    # Apply shift
    row_indices = row_indices + shifts
    col_indices = col_indices + shifts
    
    # Flatten and mask
    mask = live.flatten()
    final_rows = row_indices.flatten()[mask]
    final_cols = col_indices.flatten()[mask]
    
    # Create sparse tensor
    # Size: (runs*N, runs*N)
    # Values are all 1.0 (unweighted connectivity in live graph)
    final_indices = torch.stack([final_rows, final_cols])
    final_values = torch.ones(final_rows.shape[0], device=device, dtype=torch.float32)
    
    adj_neg_sparse = torch.sparse_coo_tensor(
        final_indices, 
        final_values, 
        (runs * num_nodes, runs * num_nodes)
    )
    
    # 3. Construct Positive Adjacency (Deterministic)
    # Positive edges are static. We can repeat them or use broadcasting?
    # Sparse MM doesn't support broadcasting easily.
    # We construct the same block diagonal structure but with ALL edges live.
    
    # Reuse shifts but without live mask
    row_indices_pos = (indices[0].unsqueeze(0).expand(runs, E) + shifts).flatten()
    col_indices_pos = (indices[1].unsqueeze(0).expand(runs, E) + shifts).flatten()
    
    final_indices_pos = torch.stack([row_indices_pos, col_indices_pos])
    final_values_pos = torch.ones(row_indices_pos.shape[0], device=device, dtype=torch.float32)
    
    adj_pos_sparse = torch.sparse_coo_tensor(
        final_indices_pos,
        final_values_pos,
        (runs * num_nodes, runs * num_nodes)
    )
    
    # 4. Initialize State Vectors
    # Size: (runs * num_nodes, 1)
    active_pos = torch.zeros((runs * num_nodes, 1), device=device)
    active_neg = torch.zeros((runs * num_nodes, 1), device=device)
    
    # Set seeds
    if S_P:
        # S_P is list of nodes. We need to set S_P + r*N for all r.
        sp_tensor = torch.tensor(S_P, device=device)
        # (runs, |S_P|)
        sp_indices = sp_tensor.unsqueeze(0).expand(runs, len(S_P)) + shifts[:, :len(S_P)]
        active_pos[sp_indices.flatten(), 0] = 1.0
        
    if S_N:
        sn_tensor = torch.tensor(S_N, device=device)
        sn_indices = sn_tensor.unsqueeze(0).expand(runs, len(S_N)) + shifts[:, :len(S_N)]
        active_neg[sn_indices.flatten(), 0] = 1.0
        
    frontier_pos = active_pos.clone()
    frontier_neg = active_neg.clone()
    
    # 5. Propagation Loop
    for _ in range(num_nodes):
        if not frontier_pos.any() and not frontier_neg.any():
             break
        
        # Positive Step
        # vector (R*N, 1). Matrix (R*N, R*N).
        # We need Transpose multiply: A.T @ x
        # Our indices are [u, v] meaning u->v. 
        # To propagate u->v, we need sum over u of A[u,v]*x[u].
        # This corresponds to A.T @ x.
        # torch.sparse.mm(mat, vec) does mat @ vec.
        # So we need Transpose of Adjacency.
        
        # Transpose Sparse Matrix efficiently?
        # Just swap row/col indices during creation? 
        # Yes. Let's create Transposed Adjacency for propagation.
        
        # NOTE: adj_neg_sparse created above used [rows, cols]. 
        # If rows=u, cols=v (u->v).
        # We want y[v] = sum(x[u]).
        # This is y = A.T @ x.
        # Or create A_transposed where indices are [cols, rows].
        
        # Let's recreate the sparse matrices as Transposed (v, u) for efficient matvec
        # Optimization: We can just swap indices in the constructor calls above?
        # Yes. But let's assume we pass the transposed version to mm.
        # Sparse .t() is supported.
        
        next_pos = torch.sparse.mm(adj_pos_sparse.t(), frontier_pos)
        next_pos = (next_pos > 0).float()
        
        # Negative Step
        next_neg = torch.sparse.mm(adj_neg_sparse.t(), frontier_neg)
        next_neg = (next_neg > 0).float()
        
        # Conflict Resolution
        already_active = (active_pos + active_neg).clamp(0, 1)
        
        new_pos = next_pos * (1 - already_active)
        new_neg = next_neg * (1 - already_active)
        
        final_pos = new_pos
        final_neg = new_neg * (1 - new_pos)
        
        active_pos += final_pos
        active_neg += final_neg
        
        frontier_pos = final_pos
        frontier_neg = final_neg
        
    # 6. Count Results
    # active_neg is (runs*N, 1)
    # Reshape to (runs, N)
    active_neg_reshaped = active_neg.view(runs, num_nodes)
    neg_counts = active_neg_reshaped.sum(dim=1)
    return neg_counts.mean().item()

def monte_carlo_gpu(num_nodes, indices, values, S_P, S_N, runs=100):
    return monte_carlo_sparse_batch(num_nodes, indices, values, S_P, S_N, runs)

# --- CMIA-H Helpers (Optimized) ---

def get_mia_structure_fast(G_succ, G_pred, u, theta, is_in=True):
    dist = {u: 0}
    pq = [(0, u)]
    nodes_in_mia = {u}
    max_dist = -math.log(theta) if theta > 0 else float('inf')

    while pq:
        d, curr = heapq.heappop(pq)
        if d > dist.get(curr, float('inf')): continue
        
        # Directed: 
        # is_in=True (MIIA): Paths TO u. Traverse INCOMING (G_pred).
        # is_in=False (MIOA): Paths FROM u. Traverse OUTGOING (G_succ).
        neighbors = G_pred[curr] if is_in else G_succ[curr]
        
        for neighbor, weight in neighbors.items():
            if weight <= 0: continue
            new_dist = d - math.log(weight)
            if new_dist <= max_dist:
                if new_dist < dist.get(neighbor, float('inf')):
                    dist[neighbor] = new_dist
                    heapq.heappush(pq, (new_dist, neighbor))
                    nodes_in_mia.add(neighbor)
    
    return nodes_in_mia, dist

def get_hops_fast(u, MIIA_nodes, MIIA_dists, G_pred):
    hops = {u: 0}
    queue = [u]
    visited = {u}
    while queue:
        curr = queue.pop(0)
        current_hops = hops[curr]
        
        # Traverse "up" (v -> curr)
        for v, weight in G_pred[curr].items():
            if v in MIIA_nodes and v not in visited:
                v_dist = MIIA_dists[v]
                curr_dist = MIIA_dists[curr]
                weight_log = -math.log(weight)
                if abs(v_dist - (curr_dist + weight_log)) < 1e-9:
                    hops[v] = current_hops + 1
                    visited.add(v)
                    queue.append(v)
    return hops

def compute_dec_inf_fast(u, S_N, MIIA_nodes, MIIA_dists, d_c, G_pred, hops_to_u):
    d_c_u = d_c.get(u, float('inf'))
    valid_hops = [hops_to_u[s] for s in S_N if s in MIIA_nodes and hops_to_u[s] < d_c_u]
    if not valid_hops: return 0.0
    d_N_max = max(valid_hops)
    
    sorted_nodes = sorted(list(MIIA_nodes), key=lambda x: MIIA_dists[x], reverse=True)
    ap = {}
    for v in sorted_nodes:
        if v in S_N:
            ap[v] = 1.0 if hops_to_u.get(v, float('inf')) <= d_N_max else 0.0
        else:
            product = 1.0
            has_neighbors = False
            for w, weight in G_pred[v].items():
                if w in MIIA_nodes:
                    w_dist = MIIA_dists[w]
                    v_dist = MIIA_dists[v]
                    w_weight = -math.log(weight)
                    if abs(w_dist - (v_dist + w_weight)) < 1e-9:
                         has_neighbors = True
                         val = ap.get(w, 0.0)
                         product *= (1 - val * weight)
            if not has_neighbors: ap[v] = 0.0
            else: ap[v] = 1.0 - product
    return ap.get(u, 0.0)

def cmia_h_gpu(G, S_N, k, theta=1/320):
    nodes = list(G.nodes())
    # G_succ: u -> {v: w}
    G_succ = {u: {v: G[u][v]['weight'] for v in G[u]} for u in nodes}
    # For undirected, G_pred == G_succ
    G_pred = G_succ
    
    S_P = []
    d_c = {v: float('inf') for v in nodes} 
    DecInf = {v: 0.0 for v in nodes}
    
    # Initialization
    NegS = set()
    for u in S_N:
        mia_nodes, _ = get_mia_structure_fast(G_succ, G_pred, u, theta, is_in=False)
        NegS.update(mia_nodes)
    NegS = NegS - set(S_N)
    
    MIIA_cache = {}
    for u in NegS:
        nodes_in, dists = get_mia_structure_fast(G_succ, G_pred, u, theta, is_in=True)
        hops = get_hops_fast(u, nodes_in, dists, G_pred)
        MIIA_cache[u] = (nodes_in, dists, hops)
        
        ap_old = compute_dec_inf_fast(u, S_N, nodes_in, dists, d_c, G_pred, hops)
        
        valid_hops = [hops[s] for s in S_N if s in nodes_in]
        if not valid_hops: continue
        d_N_max_curr = max(valid_hops)
        
        for v in nodes_in:
            if hops.get(v, float('inf')) <= d_N_max_curr:
                d_c_temp = {u: hops[v]} 
                ap_new = compute_dec_inf_fast(u, S_N, nodes_in, dists, d_c_temp, G_pred, hops)
                DecInf[v] += (ap_old - ap_new)

    # Greedy Loop
    for i in range(k):
        candidates = [n for n in nodes if n not in S_P and n not in S_N]
        if not candidates: break
        
        u_best = max(candidates, key=lambda x: DecInf.get(x, 0))
        S_P.append(u_best)
        
        # Update d_c using BFS (Outgoing edges)
        dists_from_new_seed = {u_best: 0}
        bfs_q = [(u_best, 0)]
        while bfs_q:
            curr, d = bfs_q.pop(0)
            if d > 10: continue
            for nbr in G_succ[curr]:
                if nbr not in dists_from_new_seed:
                    dists_from_new_seed[nbr] = d + 1
                    bfs_q.append((nbr, d + 1))
        
        affected_targets = [t for t in NegS if dists_from_new_seed.get(t, float('inf')) < d_c[t]]
        
        for t in affected_targets:
            old_dc = d_c[t]
            nodes_in, dists, hops = MIIA_cache[t]
            
            ap_old_state = compute_dec_inf_fast(t, S_N, nodes_in, dists, {t: old_dc}, G_pred, hops)
            
            valid_hops_old = [hops[s] for s in S_N if s in nodes_in and hops[s] < old_dc]
            if valid_hops_old:
                d_N_max_old = max(valid_hops_old)
                for v in nodes_in:
                    if hops.get(v, float('inf')) <= d_N_max_old:
                        ap_with_v = compute_dec_inf_fast(t, S_N, nodes_in, dists, {t: hops[v]}, G_pred, hops)
                        DecInf[v] -= (ap_old_state - ap_with_v)

            d_c[t] = dists_from_new_seed[t]
            
            new_dc = d_c[t]
            ap_new_state = compute_dec_inf_fast(t, S_N, nodes_in, dists, {t: new_dc}, G_pred, hops)
            
            valid_hops_new = [hops[s] for s in S_N if s in nodes_in and hops[s] < new_dc]
            if valid_hops_new:
                d_N_max_new = max(valid_hops_new)
                for v in nodes_in:
                    if hops.get(v, float('inf')) <= d_N_max_new:
                        ap_with_v = compute_dec_inf_fast(t, S_N, nodes_in, dists, {t: hops[v]}, G_pred, hops)
                        DecInf[v] += (ap_new_state - ap_with_v)

    return S_P

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, nargs="+", default=[20, 110, 200])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--graphs", type=str, nargs="+", default=["facebook"])
    parser.add_argument("--mc_runs", type=int, default=100)
    args = parser.parse_args()

    SEED = 42
    random.seed(SEED)
    torch.manual_seed(SEED)
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
        
        # Prepare GPU Data
        # Map nodes to 0..N-1
        node_map = {node: i for i, node in enumerate(nodes)}
        reverse_map = {i: node for node, i in node_map.items()}
        num_nodes = len(nodes)
        
        # Build indices/values for Torch
        # Undirected: add (u,v) and (v,u)
        edges_u = []
        edges_v = []
        weights = []
        for u, v, data in G.edges(data=True):
            idx_u, idx_v = node_map[u], node_map[v]
            w = data['weight']
            edges_u.extend([idx_u, idx_v])
            edges_v.extend([idx_v, idx_u])
            weights.extend([w, w])
            
        indices = torch.tensor([edges_u, edges_v], dtype=torch.long).to(device)
        values = torch.tensor(weights, dtype=torch.float32).to(device)
        
        # Map SN
        SN_mapped = [node_map[u] for u in SN]

        k_values = args.k
        avg_neg_nodes_MCICM = []

        for k in k_values:
            repeats = args.repeats
            current_k_mcicm = []

            for r in range(repeats):
                print(f"\nCMIA-H: {file_name}, k={k}, run={r+1}/{repeats}")
                
                start_time = time.time()
                bestS = cmia_h_gpu(G, SN, k)
                end_time = time.time()
                print(f"Selection Time: {end_time - start_time:.2f}s")
                
                # GPU Evaluation
                print(f"Running Monte Carlo Evaluation...")
                bestS_mapped = [node_map[u] for u in bestS]
                res_mcicm = monte_carlo_gpu(num_nodes, indices, values, bestS_mapped, SN_mapped, runs=args.mc_runs)
                print(f"Average Negatively Activated Nodes: {res_mcicm:.0f}")
                current_k_mcicm.append(res_mcicm.round().astype(int))
            
            avg_neg_nodes_MCICM.append((sum(current_k_mcicm) / len(current_k_mcicm)).round().astype(int))

        # Plot
        try:
            output_fig_dir = f"../../results/MCICM/CMIA-H/repeats{args.repeats}_runs{args.mc_runs}"
            if not os.path.exists(output_fig_dir):
                os.makedirs(output_fig_dir)
            
            plt.figure(figsize=(6, 6))
            plt.plot(k_values, avg_neg_nodes_MCICM, marker='o', linestyle='--', label=file_name, color='skyblue')
            for x, y in zip(k_values, avg_neg_nodes_MCICM):
                plt.text(x, y, f'{y:.0f}', ha='center', va='bottom')
            plt.title(f'MCICM CMIA-H {file_name}')
            plt.xlabel('k')
            plt.ylabel('Negatively Activated Nodes')
            plt.xticks(k_values)
            plt.tight_layout()
            plt.savefig(os.path.join(output_fig_dir, f'MCICM_{file_name}.png'))
            plt.close()
            print(f"Saved plot.")
        except Exception as e:
            print(f"Error plotting: {e}")

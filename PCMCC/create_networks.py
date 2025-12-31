# -*- coding: utf-8 -*-
import networkx as nx
import random
import os
import igraph as ig
import leidenalg
import math

random.seed(42)

save_dir = "./synthetic_graphs_best"
os.makedirs(save_dir, exist_ok=True)

WEIGHT_CHOICES = [0.01, 0.05, 0.2]

# =============================================
# networkx 转 igraph（用于 Leiden）
# =============================================
def nx_to_igraph(G):
    mapping = {node: i for i, node in enumerate(G.nodes())}
    edges = [(mapping[u], mapping[v]) for u, v in G.edges()]
    g = ig.Graph(edges=edges, directed=False)
    return g

# =============================================
# 保存为 txt (论文格式：u v w)
# =============================================
def save_txt(G, path, one_based=False):
    with open(path, "w", encoding="utf-8") as f:
        for u, v in G.edges():
            w = random.choice(WEIGHT_CHOICES)
            if one_based:
                f.write(f"{u+1} {v+1} {w}\n")
            else:
                f.write(f"{u} {v} {w}\n")

# =============================================
# Leiden 模块度
# =============================================
def modularity_leiden(G):
    g = nx_to_igraph(G)
    partition = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition)
    return partition.modularity

# =============================================
# 四种网络的生成定义（严格按论文）
# =============================================
def gen_BA():
    return nx.barabasi_albert_graph(3000, 6)

def gen_ER():
    n = 3000
    p = 0.006
    m = int(p * n * (n - 1) / 2)  # 固定边数版本
    return nx.gnm_random_graph(n, m)

def gen_RG():
    return nx.watts_strogatz_graph(3000, 6, 0.0)

def gen_WS():
    return nx.watts_strogatz_graph(3000, 6, 0.3)

# =============================================
# 搜索最接近目标 Q 的网络
# =============================================
def search_best_graph(name, gen_func, target_Q, trials=30):
    print(f"\n===== 搜索 {name}，目标 Q = {target_Q} =====")

    best_gap = 999
    best_G = None
    best_Q = None
    best_seed = None

    for seed in range(trials):
        random.seed(seed)
        G = gen_func()
        Q = modularity_leiden(G)

        gap = abs(Q - target_Q)
        print(f"[seed={seed}] Q={Q:.4f} gap={gap:.4f}")

        if gap < best_gap:
            best_gap = gap
            best_G = G
            best_Q = Q
            best_seed = seed

    print(f"\n👉 最佳 {name}：seed={best_seed}, Q={best_Q:.4f}, gap={best_gap:.4f}")

    # 保存为 txt
    out_path = os.path.join(save_dir, f"{name}_best.txt")
    save_txt(best_G, out_path)
    print(f"✔ 已保存到：{out_path}")

    return best_G, best_Q


if __name__ == "__main__":
    search_best_graph("BA3000", gen_BA, 0.21, trials=30)
    search_best_graph("ER3000", gen_ER, 0.16, trials=30)
    search_best_graph("RG3000", gen_RG, 0.35, trials=30)
    search_best_graph("WS3000", gen_WS, 0.57, trials=30)

    print("\n===== 全部网络搜索完成 =====")

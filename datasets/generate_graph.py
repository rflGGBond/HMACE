import networkx as nx
import random
import os

random.seed(42)

# 保存数据
def save_graph(G, filename):
    with open(filename, 'w') as f:
        for u, v in G.edges():
            f.write(f'{u} {v}\n')

# 打印图的结构信息
def print_graph_stat(graph_name, G):
    print(f"\n------------------- {graph_name} ------------------------")
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    print(f"Average degree: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")

# 生成图
def generate_graph(graph_name):
    os.makedirs(f"{graph_name}", exist_ok=True)
    n = 3000

    # BA3000
    if graph_name == "BA3000":
        G_BA3000 = nx.barabasi_albert_graph(n, m=6)
        save_graph(G_BA3000, f"BA3000/BA3000.txt")
        print_graph_stat("BA3000", G_BA3000)
    
    # ER3000
    if graph_name == "ER3000":
        G_ER3000 = nx.erdos_renyi_graph(n, p=0.006)
        save_graph(G_ER3000, f"ER3000/ER3000.txt")
        print_graph_stat("ER3000", G_ER3000)

    # RG3000
    if graph_name == "RG3000":
        G_RG3000 = nx.watts_strogatz_graph(n, k=6, p=0)
        save_graph(G_RG3000, f"RG3000/RG3000.txt")
        print_graph_stat("RG3000", G_RG3000)

    # WS3000
    if graph_name == "WS3000":
        G_WS3000 = nx.watts_strogatz_graph(n, k=6, p=0.3)
        save_graph(G_WS3000, f"WS3000/WS3000.txt")
        print_graph_stat("WS3000", G_WS3000)

if __name__ == "__main__":
    graph_names = ["BA3000", "ER3000", "RG3000", "WS3000"]
    for graph_name in graph_names:
        generate_graph(graph_name)


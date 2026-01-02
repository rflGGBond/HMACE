import networkx as nx

def select_SN(G_name, SN_size):
    file_path = f"../../graph/{G_name}.txt"

    G = nx.Graph()

    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            u, v = int(parts[0]), int(parts[1])
            w = float(parts[2])
            G.add_edge(u, v, weight=w)

    # print("图加载完成")
    # print("节点数:", G.number_of_nodes())
    # print("边数:", G.number_of_edges())

    # ===============================
    # 选取度数最高的 50 个节点作为 SN
    # ===============================

    # 按 degree 从大到小排序
    degree_list = sorted(G.degree(), key=lambda x: x[1], reverse=True)

    # 取前 50 个节点
    SN = [node for node, deg in degree_list[:SN_size]]
    
    return SN

# print(select_SN("RG3000", 50))

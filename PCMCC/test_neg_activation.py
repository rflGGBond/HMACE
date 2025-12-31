import networkx as nx
import copy
from collections import defaultdict

# 导入我们修改的fitness_C_7函数
def fitness_C_7(seed_7, G_7, SN_7, comAndFS_7, hop_7):
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
    for v in seed_7:
        pP_fc[v, 0] = 1
        for h in range(hop_7 + 1):
            apP_fc[v, h] = 1
    for v in SN_7:
        pN_fc[v, 0] = 1
        for h in range(hop_7 + 1):
            apN_fc[v, h] = 1
    for h in range(hop_7):
        temppP_fc = defaultdict(lambda: 1)
        temppN_fc = defaultdict(lambda: 1)
        for v in ZP_fc[h]:
            W_fc = list(G_7.neighbors(v))
            ZP_fc[h + 1] += W_fc
            for w in W_fc:
                temppP_fc[w] *= (1 - pP_fc[v, h] * G_7[v][w]['weight'])
        ZP_fc[h + 1] = list(set(ZP_fc[h + 1]))
        for v in ZP_fc[h + 1]:
            pP_fc[v, h + 1] = (1 - temppP_fc[v]) * (1 - apN_fc[v, h]) * (1 - apP_fc[v, h])
            for tau_f in range(h + 1, hop_7 + 1):
                apP_fc[v, tau_f] = apP_fc[v, h] + pP_fc[v, h + 1]
        for v in ZN_fc[h]:
            W_fc = list(G_7.neighbors(v))
            ZN_fc[h + 1] += W_fc
            for w in W_fc:
                temppN_fc[w] *= (1 - pN_fc[v, h] * G_7[v][w]['weight'])
        ZN_fc[h + 1] = list(set(ZN_fc[h + 1]))
        for v in ZN_fc[h + 1]:
            pN_fc[v, h + 1] = temppP_fc[v] * (1 - temppN_fc[v]) * (1 - apN_fc[v, h]) * (1 - apP_fc[v, h])
            for tau_f in range(h + 1, hop_7 + 1):
                apN_fc[v, tau_f] = apN_fc[v, h] + pN_fc[v, h + 1]
    for u in comAndFS_7:
        effect_fc += apN_fc[u, hop_7]
    
    # === 统计最终被负向激活的节点 ===
    neg_activated_nodes = []
    # 只统计fitnessSpace中的节点，并且pN_fc[u, t] > 0表示在t步被负向激活
    for u in comAndFS_7:
        # 检查是否在任何步骤t被负向激活
        for t in range(1, hop_7 + 1):
            if pN_fc[u, t] > 0:
                neg_activated_nodes.append(u)
                break
    
    num_neg_activated = len(neg_activated_nodes)
    
    # 打印详细信息以便调试
    print(f"负激活节点数量: {num_neg_activated}")
    print(f"负激活节点列表: {neg_activated_nodes}")
    
    # 打印一些pN_fc的值来验证
    print("\n部分节点的pN_fc值:")
    for u in comAndFS_7[:5]:  # 只打印前5个节点
        for t in range(1, hop_7 + 1):
            print(f"pN_fc[{u}, {t}] = {pN_fc[u, t]}")
    
    return effect_fc, num_neg_activated

# 创建一个简单的测试图
def create_test_graph():
    G = nx.Graph()
    # 添加节点和边
    nodes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    edges = [
        (0, 1, 0.8),
        (1, 2, 0.7),
        (2, 3, 0.6),
        (3, 4, 0.5),
        (5, 6, 0.8),
        (6, 7, 0.7),
        (7, 8, 0.6),
        (8, 9, 0.5),
        (2, 6, 0.3)  # 连接两个子图
    ]
    
    for u, v, w in edges:
        G.add_edge(u, v, weight=w)
    
    return G

# 主测试函数
def main():
    print("开始测试负激活节点统计...")
    
    # 创建测试图
    G = create_test_graph()
    
    # 定义参数
    seed = [0]  # 正向种子节点
    SN = [5]    # 负向种子节点
    comAndFS = [1, 2, 3, 4, 6, 7, 8, 9]  # fitness space中的节点
    hop = 2     # 传播步数
    
    print(f"\n测试参数:")
    print(f"正向种子节点: {seed}")
    print(f"负向种子节点: {SN}")
    print(f"Fitness Space: {comAndFS}")
    print(f"传播步数: {hop}")
    
    # 运行测试
    effect, num_neg = fitness_C_7(seed, G, SN, comAndFS, hop)
    
    print(f"\n测试完成!")
    print(f"Effect值: {effect}")
    print(f"负激活节点数量: {num_neg}")

if __name__ == "__main__":
    main()

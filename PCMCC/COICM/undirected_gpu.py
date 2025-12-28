import re
import os, sys
import time
import copy
import heapq
import random
import leidenalg
import igraph as ig
import networkx as nx
from collections import defaultdict
import torch
from select_SN import select_SN

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# log recorder
class Logger(object):

    def __init__(self, stream=sys.stdout):
        output_dir = "../../results/undirected" 
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        log_name = "facebook_log_gpu.txt"
        filename = os.path.join(output_dir, log_name)

        self.terminal = stream
        self.log = open(filename, 'a+')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

class GraphGPU:
    def __init__(self, nx_graph, nodes_list):
        self.nodes = list(nodes_list)
        self.node_to_idx = {node: i for i, node in enumerate(self.nodes)}
        self.num_nodes = len(self.nodes)
        
        # Create adjacency matrix
        # shape: (num_nodes, num_nodes)
        # We use dense tensor for small graphs (< 10k nodes)
        adj = torch.zeros((self.num_nodes, self.num_nodes), device=device, dtype=torch.float32)
        
        for u, v, data in nx_graph.edges(data=True):
            if u in self.node_to_idx and v in self.node_to_idx:
                idx_u = self.node_to_idx[u]
                idx_v = self.node_to_idx[v]
                w = data.get('weight', 0.0)
                adj[idx_u, idx_v] = w
                adj[idx_v, idx_u] = w # Undirected
                
        self.adj = adj

def fitness_gpu(seeds_indices, sn_indices, graph_gpu, com_and_fs_indices, hop):
    """
    Vectorized fitness calculation using PyTorch.
    seeds_indices: list or tensor of seed node indices
    sn_indices: list or tensor of SN node indices
    graph_gpu: GraphGPU object
    com_and_fs_indices: indices of nodes to sum effect over
    """
    num_nodes = graph_gpu.num_nodes
    adj = graph_gpu.adj
    
    # Initialize probabilities
    pP = torch.zeros(num_nodes, device=device)
    pN = torch.zeros(num_nodes, device=device)
    apP = torch.zeros(num_nodes, device=device)
    apN = torch.zeros(num_nodes, device=device)
    
    # Set initial seeds
    if len(seeds_indices) > 0:
        pP[seeds_indices] = 1.0
        apP[seeds_indices] = 1.0
        
    if len(sn_indices) > 0:
        pN[sn_indices] = 1.0
        apN[sn_indices] = 1.0
        
    # Simulation loop
    for h in range(hop):
        # Positive propagation
        # P_weighted = pP.view(-1, 1) * adj
        # temppP = product(1 - P_weighted) -> exp(sum(log(1-P_weighted)))
        
        P_weighted = pP.view(-1, 1) * adj
        P_weighted = torch.clamp(P_weighted, max=1.0 - 1e-7)
        term = torch.log(1.0 - P_weighted)
        temppP = torch.exp(torch.sum(term, dim=0))
        
        # Negative propagation
        N_weighted = pN.view(-1, 1) * adj
        N_weighted = torch.clamp(N_weighted, max=1.0 - 1e-7)
        term_n = torch.log(1.0 - N_weighted)
        temppN = torch.exp(torch.sum(term_n, dim=0))
        
        # Update probabilities
        # Logic: pP_new = (1 - temppP) * (1 - apN) * (1 - apP)
        pP_new = (1.0 - temppP) * (1.0 - apN) * (1.0 - apP)
        
        # Logic: pN_new = temppP * (1 - temppN) * (1 - apN) * (1 - apP)
        pN_new = temppP * (1.0 - temppN) * (1.0 - apN) * (1.0 - apP)
        
        # Update accumulated
        apP = apP + pP_new
        apN = apN + pN_new
        
        pP = pP_new
        pN = pN_new
        
    if len(com_and_fs_indices) > 0:
        effect = torch.sum(apN[com_and_fs_indices])
    else:
        effect = torch.tensor(0.0, device=device)
        
    return effect.item()

def communityDivision_1(G_1, C_1):
    c_G = G_1.copy()
    c_g = copy.deepcopy(ig.Graph.TupleList(list(c_G.edges(data='weight')), directed=False, edge_attrs=['weight']))
    c_part = leidenalg.find_partition(c_g, leidenalg.ModularityVertexPartition, weights=c_g.es['weight'],
                                      n_iterations=-1)
    print("------------------- communityDivision start -----------------------")
    print(f"Modularity: {c_part.modularity}\n")

    rs_part = []
    pattern1 = re.compile(r"(?<=])[^][]+(?=\n\[)")
    matches = pattern1.findall(str(c_part) + "\n[")
    for match1 in matches:
        pattern2 = r"\d+"
        numbers = [int(match2) for match2 in re.findall(pattern2, match1)]
        rs_part.append(numbers)

    while len(rs_part) != C_1:
        if len(rs_part) > C_1:
            lengths = [len(lst) for lst in rs_part]
            min_indices = sorted(range(len(lengths)), key=lambda k: lengths[k])[:2]
            rs_part[min_indices[1]].extend(rs_part[min_indices[0]])
            del rs_part[min_indices[0]]
        if len(rs_part) < C_1:
            lengths = [len(lst) for lst in rs_part]
            max_indices = sorted(range(len(lengths)), key=lambda k: lengths[k])[-1]
            a_G = G_1.subgraph(rs_part[max_indices]).copy()
            a_g = copy.deepcopy(
                ig.Graph.TupleList(list(a_G.edges(data='weight')), directed=False, edge_attrs=['weight']))
            a_part = leidenalg.find_partition(a_g, leidenalg.ModularityVertexPartition, weights=a_g.es['weight'],
                                              n_iterations=-1)
            a_rs_part = []
            pattern_a = re.compile(r"(?<=])[^][]+(?=\n\[)")
            matches_a = pattern_a.findall(str(a_part) + "\n[")
            for match1_a in matches_a:
                pattern2_a = r"\d+"
                numbers_a = [int(match2_a) for match2_a in re.findall(pattern2_a, match1_a)]
                a_rs_part.append(numbers_a)
            del rs_part[max_indices]
            rs_part[max_indices:max_indices] = copy.deepcopy(a_rs_part)

    for i in range(C_1):
        print(len(rs_part[i]))

    print("------------------- communityDivision end ------------------------")
    print("\n")

    return rs_part


def negativeProbability_2(G_2, SN_2, fitnessSpace_2, hop_2, all_FP_2):
    rs_N_p = {}
    ZN_f = []
    ZN_f.append(SN_2)
    for h in range(1, hop_2 + 1):
        ZN_f.append([])
    pN_f = defaultdict(lambda: 0)
    apN_f = defaultdict(lambda: 0)
    for v in SN_2:
        pN_f[v, 0] = 1
        for h in range(hop_2 + 1):
            apN_f[v, h] = 1
    for h in range(hop_2):
        temppN_f = defaultdict(lambda: 1)
        for v in ZN_f[h]:
            W_f = list(G_2.neighbors(v))
            ZN_f[h + 1] += W_f
            for w in W_f:
                temppN_f[w] *= (1 - pN_f[v, h] * G_2[v][w]['weight'])
        ZN_f[h + 1] = list(set(ZN_f[h + 1]))
        for v in ZN_f[h + 1]:
            pN_f[v, h + 1] = (1 - temppN_f[v]) * (1 - apN_f[v, h])
            for tau_f in range(h + 1, hop_2 + 1):
                apN_f[v, tau_f] = apN_f[v, h] + pN_f[v, h + 1]

    for u in fitnessSpace_2:
        for t in range(1, hop_2 + 1):
            rs_N_p[u, t] = pN_f[u, t]

    for u in all_FP_2:
        for t in range(1, hop_2 + 1):
            rs_N_p[u, t] = 0
    return rs_N_p


def positiveScore_3(G_3, fitnessSpace_3, searchSpace_3, hop_3, N_prob_3):
    rs_P_S = {}

    for u in searchSpace_3:
        predecessors = defaultdict(lambda: [])
        rs_P_S[u] = 0
        one_hop_neighbors = []
        two_hop_neighbors = []
        for v in G_3.neighbors(u):
            one_hop_neighbors.append(v)
            for w in G_3.neighbors(v):
                two_hop_neighbors.append(w)
                predecessors[w].append(v)

        oneAndF = set(one_hop_neighbors).intersection(set(fitnessSpace_3))
        two_hop_neighbors = set(two_hop_neighbors).intersection(set(fitnessSpace_3)) - set([u])

        twoAndOne = two_hop_neighbors.intersection(oneAndF)
        two_One = two_hop_neighbors - oneAndF

        for t in range(1, hop_3 + 1):
            rs_P_S[u] += N_prob_3[u, t]

        for v in oneAndF:
            for t in range(1, hop_3 + 1):
                rs_P_S[u] += G_3[u][v]['weight'] * N_prob_3[v, t]

        for w in twoAndOne:
            temp_p = 1
            for v in set(predecessors[w]):
                temp_p *= (1 - G_3[u][v]['weight'] * G_3[v][w]['weight'])
            for t in range(2, hop_3 + 1):
                rs_P_S[u] += (1 - G_3[u][w]['weight']) * (1 - temp_p) * (1 - N_prob_3[w, 1]) * N_prob_3[w, t]

        for w in two_One:
            temp_p = 1
            for v in set(predecessors[w]):
                temp_p *= (1 - G_3[u][v]['weight'] * G_3[v][w]['weight'])
            for t in range(2, hop_3 + 1):
                rs_P_S[u] += (1 - temp_p) * (1 - N_prob_3[w, 1]) * N_prob_3[w, t]

    return rs_P_S

def populationInitialization_4(i_4, j_4, Ni_4, comAndSeai_4, community_ki_4):
    population_4 = {}
    for I in range(Ni_4):
        population_4[i_4, j_4, I] = random.sample(comAndSeai_4, k=community_ki_4)

    return population_4

def sample(l1, w1, k):
    l = copy.deepcopy(l1)
    w = copy.deepcopy(w1)

    randoms = [random.random() for i in range(k)]

    total_w = 0
    bu = {}
    bd = {}
    for u in l:
        bd[u] = total_w
        bu[u] = total_w + w[u]
        total_w += w[u]

    l_new = copy.deepcopy(l)
    total_w_new = total_w

    rs = []
    for r in randoms:
        total_w = total_w_new
        r_total_w = r * total_w
        l = copy.deepcopy(l_new)
        a = 0
        count = 0
        for u in l:
            if a == 1:
                bd[u] -= w[rs[-1]]
                bu[u] -= w[rs[-1]]

            if (a == 0) and (r_total_w > bd[u]) and (r_total_w <= bu[u]):
                rs.append(u)
                del l_new[count]
                a = 1

            count += 1

        total_w_new = total_w - w[rs[-1]]

    return rs

def convert_Index_10(islands_10):
    res_10_1 = {}
    res_10_2 = {}
    res_10_3 = {}
    count_10_1 = 0
    count_10_2 = 0
    count_10_3 = 0

    for i in range(len(islands_10)):
        for j in range(len(islands_10[i])):
            res_10_3[i, j] = count_10_3
            res_10_3[count_10_3] = (i, j)
            count_10_3 += 1

            for N in range(len(islands_10[i][j])):
                res_10_2[i, j, N] = count_10_2
                res_10_2[count_10_2] = (i, j, N)
                count_10_2 += 1

                for X in range(len(islands_10[i][j][N])):
                    res_10_1[i, j, N, X] = count_10_1
                    res_10_1[count_10_1] = (i, j, N, X)
                    count_10_1 += 1

    return res_10_1, res_10_2, res_10_3

def mergeCommunity_12(merge_12, communityList_12, community_k_12, islands_12, islandsEffect_12, comRes_12, N_12,
                      G_12, subG_12, SN_12, fitnessSpace_12, hop_12, s_t_l_12, curT_12, comGenAcc_12, comBen_12,
                      P_score_12, gama_12, searchSpace_12):
    if len(communityList_12) > 2:
        lengths = [len(sublist) for sublist in communityList_12]
        lengthsIndex = [i[0] for i in sorted(enumerate(lengths), key=lambda x: x[1], reverse=True)]
        max_connection_index = [-1 for _ in range(len(communityList_12))]

        merge_score = defaultdict(lambda: -1)

        for i in range(len(communityList_12)):
            if merge_12[i] == -1:
                max_connection_i = 0
                max_connection_index_i = 0
                for j in range(len(communityList_12)):
                    if i != j:
                        if merge_score[i, j] == -1:

                            merge_score[i, j] = 0
                            merge_score[j, i] = 0

                            for edge in list(nx.edge_boundary(G_12, communityList_12[i], communityList_12[j])):

                                one_score = 0

                                for v in subG_12[j].neighbors(edge[1]):
                                    one_score += subG_12[j][edge[1]][v]['weight']

                                merge_score[i, j] += (one_score * G_12[edge[0]][edge[1]]['weight'])
                                merge_score[j, i] += (one_score * G_12[edge[0]][edge[1]]['weight'])

                                one_score = 0

                                for v in subG_12[i].neighbors(edge[0]):
                                    one_score += subG_12[i][edge[0]][v]['weight']

                                merge_score[i, j] += (one_score * G_12[edge[0]][edge[1]]['weight'])
                                merge_score[j, i] += (one_score * G_12[edge[0]][edge[1]]['weight'])

                        if max_connection_i <= merge_score[i, j]:
                            max_connection_i = merge_score[i, j]
                            max_connection_index_i = j

                max_connection_index[i] = max_connection_index_i

        for i in lengthsIndex:
            if merge_12[i] == -1:
                if (merge_12[max_connection_index[i]] == -2) or (merge_12[max_connection_index[i]] == -1):
                    merge_12[i] = i
                    merge_12[max_connection_index[i]] = i
                else:
                    merge_12[i] = merge_12[max_connection_index[i]]

    elif len(communityList_12) == 2:
        if merge_12[0] == -1 and merge_12[1] == -2:
            merge_12[0] = 0
            merge_12[1] = 0
        elif merge_12[0] == -2 and merge_12[1] == -1:
            merge_12[0] = 1
            merge_12[1] = 1
        elif merge_12[0] == -1 and merge_12[1] == -1:
            merge_12[0] = 0
            merge_12[1] = 0

    my_dict = {}
    for i, value in enumerate(merge_12):
        if value >= 0:
            if value not in my_dict:
                my_dict[value] = [i]
            else:
                my_dict[value].append(i)

    sorted_dict = sorted(my_dict.items(), key=lambda x: len(x[1]), reverse=True)
    toBeMerged = [x[1] for x in sorted_dict if len(x[1]) > 1]

    for i in range(len(communityList_12)):
        rs = True
        for row in toBeMerged:
            for element in row:
                if element == i:
                    rs = False
        if rs:
            toBeMerged.append([i])

    sort_islands_copy = copy.deepcopy(islands_12)
    for i in range(len(communityList_12)):
        for j in range(comRes_12[i]):
            islands_12[i][j].sort(key=lambda x: islandsEffect_12[i][j][sort_islands_copy[i][j].index(x)])

    for i in range(len(communityList_12)):
        for j in range(comRes_12[i]):
            islandsEffect_12[i][j].sort()

    new_comRes = []

    for i in range(len(toBeMerged)):
        new_comRes.append(0)
        for j in toBeMerged[i]:
            new_comRes[i] += comRes_12[j]

    new_community_k = []
    new_communityList = []

    new_islands = [[[] for j in range(new_comRes[i])] for i in range(len(toBeMerged))]
    new_islandsEffect = [[[] for j in range(new_comRes[i])] for i in range(len(toBeMerged))]

    for i in range(len(toBeMerged)):
        new_community_k.append(0)
        new_communityList.append([])

        for j in toBeMerged[i]:
            new_community_k[i] += community_k_12[j]
            new_communityList[i] += communityList_12[j]

    comAndSea_12 = []
    comAndFS_12 = []
    comOrSN_12 = []
    comGs_12 = []
    gamaCom_12 = []
    for i in range(len(new_communityList)):
        comAndSea_12.append(list(set(searchSpace_12).intersection(set(new_communityList[i]))))  # Calculate comAndSea in advance
        comAndFS_12.append(
            list(set(new_communityList[i]).intersection(set(fitnessSpace_12))))  # Calculate comAndFS in advance
        comOrSN_12.append(list(set(new_communityList[i] + SN_12)))  # Calculate comOrSN in advance
        tempsubGi = G_12.subgraph(comOrSN_12[i])
        subGi = nx.Graph(tempsubGi.edges(data=True))
        subGi.add_nodes_from(comOrSN_12[i])
        comGs_12.append(subGi.copy())  # Calculate subGi in advance
        gamaCom_12.append(
            heapq.nlargest(min(int(round(gama_12 * new_community_k[i])), len(comAndSea_12[i])), comAndSea_12[i],
                           key=lambda x: P_score_12[x]))  # Calculate gamaCom in advance
    
    for i in range(len(toBeMerged)):
        temp_islands_i = [[[] for N in range(N_12)] for J in range(new_comRes[i])]
        for J in range(new_comRes[i]):
            for N in range(N_12):
                for j in toBeMerged[i]:
                    temp_islands_i[J][N] += islands_12[j][J % comRes_12[j]][N]

        for J in range(new_comRes[i]):
            for N in range(N_12):
                new_islands[i][J].append(temp_islands_i[J][N])

    for i in range(len(toBeMerged)):
        if len(toBeMerged[i]) == 1:
            for J in range(new_comRes[i]):
                for N in range(N_12):
                    new_islandsEffect[i][J].append(islandsEffect_12[toBeMerged[i][0]][J][N])
        else:
            for J in range(new_comRes[i]):
                for N in range(N_12):
                    new_islandsEffect[i][J].append(
                        fitness_C_7_compat(new_islands[i][J][N], comGs_12[i], SN_12,
                                    comAndFS_12[i], hop_12))

    minE = [0 for i in range(len(toBeMerged))]
    for i in range(len(toBeMerged)):
        minE_i = new_islandsEffect[i][0][0]
        for j in range(new_comRes[i]):
            if min(new_islandsEffect[i][j]) < minE_i:
                minE_i = min(new_islandsEffect[i][j])
        minE[i] = minE_i

    new_comGenAcc_12 = [-1 for i in range(len(toBeMerged))]
    new_comBen_12 = [-1 for i in range(len(toBeMerged))]

    for i in range(len(toBeMerged)):
        if len(toBeMerged[i]) == 1:
            new_comGenAcc_12[i] = comGenAcc_12[toBeMerged[i][0]]
            new_comBen_12[i] = comBen_12[toBeMerged[i][0]]

            s_t_l_12[i, curT_12, len(toBeMerged)] = (minE[i], comGenAcc_12[toBeMerged[i][0]])
            s_t_l_12[i, new_comBen_12[i], len(toBeMerged)] = copy.deepcopy(
                s_t_l_12[toBeMerged[i][0], comBen_12[toBeMerged[i][0]], len(communityList_12)])

        else:
            new_comGenAcc_12[i] = comGenAcc_12[toBeMerged[i][0]]
            new_comBen_12[i] = curT_12

            for j in toBeMerged[i]:
                if new_comGenAcc_12[i] > comGenAcc_12[j]:
                    new_comGenAcc_12[i] = comGenAcc_12[j]

            s_t_l_12[i, curT_12, len(toBeMerged)] = (minE[i], new_comGenAcc_12[i])

    return new_islands, new_islandsEffect, new_communityList, new_community_k, \
           s_t_l_12, new_comGenAcc_12, new_comBen_12, comAndSea_12, comAndFS_12, comOrSN_12, comGs_12, gamaCom_12, new_comRes

def fitness_C_7_compat(seed, G, SN, comAndFS, hop):
    """
    Compatibility wrapper for fitness calculation using GPU.
    Creates a temporary GraphGPU object.
    """
    nodes = list(G.nodes)
    g_gpu = GraphGPU(G, nodes)
    
    # Map indices
    seed_indices = [g_gpu.node_to_idx[u] for u in seed if u in g_gpu.node_to_idx]
    sn_indices = [g_gpu.node_to_idx[u] for u in SN if u in g_gpu.node_to_idx]
    com_indices = [g_gpu.node_to_idx[u] for u in comAndFS if u in g_gpu.node_to_idx]
    
    return fitness_gpu(seed_indices, sn_indices, g_gpu, com_indices, hop)

def evolve_one_step(population, effect, graph_gpu, G_nx, SN, comAndFS, hop, 
                   comAndSea, community_k, Ni, cOne, cTwo, P_score, gamaCom, N_prob):
    """
    Performs one generation of evolution for a subpopulation.
    Returns updated population and effect.
    """
    if community_k == 0:
        return population, effect

    # Pre-map indices for GPU fitness
    sn_indices = [graph_gpu.node_to_idx[u] for u in SN if u in graph_gpu.node_to_idx]
    com_fs_indices = [graph_gpu.node_to_idx[u] for u in comAndFS if u in graph_gpu.node_to_idx]
    
    # We have Ni individuals in this subpopulation
    # population is a list of lists (individuals)
    # effect is a list of floats
    
    # Find best individual in this subpopulation
    indexS1 = effect.index(min(effect))
    
    for I in range(Ni):
        if I == indexS1:
            continue
            
        S1 = copy.deepcopy(population[indexS1])
        SI = copy.deepcopy(population[I])
        
        repeatS1 = 0
        repeatSI = 0
        
        # Crossover
        for J in range(community_k):
            if random.random() < cOne:
                if random.random() < cTwo:  # two-way cross
                    temp = S1[J]
                    if SI[J] not in S1 or SI[J] == S1[J]:
                        S1[J] = SI[J]
                    else:
                        S1[J] = -1
                        repeatS1 += 1
                    if temp not in SI or temp == SI[J]:
                        SI[J] = temp
                    else:
                        SI[J] = -1
                        repeatSI += 1
                else:  # one-way cross
                    if S1[J] not in SI or S1[J] == SI[J]:
                        SI[J] = S1[J]
                    else:
                        SI[J] = -1
                        repeatSI += 1
                        
        if repeatS1 != 0:
            replaceS1 = sample(list(set(comAndSea) - set(S1)), P_score, repeatS1)
            J_idx = 0
            for e in range(community_k):
                if S1[e] == -1:
                    S1[e] = replaceS1[J_idx]
                    J_idx += 1

        if repeatSI != 0:
            replaceSI = sample(list(set(comAndSea) - set(SI)), P_score, repeatSI)
            J_idx = 0
            for e in range(community_k):
                if SI[e] == -1:
                    SI[e] = replaceSI[J_idx]
                    J_idx += 1
                    
        # Evaluate on GPU
        # Need to map S1, SI to indices
        s1_indices = [graph_gpu.node_to_idx[u] for u in S1 if u in graph_gpu.node_to_idx]
        si_indices = [graph_gpu.node_to_idx[u] for u in SI if u in graph_gpu.node_to_idx]
        
        effectS1 = fitness_gpu(s1_indices, sn_indices, graph_gpu, com_fs_indices, hop)
        effectSI = fitness_gpu(si_indices, sn_indices, graph_gpu, com_fs_indices, hop)
        
        if effectS1 < effect[indexS1]:
            population[indexS1] = S1
            effect[indexS1] = effectS1
            # Update best index if changed
            indexS1 = effect.index(min(effect))
            
        if effectSI < effect[I]:
            population[I] = SI
            effect[I] = effectSI
            
    # Local Search
    # Find best again (might have changed)
    indexS1 = effect.index(min(effect))
    S1 = copy.deepcopy(population[indexS1])
    
    # We use CPU logic for neighbor traversal as it involves complex filtering
    discount_P_score_diff = []
    replace_discount_P_score_diff = {}

    for I in range(community_k):
        rs = 0
        predecessors = defaultdict(lambda: [])
        one_hop_neighbors = []
        two_hop_neighbors = []

        for v in G_nx.neighbors(S1[I]):
            one_hop_neighbors.append(v)
            for w in G_nx.neighbors(v):
                two_hop_neighbors.append(w)
                predecessors[w].append(v)

        oneAndF = set(one_hop_neighbors).intersection(set(comAndFS)) - set(S1)
        two_hop_neighbors = set(two_hop_neighbors).intersection(set(comAndFS)) - set(S1)

        twoAndOne = two_hop_neighbors.intersection(oneAndF)
        two_one = two_hop_neighbors - oneAndF

        for t in range(1, hop + 1):
            rs += N_prob[S1[I], t]

        for v in oneAndF:
            for t in range(1, hop + 1):
                rs += G_nx[S1[I]][v]['weight'] * N_prob[v, t]

        for w in twoAndOne:
            temp_p = 1
            for v in set(predecessors[w]):
                temp_p *= (1 - G_nx[S1[I]][v]['weight'] * G_nx[v][w]['weight'])
            for t in range(2, hop + 1):
                rs += (1 - G_nx[S1[I]][w]['weight']) * (1 - temp_p) * (1 - N_prob[w, 1]) * N_prob[w, t]

        for w in two_one:
            temp_p = 1
            for v in set(predecessors[w]):
                temp_p *= (1 - G_nx[S1[I]][v]['weight'] * G_nx[v][w]['weight'])
            for t in range(2, hop + 1):
                rs += (1 - temp_p) * (1 - N_prob[w, 1]) * N_prob[w, t]

        temp = 1
        rs1 = 0

        for u in set(S1).intersection(set(G_nx.neighbors(S1[I]))):
            temp *= (1 - G_nx[u][S1[I]]['weight'])

        for t in range(1, hop + 1):
            rs1 += (1 - temp) * N_prob[S1[I], t]

        for v in oneAndF:
            for t in range(2, hop + 1):
                rs1 += (1 - temp) * G_nx[S1[I]][v]['weight'] * N_prob[v, t]

        discount_P_score_diff.append(rs - rs1)

    I = discount_P_score_diff.index(min(discount_P_score_diff))
    rn = S1[I]

    Sbest = copy.deepcopy(S1)
    for nn in (set(gamaCom) - set(Sbest)):
        S1[I] = nn

        rs = 0

        predecessors = defaultdict(lambda: [])
        one_hop_neighbors = []
        two_hop_neighbors = []

        for v in G_nx.neighbors(S1[I]):
            one_hop_neighbors.append(v)
            for w in G_nx.neighbors(v):
                two_hop_neighbors.append(w)
                predecessors[w].append(v)

        oneAndF = set(one_hop_neighbors).intersection(set(comAndFS)) - set(S1)
        two_hop_neighbors = set(two_hop_neighbors).intersection(set(comAndFS)) - set(S1)

        twoAndOne = two_hop_neighbors.intersection(oneAndF)
        two_one = two_hop_neighbors - oneAndF

        for t in range(1, hop + 1):
            rs += N_prob[S1[I], t]

        for v in oneAndF:
            for t in range(1, hop + 1):
                rs += G_nx[S1[I]][v]['weight'] * N_prob[v, t]

        for w in twoAndOne:
            temp_p = 1
            for v in set(predecessors[w]):
                temp_p *= (1 - G_nx[S1[I]][v]['weight'] * G_nx[v][w]['weight'])
            for t in range(2, hop + 1):
                rs += (1 - G_nx[S1[I]][w]['weight']) * (1 - temp_p) * (1 - N_prob[w, 1]) * N_prob[w, t]

        for w in two_one:
            temp_p = 1
            for v in set(predecessors[w]):
                temp_p *= (1 - G_nx[S1[I]][v]['weight'] * G_nx[v][w]['weight'])
            for t in range(2, hop + 1):
                rs += (1 - temp_p) * (1 - N_prob[w, 1]) * N_prob[w, t]

        temp = 1
        rs1 = 0

        for u in set(S1).intersection(set(G_nx.neighbors(S1[I]))):
            temp *= (1 - G_nx[u][S1[I]]['weight'])

        for t in range(1, hop + 1):
            rs1 += (1 - temp) * N_prob[S1[I], t]

        for v in oneAndF:
            for t in range(2, hop + 1):
                rs1 += (1 - temp) * G_nx[S1[I]][v]['weight'] * N_prob[v, t]

        replace_discount_P_score_diff[nn] = rs - rs1

    S1[I] = -1
    rmax = discount_P_score_diff[I]

    for nn in list(set(gamaCom) - set(Sbest)):
        if replace_discount_P_score_diff[nn] >= rmax:
            rmax = replace_discount_P_score_diff[nn]
            rn = nn

    S1[I] = rn
    
    # Evaluate new candidate on GPU
    s1_indices = [graph_gpu.node_to_idx[u] for u in S1 if u in graph_gpu.node_to_idx]
    effectS1 = fitness_gpu(s1_indices, sn_indices, graph_gpu, com_fs_indices, hop)

    if effectS1 < effect[indexS1]:
        population[indexS1] = S1
        effect[indexS1] = effectS1
    
    return population, effect

def simulate_propagation(G_sim, positive_seeds, negative_seeds, max_hop):
    """模拟在给定正面种子和负面种子下的信息传播，返回最终负面激活的节点集合。"""
    pos_activated = set(positive_seeds)        # 已被正面激活的节点集合
    neg_activated = set(negative_seeds)        # 已被负面激活的节点集合
    current_pos_frontier = set(positive_seeds) # 当前轮新激活的正面节点
    current_neg_frontier = set(negative_seeds) # 当前轮新激活的负面节点
    # 逐步扩散最多 max_hop 轮
    for h in range(max_hop):
        new_pos_frontier = set()
        new_neg_frontier = set()
        # 正面信息先传播
        for u in current_pos_frontier:
            for w in G_sim.neighbors(u):
                if w not in pos_activated and w not in neg_activated:
                    prob = G_sim[u][w]['weight']
                    if random.random() < prob:         # 按概率激活
                        pos_activated.add(w)
                        new_pos_frontier.add(w)
        # 负面信息后传播
        for u in current_neg_frontier:
            for w in G_sim.neighbors(u):
                # 邻居尚未被任何信息激活，且本轮未被正面激活，负面才能尝试
                if w not in pos_activated and w not in neg_activated and w not in new_pos_frontier:
                    prob = G_sim[u][w]['weight']
                    if random.random() < prob:
                        neg_activated.add(w)
                        new_neg_frontier.add(w)
        # 更新当前激活前沿
        current_pos_frontier = new_pos_frontier
        current_neg_frontier = new_neg_frontier
        # 若没有新激活节点，提前结束传播
        if not current_pos_frontier and not current_neg_frontier:
            break
    return neg_activated

if __name__ == "__main__":
    sys.stdout = Logger(sys.stdout)  # record log

    SN_size = 50
    SN_dic = {}
    SN_dic["facebook"] = select_SN("facebook", SN_size)
    
    SN_dic["HR"] = select_SN("HR", SN_size)
    
    SN_dic["BA3000"] = select_SN("BA3000", SN_size)
    
    SN_dic["ER3000"] = select_SN("ER3000", SN_size)
    
    SN_dic["RG3000"] = select_SN("RG3000", SN_size)
    
    SN_dic["WS3000"] = select_SN("WS3000", SN_size)

    graphs = ["facebook"]

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

        # Test settings: k=20 only, repeats=1
        for k in [20, 110, 200]:

            repeats = 2

            for r in range(repeats):
                print("\nPCMCC GPU", file_name, k, r + 1)

                start_time = time.time()

                Ni = 20
                cOne = 0.3
                cTwo = 0.3
                C = 16
                comRes = [1 for i in range(C)]  # Record community computing resources

                theta = 1
                s_l = 3
                s_g = 3
                maxT = 20

                hop = 2

                alpha = 12  # Search space reduction
                beta = 2  # MaxT is not the final generation, it is just the maximum merging algebra
                gama = 6  # Search scope, gama * ki

                fitnessSpace = []
                curUnErgodic = copy.deepcopy(SN)
                hop_f = 0
                while hop_f <= hop:
                    curErgodic = copy.deepcopy(curUnErgodic)
                    fitnessSpace += curErgodic
                    if hop_f == hop:
                        break
                    else:
                        curUnErgodic = []
                        for u in curErgodic:
                            for v in G.neighbors(u):
                                curUnErgodic.append(v)
                        curUnErgodic = list(set(curUnErgodic) - set(fitnessSpace))
                        hop_f += 1
                fitnessSpace = list(set(fitnessSpace) - set(SN))

                searchSpace = []
                curUnErgodic = copy.deepcopy(fitnessSpace)
                hop_s = 0
                while hop_s <= hop:
                    curErgodic = copy.deepcopy(curUnErgodic)
                    searchSpace += curErgodic

                    if hop_s == hop:
                        break

                    else:
                        curUnErgodic = []
                        for u in curErgodic:
                            for v in G.neighbors(u):
                                curUnErgodic.append(v)
                        curUnErgodic = list(set(curUnErgodic) - set(searchSpace + SN))
                        hop_s += 1

                allNodes = copy.deepcopy(searchSpace + SN)

                all_FP = list(set(allNodes) - set(fitnessSpace))  # Calculate in advance

                Gs = G.subgraph(allNodes).copy()  # Modify G to make it smaller

                communityList = communityDivision_1(Gs, C)

                searchSpaceReduction = heapq.nlargest(min(int(round(alpha * k)), len(searchSpace)), searchSpace,
                                                      key=lambda x: Gs.degree(x))

                searchSpace = copy.deepcopy(searchSpaceReduction)

                N_prob = negativeProbability_2(Gs, SN, fitnessSpace, hop, all_FP)  # Optimize the calculation of N_Prbo

                P_score = positiveScore_3(Gs, fitnessSpace, searchSpace, hop, N_prob)  # Optimize the calculation of P_score

                belongTo = {}
                for i in range(C):
                    for u in communityList[i]:
                        belongTo[u] = i

                communityNegativeImpact = [0 for i in range(C)]
                for u in fitnessSpace:
                    for t in range(1, hop + 1):
                        communityNegativeImpact[belongTo[u]] += N_prob[u, t]

                comAndSea = []  # Calculate comAndSea in advance
                for i in range(C):
                    comAndSea.append(list(set(searchSpace).intersection(set(communityList[i]))))

                sum_NegativeImpact = sum(communityNegativeImpact)
                community_k = [0 for i in range(C)]

                lengths = [len(sublist) for sublist in comAndSea]
                max_i = lengths.index(max(lengths))

                for i in range(C):
                    if i != max_i:
                        community_k[i] = min(int(round(k * communityNegativeImpact[i] / sum_NegativeImpact)),
                                             len(comAndSea[i]))

                community_k[max_i] = min(int(k - sum(community_k)), len(comAndSea[max_i]))

                print("comK:", community_k)
                print("sumK:", sum(community_k))

                if sum(community_k) != k:
                    break

                # Initialization
                population = []
                for i in range(C):
                    population.append([])
                    for j in range(comRes[i]):
                        population[i].append([])
                        pop_dict = populationInitialization_4(i, j, Ni, comAndSea[i], community_k[i])
                        # Flatten dict to list
                        for I in range(Ni):
                            population[i][j].append(pop_dict[i, j, I])

                comAndFS = []
                comOrSN = []
                comGs = []
                gamaCom = []
                graphGPUs = [] # Store GPU graph objects
                
                for i in range(C):
                    comAndFS.append(list(set(communityList[i]).intersection(set(fitnessSpace))))
                    comOrSN.append(list(set(communityList[i] + SN)))
                    tempsubGi = Gs.subgraph(comOrSN[i])
                    subGi = nx.Graph(tempsubGi.edges(data=True))
                    subGi.add_nodes_from(comOrSN[i])
                    comGs.append(subGi.copy())
                    
                    # Create GPU Graph
                    graphGPUs.append(GraphGPU(subGi, list(subGi.nodes)))
                    
                    gamaCom.append(
                        heapq.nlargest(min(int(round(gama * community_k[i])), len(comAndSea[i])), comAndSea[i],
                                       key=lambda x: P_score[x]))

                # Calculate initial effect
                effect = []
                for i in range(C):
                    effect.append([])
                    for j in range(comRes[i]):
                        effect[i].append([])
                        for I in range(Ni):
                            # GPU calculation
                            g_gpu = graphGPUs[i]
                            ind = population[i][j][I]
                            
                            # Map to indices
                            seed_idx = [g_gpu.node_to_idx[u] for u in ind if u in g_gpu.node_to_idx]
                            sn_idx = [g_gpu.node_to_idx[u] for u in SN if u in g_gpu.node_to_idx]
                            com_fs_idx = [g_gpu.node_to_idx[u] for u in comAndFS[i] if u in g_gpu.node_to_idx]
                            
                            val = fitness_gpu(seed_idx, sn_idx, g_gpu, com_fs_idx, hop)
                            effect[i][j].append(val)

                s_t_g = {}
                s_t_l = {}

                curT = 0

                if len(communityList) == 1:
                    e_g_b = 0

                for i in range(C):
                    minE_i = effect[i][0][0]
                    for j in range(comRes[i]):
                        if min(effect[i][j]) < minE_i:
                            minE_i = min(effect[i][j])
                    s_t_l[i, 0, C] = (minE_i, 0)

                comGenAcc = [0 for i in range(C)]  # Record the operating generations of each community
                comBen = [0 for i in range(C)]  # Record the baseline algebra comparison for each community

                # GPU Graph for full Gs (for global evaluation)
                global_gpu_graph = GraphGPU(Gs, list(Gs.nodes))
                global_sn_idx = [global_gpu_graph.node_to_idx[u] for u in SN if u in global_gpu_graph.node_to_idx]
                global_fs_idx = [global_gpu_graph.node_to_idx[u] for u in fitnessSpace if u in global_gpu_graph.node_to_idx]

                while True:
                    bestS = []
                    for i in range(len(communityList)):
                        min_islandsEffect_i = []
                        for j in range(comRes[i]):
                            minE = min(effect[i][j])
                            min_islandsEffect_i.append([effect[i][j].index(minE), minE, j])
                        rs_i_min = [copy.deepcopy(min_islandsEffect_i[0][0]), copy.deepcopy(min_islandsEffect_i[0][1]),
                                    0]
                        for j in range(comRes[i]):
                            if min_islandsEffect_i[j][1] < rs_i_min[1]:
                                rs_i_min = [copy.deepcopy(min_islandsEffect_i[j][0]),
                                            copy.deepcopy(min_islandsEffect_i[j][1]), j]
                        bestS += population[i][rs_i_min[2]][rs_i_min[0]]

                    if len(communityList) != 1:
                        # Global evaluation using GPU
                        bs_idx = [global_gpu_graph.node_to_idx[u] for u in bestS if u in global_gpu_graph.node_to_idx]
                        curEffect = fitness_gpu(bs_idx, global_sn_idx, global_gpu_graph, global_fs_idx, hop)
                        
                        s_t_g[curT] = curEffect
                        print("The optimal fitness value of the ", curT, " generation population:", curEffect)

                    else:
                        s_t_g[curT] = rs_i_min[1]
                        print("The optimal fitness value of the ", curT, " generation population:", rs_i_min[1])

                    print("bestS:", bestS)
                    print("Number of communities:", len(communityList))
                    # print("comRes:", comRes)

                    if curT == 0:
                        bestS0 = copy.deepcopy(bestS)
                        if len(communityList) != 1:
                            effect0 = curEffect
                        else:
                            effect0 = rs_i_min[1]

                    if len(communityList) != 1:
                        if curEffect < effect0 and curT > 0:
                            bestS0 = copy.deepcopy(bestS)
                            effect0 = curEffect
                    else:
                        if rs_i_min[1] < effect0 and curT > 0:
                            bestS0 = copy.deepcopy(bestS)
                            effect0 = rs_i_min[1]

                    print("effect0", effect0)

                    # Logic for "Add effect 0" (restoring bestS0 if worse)
                    if len(communityList) != 1:
                        if effect0 < curEffect and curT > 0:
                            print("Add effect 0")
                            bestS = copy.deepcopy(bestS0)
                            # Reset population maxes to bestS0 intersection
                            for i in range(len(communityList)):
                                max_islandsEffect_i = []
                                for j in range(comRes[i]):
                                    maxE = max(effect[i][j])
                                    max_islandsEffect_i.append([effect[i][j].index(maxE), maxE, j])
                                rs_i_max = [copy.deepcopy(max_islandsEffect_i[0][0]),
                                            copy.deepcopy(max_islandsEffect_i[0][1]), 0]
                                for j in range(comRes[i]):
                                    if max_islandsEffect_i[j][1] > rs_i_max[1]:
                                        rs_i_max = [copy.deepcopy(max_islandsEffect_i[j][0]),
                                                    copy.deepcopy(max_islandsEffect_i[j][1]), j]
                                population[i][rs_i_max[2]][rs_i_max[0]] = copy.deepcopy(
                                    list(set(bestS0).intersection(comAndSea[i])))
                                
                                # Re-eval
                                g_gpu = graphGPUs[i]
                                ind = population[i][rs_i_max[2]][rs_i_max[0]]
                                s_idx = [g_gpu.node_to_idx[u] for u in ind if u in g_gpu.node_to_idx]
                                sn_idx = [g_gpu.node_to_idx[u] for u in SN if u in g_gpu.node_to_idx]
                                com_fs_idx = [g_gpu.node_to_idx[u] for u in comAndFS[i] if u in g_gpu.node_to_idx]
                                effect[i][rs_i_max[2]][rs_i_max[0]] = fitness_gpu(s_idx, sn_idx, g_gpu, com_fs_idx, hop)

                    else:
                        if effect0 < rs_i_min[1] and curT > 0:
                            print("Add effect 0")
                            bestS = copy.deepcopy(bestS0)
                            for i in range(len(communityList)):
                                max_islandsEffect_i = []
                                for j in range(comRes[i]):
                                    maxE = max(effect[i][j])
                                    max_islandsEffect_i.append([effect[i][j].index(maxE), maxE, j])
                                rs_i_max = [copy.deepcopy(max_islandsEffect_i[0][0]),
                                            copy.deepcopy(max_islandsEffect_i[0][1]), 0]
                                for j in range(comRes[i]):
                                    if max_islandsEffect_i[j][1] > rs_i_max[1]:
                                        rs_i_max = [copy.deepcopy(max_islandsEffect_i[j][0]),
                                                    copy.deepcopy(max_islandsEffect_i[j][1]), j]
                                population[i][rs_i_max[2]][rs_i_max[0]] = copy.deepcopy(bestS0)
                                
                                # Re-eval
                                g_gpu = graphGPUs[i]
                                ind = population[i][rs_i_max[2]][rs_i_max[0]]
                                s_idx = [g_gpu.node_to_idx[u] for u in ind if u in g_gpu.node_to_idx]
                                sn_idx = [g_gpu.node_to_idx[u] for u in SN if u in g_gpu.node_to_idx]
                                com_fs_idx = [g_gpu.node_to_idx[u] for u in comAndFS[i] if u in g_gpu.node_to_idx]
                                effect[i][rs_i_max[2]][rs_i_max[0]] = fitness_gpu(s_idx, sn_idx, g_gpu, com_fs_idx, hop)


                    if len(communityList) == 1 and (curT > s_g) \
                            and ((s_t_g[curT - s_g] - s_t_g[curT]) <= (theta * s_g)) and ((curT - s_g) >= e_g_b):
                        print(s_t_g[curT - s_g], s_t_g[curT])
                        break

                    if int(curT) == int(maxT + beta * s_g):
                        break

                    # EVOLUTION LOOP
                    print("Start GPU serial evolution", curT, " generation")
                    
                    for i in range(len(communityList)):
                        for j in range(comRes[i]):
                            # Evolve
                            population[i][j], effect[i][j] = evolve_one_step(
                                population[i][j], effect[i][j], graphGPUs[i], comGs[i], SN, comAndFS[i], hop,
                                comAndSea[i], community_k[i], Ni, cOne, cTwo, P_score, gamaCom[i], N_prob
                            )
                            
                        comGenAcc[i] += 1
                        
                        # Subpopulation Communication (simplified: ring migration)
                        if comRes[i] > 1:
                             # Logic from original: circular shift best individual
                             # Simplified: find best in each subpop
                             minPos = []
                             for j in range(comRes[i]):
                                 minPos.append(effect[i][j].index(min(effect[i][j])))
                                 
                             # Shift
                             first_best_ind = copy.deepcopy(population[i][0][minPos[0]])
                             first_best_eff = effect[i][0][minPos[0]]
                             
                             for j in range(comRes[i] - 1):
                                 population[i][j][minPos[j]] = copy.deepcopy(population[i][j+1][minPos[j+1]])
                                 effect[i][j][minPos[j]] = effect[i][j+1][minPos[j+1]]
                                 
                             population[i][comRes[i]-1][minPos[comRes[i]-1]] = first_best_ind
                             effect[i][comRes[i]-1][minPos[comRes[i]-1]] = first_best_eff
                             
                    curT += 1
                    
                    # Merge Logic
                    if len(communityList) > 1:
                        isMerge = 0
                        merge = [-2 for _ in range(len(communityList))]

                        if curT == maxT:
                            merge = [-1 for i in range(len(communityList))]
                            isMerge = 1

                        else:
                            for i in range(len(communityList)):
                                minE_i = effect[i][0][0]
                                for j in range(comRes[i]):
                                    if min(effect[i][j]) < minE_i:
                                        minE_i = min(effect[i][j])
                                s_t_l[i, curT, len(communityList)] = (minE_i, comGenAcc[i])

                            for i in range(len(communityList)):
                                # print(s_t_l[i, comBen[i], len(communityList)], s_t_l[i, curT, len(communityList)])

                                deltaF = s_t_l[i, comBen[i], len(communityList)][0] - \
                                         s_t_l[i, curT, len(communityList)][0]
                                deltaT = s_t_l[i, curT, len(communityList)][1] - \
                                         s_t_l[i, comBen[i], len(communityList)][1]

                                if (deltaF <= (theta * deltaT * community_k[i] / k)) and (
                                        deltaT >= s_l):
                                    merge[i] = -1
                                    isMerge = 1
                                elif deltaF > (theta * deltaT * community_k[i] / k):
                                    comBen[i] = curT

                        if isMerge == 1:
                            print(f"Merging: {merge}")
                            population, effect, communityList, community_k, s_t_l, \
                            comGenAcc, comBen, comAndSea, comAndFS, comOrSN, comGs, gamaCom, comRes = \
                                mergeCommunity_12(merge, communityList, community_k, population, effect,
                                                  comRes, Ni, Gs, comGs, SN, fitnessSpace, hop, s_t_l, curT,
                                                  comGenAcc, comBen, P_score, gama, searchSpace)
                                                  
                            # Re-create GPU graphs for new communities
                            graphGPUs = []
                            for i in range(len(communityList)):
                                graphGPUs.append(GraphGPU(comGs[i], list(comGs[i].nodes)))

                        if len(communityList) == 1:
                            e_g_b = curT
                            print("The global evolution of the ", e_g_b, " generation begins!")

                end_time = time.time()
                run_time = end_time - start_time
                print(f'Running time: {run_time:.5f} s')

                bs_idx = [global_gpu_graph.node_to_idx[u] for u in bestS if u in global_gpu_graph.node_to_idx]
                bestE = fitness_gpu(bs_idx, global_sn_idx, global_gpu_graph, global_fs_idx, hop)

                print("bestS:", bestS)
                print("Optimal fitness value:", bestE)
                
                neg_activated_set = simulate_propagation(Gs, bestS, SN, hop)
                print(f"Negatively Activated Nodes: {len(neg_activated_set)}")

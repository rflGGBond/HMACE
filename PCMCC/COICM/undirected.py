import re
import os, sys
import time
import copy
import heapq
import random
import leidenalg
import igraph as ig
import networkx as nx
import multiprocessing
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from select_SN import select_SN

# log recorder
class Logger(object):

    def __init__(self, stream=sys.stdout):
        output_dir = "../../results/undirected" 
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        log_name = "facebook_log_20251228.txt"
        filename = os.path.join(output_dir, log_name)

        self.terminal = stream
        self.log = open(filename, 'a+')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def communityDivision_1(G_1, C_1):
    c_G = G_1.copy()
    c_g = copy.deepcopy(ig.Graph.TupleList(list(c_G.edges(data='weight')), directed=False, edge_attrs=['weight']))
    c_part = leidenalg.find_partition(c_g, leidenalg.ModularityVertexPartition, weights=c_g.es['weight'],
                                      n_iterations=-1)
    print("------------------- communityDivision start -----------------------")
    print(f"Modularity: {c_part.modularity}\n")
    # print(c_part)
    # print("\n")

    rs_part = []
    pattern1 = re.compile(r"(?<=])[^][]+(?=\n\[)")
    matches = pattern1.findall(str(c_part) + "\n[")
    for match1 in matches:
        pattern2 = r"\d+"
        numbers = [int(match2) for match2 in re.findall(pattern2, match1)]
        rs_part.append(numbers)

    # print(rs_part)

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

    # print(rs_part)
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


def outer_5(population_5, i_5, j_5, Ni_5):
    def done(res, *args, **kwargs):
        for I in range(Ni_5):
            population_5[i_5][j_5][I] = res.result()[i_5, j_5, I]

    return done

def calEffect_6(i_6, j_6, Ni_6, populationij_6, comGsi_6, SN_6, comAndFSi_6, hop_6):
    effect_6 = {}
    for I in range(Ni_6):
        effect_6[i_6, j_6, I] = fitness_C_7(populationij_6[I], comGsi_6, SN_6, comAndFSi_6, hop_6)
    return effect_6

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
    
#     # === 统计最终被负向激活的节点 ===
#     neg_activated_nodes = []
#     for (u, t) in apN_fc:
#         if t == hop_7 and apN_fc[u, hop_7] > 0:
#             neg_activated_nodes.append(u)
# #
#     num_neg_activated = len(neg_activated_nodes)
    
    # =================================

    return effect_fc

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

def outer_8(effect_8, i_8, j_8, Ni_8):
    def done(res, *args, **kwargs):
        for I in range(Ni_8):
            effect_8[i_8][j_8][I] = res.result()[i_8, j_8, I]

    return done

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

def evolution_11(i_11, j_11, maxCommunityEnd_11, max_i_11,
                 shared_islands_11, shared_islandsEffect_11, locks_11,
                 G_11, SN_11, comAndFSi_11, hop_11, N_prob_11,
                 com_SN_i_11, community_k_i_11, comRes_11, N_11, cOne_11, cTwo_11,
                 islands_index_11, islandsEffect_index_11, locks_index_11, P_score_11, lastP_11,
                 shareLocksIndex_11, begin_11, gamaComi_11, comGenAcc_11):
    if shareLocksIndex_11[i_11, j_11] == lastP_11:
        begin_11[0] = 1

    while int(round(begin_11[0])) == 0:
        pass

    print(i_11, j_11, " Process started!")
    g_11 = 0

    if i_11 == max_i_11:
        maxCommunityEnd_11[j_11] = 0

        if community_k_i_11 >= 1:
            indexS1 = shared_islandsEffect_11[
                      islandsEffect_index_11[i_11, j_11, 0]:islandsEffect_index_11[i_11, j_11, N_11 - 1] + 1].index(min(
                shared_islandsEffect_11[
                islandsEffect_index_11[i_11, j_11, 0]:islandsEffect_index_11[i_11, j_11, N_11 - 1] + 1]))

            for I in range(N_11):
                if I == indexS1:
                    continue
                S1 = copy.deepcopy(shared_islands_11 \
                                       [islands_index_11[i_11, j_11, indexS1, 0]:
                                        islands_index_11[i_11, j_11, indexS1, community_k_i_11 - 1] + 1])

                SI = copy.deepcopy(shared_islands_11 \
                                       [islands_index_11[i_11, j_11, I, 0]:
                                        islands_index_11[i_11, j_11, I, community_k_i_11 - 1] + 1])

                repeatS1 = 0
                repeatSI = 0

                for J in range(community_k_i_11):
                    if random.random() < cOne_11:
                        if random.random() < cTwo_11:  # two-way cross
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
                    replaceS1 = sample(list(set(com_SN_i_11) - set(S1)), P_score_11, repeatS1)
                    J = 0
                    for e in range(community_k_i_11):
                        if S1[e] == -1:
                            S1[e] = replaceS1[J]
                            J += 1

                if repeatSI != 0:
                    replaceSI = sample(list(set(com_SN_i_11) - set(SI)), P_score_11, repeatSI)
                    J = 0
                    for e in range(community_k_i_11):
                        if SI[e] == -1:
                            SI[e] = replaceSI[J]
                            J += 1

                effectS1 = fitness_C_7(S1, G_11, SN_11, comAndFSi_11, hop_11)
                effectSI = fitness_C_7(SI, G_11, SN_11, comAndFSi_11, hop_11)

                if effectS1 < shared_islandsEffect_11[islandsEffect_index_11[i_11, j_11, indexS1]]:
                    for X in range(community_k_i_11):
                        shared_islands_11[islands_index_11[i_11, j_11, indexS1, X]] = S1[X]
                    shared_islandsEffect_11[islandsEffect_index_11[i_11, j_11, indexS1]] = effectS1

                if effectSI < shared_islandsEffect_11[islandsEffect_index_11[i_11, j_11, I]]:
                    for X in range(community_k_i_11):
                        shared_islands_11[islands_index_11[i_11, j_11, I, X]] = SI[X]
                    shared_islandsEffect_11[islandsEffect_index_11[i_11, j_11, I]] = effectSI

            # local search
            while True:
                indexS1 = shared_islandsEffect_11[
                          islandsEffect_index_11[i_11, j_11, 0]:islandsEffect_index_11[i_11, j_11, N_11 - 1] + 1].index(
                    min(shared_islandsEffect_11[
                        islandsEffect_index_11[i_11, j_11, 0]:islandsEffect_index_11[i_11, j_11, N_11 - 1] + 1]))

                S1 = copy.deepcopy(shared_islands_11[islands_index_11[i_11, j_11, indexS1, 0]:
                                                     islands_index_11[i_11, j_11, indexS1, community_k_i_11 - 1] + 1])

                discount_P_score_diff = []
                replace_discount_P_score_diff = {}

                for I in range(community_k_i_11):
                    rs = 0
                    predecessors = defaultdict(lambda: [])
                    one_hop_neighbors = []
                    two_hop_neighbors = []

                    for v in G_11.neighbors(S1[I]):
                        one_hop_neighbors.append(v)
                        for w in G_11.neighbors(v):
                            two_hop_neighbors.append(w)
                            predecessors[w].append(v)

                    oneAndF = set(one_hop_neighbors).intersection(set(comAndFSi_11)) - set(S1)
                    two_hop_neighbors = set(two_hop_neighbors).intersection(set(comAndFSi_11)) - set(S1)

                    twoAndOne = two_hop_neighbors.intersection(oneAndF)
                    two_one = two_hop_neighbors - oneAndF

                    for t in range(1, hop_11 + 1):
                        rs += N_prob_11[S1[I], t]

                    for v in oneAndF:
                        for t in range(1, hop_11 + 1):
                            rs += G_11[S1[I]][v]['weight'] * N_prob_11[v, t]

                    for w in twoAndOne:
                        temp_p = 1
                        for v in set(predecessors[w]):
                            temp_p *= (1 - G_11[S1[I]][v]['weight'] * G_11[v][w]['weight'])
                        for t in range(2, hop_11 + 1):
                            rs += (1 - G_11[S1[I]][w]['weight']) * (1 - temp_p) * (1 - N_prob_11[w, 1]) * N_prob_11[
                                w, t]

                    for w in two_one:
                        temp_p = 1
                        for v in set(predecessors[w]):
                            temp_p *= (1 - G_11[S1[I]][v]['weight'] * G_11[v][w]['weight'])
                        for t in range(2, hop_11 + 1):
                            rs += (1 - temp_p) * (1 - N_prob_11[w, 1]) * N_prob_11[w, t]

                    temp = 1
                    rs1 = 0

                    for u in set(S1).intersection(set(G_11.neighbors(S1[I]))):
                        temp *= (1 - G_11[u][S1[I]]['weight'])

                    for t in range(1, hop_11 + 1):
                        rs1 += (1 - temp) * N_prob_11[S1[I], t]

                    for v in oneAndF:
                        for t in range(2, hop_11 + 1):
                            rs1 += (1 - temp) * G_11[S1[I]][v]['weight'] * N_prob_11[v, t]

                    discount_P_score_diff.append(rs - rs1)

                I = discount_P_score_diff.index(min(discount_P_score_diff))
                rn = S1[I]

                Sbest = copy.deepcopy(S1)
                for nn in (set(gamaComi_11) - set(Sbest)):
                    S1[I] = nn

                    rs = 0

                    predecessors = defaultdict(lambda: [])
                    one_hop_neighbors = []
                    two_hop_neighbors = []

                    for v in G_11.neighbors(S1[I]):
                        one_hop_neighbors.append(v)
                        for w in G_11.neighbors(v):
                            two_hop_neighbors.append(w)
                            predecessors[w].append(v)

                    oneAndF = set(one_hop_neighbors).intersection(set(comAndFSi_11)) - set(S1)
                    two_hop_neighbors = set(two_hop_neighbors).intersection(set(comAndFSi_11)) - set(S1)

                    twoAndOne = two_hop_neighbors.intersection(oneAndF)
                    two_one = two_hop_neighbors - oneAndF

                    for t in range(1, hop_11 + 1):
                        rs += N_prob_11[S1[I], t]

                    for v in oneAndF:
                        for t in range(1, hop_11 + 1):
                            rs += G_11[S1[I]][v]['weight'] * N_prob_11[v, t]

                    for w in twoAndOne:
                        temp_p = 1
                        for v in set(predecessors[w]):
                            temp_p *= (1 - G_11[S1[I]][v]['weight'] * G_11[v][w]['weight'])
                        for t in range(2, hop_11 + 1):
                            rs += (1 - G_11[S1[I]][w]['weight']) * (1 - temp_p) * (1 - N_prob_11[w, 1]) * N_prob_11[
                                w, t]

                    for w in two_one:
                        temp_p = 1
                        for v in set(predecessors[w]):
                            temp_p *= (1 - G_11[S1[I]][v]['weight'] * G_11[v][w]['weight'])
                        for t in range(2, hop_11 + 1):
                            rs += (1 - temp_p) * (1 - N_prob_11[w, 1]) * N_prob_11[w, t]

                    temp = 1
                    rs1 = 0

                    for u in set(S1).intersection(set(G_11.neighbors(S1[I]))):
                        temp *= (1 - G_11[u][S1[I]]['weight'])

                    for t in range(1, hop_11 + 1):
                        rs1 += (1 - temp) * N_prob_11[S1[I], t]

                    for v in oneAndF:
                        for t in range(2, hop_11 + 1):
                            rs1 += (1 - temp) * G_11[S1[I]][v]['weight'] * N_prob_11[v, t]

                    replace_discount_P_score_diff[nn] = rs - rs1

                S1[I] = -1
                rmax = discount_P_score_diff[I]

                for nn in list(set(gamaComi_11) - set(Sbest)):
                    if replace_discount_P_score_diff[nn] >= rmax:
                        rmax = replace_discount_P_score_diff[nn]
                        rn = nn

                S1[I] = rn

                effectS1 = fitness_C_7(S1, G_11, SN_11, comAndFSi_11, hop_11)

                if effectS1 < shared_islandsEffect_11[islandsEffect_index_11[i_11, j_11, indexS1]]:
                    for X in range(community_k_i_11):
                        shared_islands_11[islands_index_11[i_11, j_11, indexS1, X]] = S1[X]
                    shared_islandsEffect_11[islandsEffect_index_11[i_11, j_11, indexS1]] = effectS1
                else:
                    break

            if comRes_11[i_11] > 1:
                if j_11 == 0:
                    while int(
                            round(
                                sum(locks_11[
                                    locks_index_11[i_11, 1]: locks_index_11[i_11, comRes_11[i_11] - 1] + 1]))) \
                            != int(round((g_11 + 1) * (comRes_11[i_11] - 1))):
                        pass

                    # Subpopulation Communication
                    minPos = []
                    for J in range(comRes_11[i_11]):
                        minPos.append(shared_islandsEffect_11
                                      [islandsEffect_index_11[i_11, J, 0]:islandsEffect_index_11[
                                                                              i_11, J, N_11 - 1] + 1]
                                      .index(min(shared_islandsEffect_11
                                                 [islandsEffect_index_11[i_11, J, 0]:islandsEffect_index_11[
                                                                                         i_11, J, N_11 - 1] + 1])))

                    temp_shared_islands_11_i_0_min = copy.deepcopy(
                        shared_islands_11[islands_index_11[i_11, 0, minPos[0], 0]:
                                          islands_index_11[i_11, 0, minPos[0], community_k_i_11 - 1] + 1])

                    temp_shared_islandsEffect_11_i_0_min = shared_islandsEffect_11[
                        islandsEffect_index_11[i_11, 0, minPos[0]]]

                    for J in range(comRes_11[i_11] - 1):
                        for X in range(community_k_i_11):
                            shared_islands_11[islands_index_11[i_11, J, minPos[J], X]] = \
                                shared_islands_11[islands_index_11[i_11, J + 1, minPos[J + 1], X]]

                        shared_islandsEffect_11[islandsEffect_index_11[i_11, J, minPos[J]]] = \
                            shared_islandsEffect_11[islandsEffect_index_11[i_11, J + 1, minPos[J + 1]]]

                    for X in range(community_k_i_11):
                        shared_islands_11[islands_index_11[i_11, comRes_11[i_11] - 1, minPos[comRes_11[i_11] - 1], X]] = \
                            temp_shared_islands_11_i_0_min[X]

                    shared_islandsEffect_11[
                        islandsEffect_index_11[i_11, comRes_11[i_11] - 1, minPos[comRes_11[i_11] - 1]]] = \
                        temp_shared_islandsEffect_11_i_0_min

        g_11 += 1
        print(i_11, j_11, "Process Evolution ", g_11, " end!")

        maxCommunityEnd_11[j_11] = 1

        locks_11[locks_index_11[i_11, j_11]] = g_11

        if j_11 != 0:
            while int(round(locks_11[locks_index_11[i_11, 0]])) != int(round(g_11)):
                pass
        else:
            comGenAcc_11[i_11] += 1

    else:
        while int(round(sum(maxCommunityEnd_11))) != comRes_11[max_i_11]:

            if community_k_i_11 >= 1:

                indexS1 = shared_islandsEffect_11 \
                    [islandsEffect_index_11[i_11, j_11, 0]:islandsEffect_index_11[i_11, j_11, N_11 - 1] + 1]. \
                    index(min(shared_islandsEffect_11 \
                                  [
                              islandsEffect_index_11[i_11, j_11, 0]:islandsEffect_index_11[i_11, j_11, N_11 - 1] + 1]))

                for I in range(N_11):
                    if I == indexS1:
                        continue
                    S1 = copy.deepcopy(shared_islands_11 \
                                           [islands_index_11[i_11, j_11, indexS1, 0]:
                                            islands_index_11[i_11, j_11, indexS1, community_k_i_11 - 1] + 1])

                    SI = copy.deepcopy(shared_islands_11 \
                                           [islands_index_11[i_11, j_11, I, 0]:
                                            islands_index_11[i_11, j_11, I, community_k_i_11 - 1] + 1])

                    repeatS1 = 0
                    repeatSI = 0

                    for J in range(community_k_i_11):
                        if random.random() < cOne_11:
                            if random.random() < cTwo_11:  # two-way cross
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
                        replaceS1 = sample(list(set(com_SN_i_11) - set(S1)), P_score_11, repeatS1)
                        J = 0
                        for e in range(community_k_i_11):
                            if S1[e] == -1:
                                S1[e] = replaceS1[J]
                                J += 1

                    if repeatSI != 0:
                        replaceSI = sample(list(set(com_SN_i_11) - set(SI)), P_score_11, repeatSI)
                        J = 0
                        for e in range(community_k_i_11):
                            if SI[e] == -1:
                                SI[e] = replaceSI[J]
                                J += 1

                    effectS1 = fitness_C_7(S1, G_11, SN_11, comAndFSi_11, hop_11)
                    effectSI = fitness_C_7(SI, G_11, SN_11, comAndFSi_11, hop_11)

                    if effectS1 < shared_islandsEffect_11[islandsEffect_index_11[i_11, j_11, indexS1]]:
                        for X in range(community_k_i_11):
                            shared_islands_11[islands_index_11[i_11, j_11, indexS1, X]] = S1[X]
                        shared_islandsEffect_11[islandsEffect_index_11[i_11, j_11, indexS1]] = effectS1

                    if effectSI < shared_islandsEffect_11[islandsEffect_index_11[i_11, j_11, I]]:
                        for X in range(community_k_i_11):
                            shared_islands_11[islands_index_11[i_11, j_11, I, X]] = SI[X]
                        shared_islandsEffect_11[islandsEffect_index_11[i_11, j_11, I]] = effectSI

                # local search
                while True:
                    indexS1 = shared_islandsEffect_11[
                              islandsEffect_index_11[i_11, j_11, 0]:islandsEffect_index_11[
                                                                        i_11, j_11, N_11 - 1] + 1].index(
                        min(shared_islandsEffect_11[
                            islandsEffect_index_11[i_11, j_11, 0]:islandsEffect_index_11[i_11, j_11, N_11 - 1] + 1]))

                    S1 = copy.deepcopy(shared_islands_11[islands_index_11[i_11, j_11, indexS1, 0]:
                                                         islands_index_11[
                                                             i_11, j_11, indexS1, community_k_i_11 - 1] + 1])

                    discount_P_score_diff = []
                    replace_discount_P_score_diff = {}

                    for I in range(community_k_i_11):
                        rs = 0
                        predecessors = defaultdict(lambda: [])
                        one_hop_neighbors = []
                        two_hop_neighbors = []

                        for v in G_11.neighbors(S1[I]):
                            one_hop_neighbors.append(v)
                            for w in G_11.neighbors(v):
                                two_hop_neighbors.append(w)
                                predecessors[w].append(v)

                        oneAndF = set(one_hop_neighbors).intersection(set(comAndFSi_11)) - set(S1)
                        two_hop_neighbors = set(two_hop_neighbors).intersection(set(comAndFSi_11)) - set(S1)

                        twoAndOne = two_hop_neighbors.intersection(oneAndF)
                        two_one = two_hop_neighbors - oneAndF

                        for t in range(1, hop_11 + 1):
                            rs += N_prob_11[S1[I], t]

                        for v in oneAndF:
                            for t in range(1, hop_11 + 1):
                                rs += G_11[S1[I]][v]['weight'] * N_prob_11[v, t]

                        for w in twoAndOne:
                            temp_p = 1
                            for v in set(predecessors[w]):
                                temp_p *= (1 - G_11[S1[I]][v]['weight'] * G_11[v][w]['weight'])
                            for t in range(2, hop_11 + 1):
                                rs += (1 - G_11[S1[I]][w]['weight']) * (1 - temp_p) * (1 - N_prob_11[w, 1]) * N_prob_11[
                                    w, t]

                        for w in two_one:
                            temp_p = 1
                            for v in set(predecessors[w]):
                                temp_p *= (1 - G_11[S1[I]][v]['weight'] * G_11[v][w]['weight'])
                            for t in range(2, hop_11 + 1):
                                rs += (1 - temp_p) * (1 - N_prob_11[w, 1]) * N_prob_11[w, t]

                        temp = 1
                        rs1 = 0

                        for u in set(S1).intersection(set(G_11.neighbors(S1[I]))):
                            temp *= (1 - G_11[u][S1[I]]['weight'])


                        for t in range(1, hop_11 + 1):
                            rs1 += (1 - temp) * N_prob_11[S1[I], t]

                        for v in oneAndF:
                            for t in range(2, hop_11 + 1):
                                rs1 += (1 - temp) * G_11[S1[I]][v]['weight'] * N_prob_11[v, t]

                        discount_P_score_diff.append(rs - rs1)

                    I = discount_P_score_diff.index(min(discount_P_score_diff))
                    rn = S1[I]

                    Sbest = copy.deepcopy(S1)
                    for nn in (set(gamaComi_11) - set(Sbest)):
                        S1[I] = nn

                        rs = 0

                        predecessors = defaultdict(lambda: [])
                        one_hop_neighbors = []
                        two_hop_neighbors = []

                        for v in G_11.neighbors(S1[I]):
                            one_hop_neighbors.append(v)
                            for w in G_11.neighbors(v):
                                two_hop_neighbors.append(w)
                                predecessors[w].append(v)

                        oneAndF = set(one_hop_neighbors).intersection(set(comAndFSi_11)) - set(S1)
                        two_hop_neighbors = set(two_hop_neighbors).intersection(set(comAndFSi_11)) - set(S1)

                        twoAndOne = two_hop_neighbors.intersection(oneAndF)
                        two_one = two_hop_neighbors - oneAndF

                        for t in range(1, hop_11 + 1):
                            rs += N_prob_11[S1[I], t]

                        for v in oneAndF:
                            for t in range(1, hop_11 + 1):
                                rs += G_11[S1[I]][v]['weight'] * N_prob_11[v, t]

                        for w in twoAndOne:
                            temp_p = 1
                            for v in set(predecessors[w]):
                                temp_p *= (1 - G_11[S1[I]][v]['weight'] * G_11[v][w]['weight'])
                            for t in range(2, hop_11 + 1):
                                rs += (1 - G_11[S1[I]][w]['weight']) * (1 - temp_p) * (1 - N_prob_11[w, 1]) * N_prob_11[
                                    w, t]

                        for w in two_one:
                            temp_p = 1
                            for v in set(predecessors[w]):
                                temp_p *= (1 - G_11[S1[I]][v]['weight'] * G_11[v][w]['weight'])
                            for t in range(2, hop_11 + 1):
                                rs += (1 - temp_p) * (1 - N_prob_11[w, 1]) * N_prob_11[w, t]

                        temp = 1
                        rs1 = 0

                        for u in set(S1).intersection(set(G_11.neighbors(S1[I]))):
                            temp *= (1 - G_11[u][S1[I]]['weight'])

                        for t in range(1, hop_11 + 1):
                            rs1 += (1 - temp) * N_prob_11[S1[I], t]

                        for v in oneAndF:
                            for t in range(2, hop_11 + 1):
                                rs1 += (1 - temp) * G_11[S1[I]][v]['weight'] * N_prob_11[v, t]

                        replace_discount_P_score_diff[nn] = rs - rs1

                    S1[I] = -1
                    rmax = discount_P_score_diff[I]

                    for nn in list(set(gamaComi_11) - set(Sbest)):
                        if replace_discount_P_score_diff[nn] >= rmax:
                            rmax = replace_discount_P_score_diff[nn]
                            rn = nn

                    S1[I] = rn

                    effectS1 = fitness_C_7(S1, G_11, SN_11, comAndFSi_11, hop_11)

                    if effectS1 < shared_islandsEffect_11[islandsEffect_index_11[i_11, j_11, indexS1]]:
                        for X in range(community_k_i_11):
                            shared_islands_11[islands_index_11[i_11, j_11, indexS1, X]] = S1[X]
                        shared_islandsEffect_11[islandsEffect_index_11[i_11, j_11, indexS1]] = effectS1
                    else:
                        break

                if comRes_11[i_11] > 1:
                    if j_11 == 0:
                        while int(
                                round(
                                    sum(locks_11[
                                        locks_index_11[i_11, 1]: locks_index_11[i_11, comRes_11[i_11] - 1] + 1]))) \
                                != int(round((g_11 + 1) * (comRes_11[i_11] - 1))):
                            pass

                        # Subpopulation Communication
                        minPos = []
                        for J in range(comRes_11[i_11]):
                            minPos.append(shared_islandsEffect_11
                                          [islandsEffect_index_11[i_11, J, 0]:islandsEffect_index_11[
                                                                                  i_11, J, N_11 - 1] + 1]
                                          .index(min(shared_islandsEffect_11
                                                     [islandsEffect_index_11[i_11, J, 0]:islandsEffect_index_11[
                                                                                             i_11, J, N_11 - 1] + 1])))

                        temp_shared_islands_11_i_0_min = copy.deepcopy(
                            shared_islands_11[islands_index_11[i_11, 0, minPos[0], 0]:
                                              islands_index_11[i_11, 0, minPos[0], community_k_i_11 - 1] + 1])

                        temp_shared_islandsEffect_11_i_0_min = shared_islandsEffect_11[
                            islandsEffect_index_11[i_11, 0, minPos[0]]]

                        for J in range(comRes_11[i_11] - 1):
                            for X in range(community_k_i_11):
                                shared_islands_11[islands_index_11[i_11, J, minPos[J], X]] = \
                                    shared_islands_11[islands_index_11[i_11, J + 1, minPos[J + 1], X]]

                            shared_islandsEffect_11[islandsEffect_index_11[i_11, J, minPos[J]]] = \
                                shared_islandsEffect_11[islandsEffect_index_11[i_11, J + 1, minPos[J + 1]]]

                        for X in range(community_k_i_11):
                            shared_islands_11[
                                islands_index_11[i_11, comRes_11[i_11] - 1, minPos[comRes_11[i_11] - 1], X]] = \
                                temp_shared_islands_11_i_0_min[X]

                        shared_islandsEffect_11[
                            islandsEffect_index_11[i_11, comRes_11[i_11] - 1, minPos[comRes_11[i_11] - 1]]] = \
                            temp_shared_islandsEffect_11_i_0_min

                g_11 += 1
                print(i_11, j_11, "Process Evolution", g_11, " end!")

                locks_11[locks_index_11[i_11, j_11]] = g_11

                if j_11 != 0:
                    while int(round(locks_11[locks_index_11[i_11, 0]])) != int(round(g_11)):
                        pass
                else:
                    comGenAcc_11[i_11] += 1

            else:
                g_11 += 1
                print(i_11, j_11, "Process Evolution", g_11, " end!")

                locks_11[locks_index_11[i_11, j_11]] = g_11

                if j_11 != 0:
                    while int(round(locks_11[locks_index_11[i_11, 0]])) != int(round(g_11)):
                        pass
                else:
                    comGenAcc_11[i_11] += 1

                break

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
                        fitness_C_7(new_islands[i][J][N], comGs_12[i], SN_12,
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

# def count_negative_activated_nodes(G, SN, hop):
#     """
#     在计算结束后统计被负向激活的节点数量
#     逻辑：重新跑 DPADV（负向传播）并统计 apN(u, hop) > 0 的节点
#     """
#     # 1. 需要构建 fitness space（即 FS）
#     #    FS = Nout(SN, h) \ SN
#     from collections import deque
    
#     # ---- 计算 Nout(SN, h) ----
#     FS = set()
#     queue = deque()
#     for s in SN:
#         queue.append((s, 0))

#     visited = set(SN)

#     while queue:
#         v, d = queue.popleft()
#         if d == hop:
#             continue

#         for w in G.neighbors(v):
#             if w not in visited:
#                 visited.add(w)
#                 FS.add(w)
#                 queue.append((w, d + 1))

#     # FS 去掉 SN（保险起见）
#     FS = FS - set(SN)

#     # ---------------------
#     # 2. 重新计算负向传播概率（DPADV）
#     # ---------------------
#     N_prob = negativeProbability_2(G, SN, FS, hop, all_FP_2=[])
    
#     # ---------------------
#     # 3. 统计 apN(u, hop) > 0 的节点
#     # ---------------------
#     neg_activated = []
#     for u in FS:
#         # apN 需要累加 t=1..hop
#         total_an = 0
#         for t in range(1, hop + 1):
#             total_an += N_prob.get((u, t), 0)
        
#         if total_an > 0:
#             neg_activated.append(u)

#     return len(neg_activated), neg_activated


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

        for k in [20, 110, 200]:

            repeats = 2

            for r in range(repeats):
                print("\nPCMCC", file_name, k, r + 1)

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

                population = []
                for i in range(C):
                    population.append([])
                    for j in range(comRes[i]):
                        population[i].append([])
                        for I in range(Ni):
                            population[i][j].append([])

                cpus = sum(comRes)
                pool = ProcessPoolExecutor(cpus)
                for i in range(C):
                    for j in range(comRes[i]):
                        fur = pool.submit(populationInitialization_4, i, j, Ni, comAndSea[i], community_k[i])
                        fur.add_done_callback(outer_5(population, i, j, Ni))

                pool.shutdown(True)

                comAndFS = []
                comOrSN = []
                comGs = []
                gamaCom = []
                for i in range(C):
                    comAndFS.append(list(set(communityList[i]).intersection(set(fitnessSpace))))  # Calculate comAndFS in advance
                    comOrSN.append(list(set(communityList[i] + SN)))  # Calculate comOrSN in advance
                    tempsubGi = Gs.subgraph(comOrSN[i])
                    subGi = nx.Graph(tempsubGi.edges(data=True))
                    subGi.add_nodes_from(comOrSN[i])
                    comGs.append(subGi.copy())  # Calculate subGi in advance
                    gamaCom.append(
                        heapq.nlargest(min(int(round(gama * community_k[i])), len(comAndSea[i])), comAndSea[i],
                                       key=lambda x: P_score[x]))

                effect = []
                for i in range(C):
                    effect.append([])
                    for j in range(comRes[i]):
                        effect[i].append([])
                        for I in range(Ni):
                            effect[i][j].append(float('inf'))

                pool = ProcessPoolExecutor(cpus)

                for i in range(C):
                    for j in range(comRes[i]):
                        fur = pool.submit(calEffect_6, i, j, Ni, population[i][j], comGs[i], SN, comAndFS[i], hop)
                        fur.add_done_callback(outer_8(effect, i, j, Ni))

                pool.shutdown(True)

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
                        curEffect = fitness_C_7(bestS, Gs, SN, fitnessSpace, hop)
                        s_t_g[curT] = curEffect
                        print("The optimal fitness value of the ", curT, " generation population:", curEffect)

                    else:
                        s_t_g[curT] = rs_i_min[1]
                        print("The optimal fitness value of the ", curT, " generation population:", rs_i_min[1])

                    print("bestS:", bestS)
                    print("Number of communities:", len(communityList))
                    print("comRes:", comRes)

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
                    # print("bestS0:", bestS0)

                    if len(communityList) != 1:
                        if effect0 < curEffect and curT > 0:
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
                                population[i][rs_i_max[2]][rs_i_max[0]] = copy.deepcopy(
                                    list(set(bestS0).intersection(comAndSea[i])))
                                effect[i][rs_i_max[2]][rs_i_max[0]] = fitness_C_7(
                                    population[i][rs_i_max[2]][rs_i_max[0]], comGs[i], SN, comAndFS[i], hop)
                                print(effect[i][rs_i_max[2]][rs_i_max[0]])

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
                                effect[i][rs_i_max[2]][rs_i_max[0]] = fitness_C_7(
                                    population[i][rs_i_max[2]][rs_i_max[0]], comGs[i], SN, comAndFS[i], hop)
                                print(effect[i][rs_i_max[2]][rs_i_max[0]])

                    if len(communityList) == 1 and (curT > s_g) \
                            and ((s_t_g[curT - s_g] - s_t_g[curT]) <= (theta * s_g)) and ((curT - s_g) >= e_g_b):
                        print(s_t_g[curT - s_g], s_t_g[curT])
                        break

                    if int(curT) == int(maxT + beta * s_g):
                        break

                    with multiprocessing.Manager() as manager:
                        maxCommunityEnd = manager.list()

                        lengths = [len(sublist) for sublist in communityList]
                        max_i = lengths.index(max(lengths))
                        print("Maximum Community Number:", max_i)

                        for j in range(comRes[max_i]):
                            maxCommunityEnd.append(0)

                        if len(communityList) > 1:
                            merge = [-2 for i in range(len(communityList))]  # -2 is the initial value for merge, -1 represents to merge

                        islands_index, islandsEffect_index, locks_index = convert_Index_10(population)

                        shared_islands = manager.list()
                        shared_islandsEffect = manager.list()
                        locks = manager.list()

                        for i in range(len(population)):
                            for j in range(len(population[i])):
                                locks.append(0)
                                for N in range(len(population[i][j])):
                                    shared_islandsEffect.append(effect[i][j][N])
                                    for X in range(len(population[i][j][N])):
                                        shared_islands.append(population[i][j][N][X])

                        lastP = 0
                        shareLocksIndex = {}
                        for i in range(len(communityList)):
                            for j in range(comRes[i]):
                                shareLocksIndex[i, j] = lastP
                                shareLocksIndex[lastP] = (i, j)
                                lastP += 1

                        lastP -= 1

                        begin = manager.list()
                        begin.append(0)

                        comGenAcc = manager.list(comGenAcc)

                        print("Start multiple processes", curT, " generation")

                        pool = ProcessPoolExecutor(cpus)

                        for i in range(len(communityList)):
                            for j in range(comRes[i]):
                                pool.submit(evolution_11, i, j, maxCommunityEnd, max_i,
                                            shared_islands, shared_islandsEffect, locks,
                                            comGs[i], SN, comAndFS[i], hop, N_prob,
                                            comAndSea[i], community_k[i], comRes, Ni, cOne, cTwo,
                                            islands_index, islandsEffect_index, locks_index, P_score, lastP,
                                            shareLocksIndex, begin, gamaCom[i], comGenAcc)

                        pool.shutdown(True)

                        comGenAcc = list(comGenAcc)

                        curT += 1

                        update_islands = []
                        update_islandsEffect = []

                        for i in range(len(population)):
                            update_islands.append([])
                            update_islandsEffect.append([])

                            for j in range(len(population[i])):
                                update_islands[i].append([])
                                update_islandsEffect[i].append([])

                                for N in range(len(population[i][j])):
                                    update_islands[i][j].append([])
                                    update_islandsEffect[i][j].append(
                                        shared_islandsEffect[islandsEffect_index[i, j, N]])

                                    for X in range(len(population[i][j][N])):
                                        update_islands[i][j][N].append(shared_islands[islands_index[i, j, N, X]])

                        print("Generation " + str(curT) + " comGenAcc:", comGenAcc)
                        print("Generation " + str(curT) + " comBen:", comBen)

                        population = copy.deepcopy(update_islands)
                        effect = copy.deepcopy(update_islandsEffect)

                    if len(communityList) > 1:
                        isMerge = 0

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
                                print(s_t_l[i, comBen[i], len(communityList)], s_t_l[i, curT, len(communityList)])

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

                        print(merge)

                        if isMerge == 1:
                            population, effect, communityList, community_k, s_t_l, \
                            comGenAcc, comBen, comAndSea, comAndFS, comOrSN, comGs, gamaCom, comRes = \
                                mergeCommunity_12(merge, communityList, community_k, population, effect,
                                                  comRes, Ni, Gs, comGs, SN, fitnessSpace, hop, s_t_l, curT,
                                                  comGenAcc, comBen, P_score, gama, searchSpace)

                        if len(communityList) == 1:
                            e_g_b = curT
                            print("The global evolution of the ", e_g_b, " generation begins!")

                end_time = time.time()
                run_time = end_time - start_time
                print(f'Running time: {run_time:.5f} s')

                bestE = fitness_C_7(bestS, Gs, SN, fitnessSpace, hop)

                print("bestS:", bestS)
                print("Optimal fitness value:", bestE)
                

                # cnt, nodes = count_negative_activated_nodes(G, SN, hop)
                # print("negative activated nodes: ", num_neg_activated)
                neg_activated_set = simulate_propagation(Gs, bestS, SN, hop)
                print(f"Negatively Activated Nodes: {len(neg_activated_set)}")

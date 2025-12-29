import random
from collections import defaultdict
import networkx as nx

class DPADVEvaluator:
    """
    Evaluator class for calculating DPADV (Dynamic Propagation-Activation-Degree Value)
    and other related metrics.
    """

    @staticmethod
    def calculate_negative_probability(G, SN, fitness_space, hop, all_FP=None):
        """
        Calculates the negative activation probability for nodes in the fitness space.
        Refactored from negativeProbability_2.
        """
        if all_FP is None:
            all_FP = []

        rs_N_p = {}
        ZN_f = []
        ZN_f.append(SN)
        for h in range(1, hop + 1):
            ZN_f.append([])
        
        pN_f = defaultdict(lambda: 0)
        apN_f = defaultdict(lambda: 0)
        
        for v in SN:
            pN_f[v, 0] = 1
            for h in range(hop + 1):
                apN_f[v, h] = 1
        
        for h in range(hop):
            temppN_f = defaultdict(lambda: 1)
            for v in ZN_f[h]:
                W_f = list(G.neighbors(v))
                ZN_f[h + 1] += W_f
                for w in W_f:
                    temppN_f[w] *= (1 - pN_f[v, h] * G[v][w]['weight'])
            
            ZN_f[h + 1] = list(set(ZN_f[h + 1]))
            for v in ZN_f[h + 1]:
                pN_f[v, h + 1] = (1 - temppN_f[v]) * (1 - apN_f[v, h])
                for tau_f in range(h + 1, hop + 1):
                    apN_f[v, tau_f] = apN_f[v, h] + pN_f[v, h + 1]

        for u in fitness_space:
            for t in range(1, hop + 1):
                rs_N_p[u, t] = pN_f[u, t]

        for u in all_FP:
            for t in range(1, hop + 1):
                rs_N_p[u, t] = 0
                
        return rs_N_p

    @staticmethod
    def calculate_positive_score(G, fitness_space, search_space, hop, N_prob):
        """
        Calculates the positive score for nodes in the search space.
        Refactored from positiveScore_3.
        """
        rs_P_S = {}

        for u in search_space:
            predecessors = defaultdict(lambda: [])
            rs_P_S[u] = 0
            one_hop_neighbors = []
            two_hop_neighbors = []
            
            for v in G.neighbors(u):
                one_hop_neighbors.append(v)
                for w in G.neighbors(v):
                    two_hop_neighbors.append(w)
                    predecessors[w].append(v)

            oneAndF = set(one_hop_neighbors).intersection(set(fitness_space))
            two_hop_neighbors = set(two_hop_neighbors).intersection(set(fitness_space)) - set([u])

            twoAndOne = two_hop_neighbors.intersection(oneAndF)
            two_One = two_hop_neighbors - oneAndF

            for t in range(1, hop + 1):
                rs_P_S[u] += N_prob.get((u, t), 0)

            for v in oneAndF:
                for t in range(1, hop + 1):
                    rs_P_S[u] += G[u][v]['weight'] * N_prob.get((v, t), 0)

            for w in twoAndOne:
                temp_p = 1
                for v in set(predecessors[w]):
                    temp_p *= (1 - G[u][v]['weight'] * G[v][w]['weight'])
                for t in range(2, hop + 1):
                    rs_P_S[u] += (1 - G[u][w]['weight']) * (1 - temp_p) * (1 - N_prob.get((w, 1), 0)) * N_prob.get((w, t), 0)

            for w in two_One:
                temp_p = 1
                for v in set(predecessors[w]):
                    temp_p *= (1 - G[u][v]['weight'] * G[v][w]['weight'])
                for t in range(2, hop + 1):
                    rs_P_S[u] += (1 - temp_p) * (1 - N_prob.get((w, 1), 0)) * N_prob.get((w, t), 0)

        return rs_P_S

    @staticmethod
    def calculate_fitness(seed, G, SN, com_and_fs, hop):
        """
        Calculates the fitness (DPADV) of a given seed set.
        Refactored from fitness_C_7.
        """
        effect_fc = 0
        ZP_fc = []
        ZN_fc = []
        ZP_fc.append(seed)
        ZN_fc.append(SN)
        
        for h in range(1, hop + 1):
            ZP_fc.append([])
            ZN_fc.append([])
            
        pP_fc = defaultdict(lambda: 0)
        apP_fc = defaultdict(lambda: 0)
        pN_fc = defaultdict(lambda: 0)
        apN_fc = defaultdict(lambda: 0)
        
        for v in seed:
            pP_fc[v, 0] = 1
            for h in range(hop + 1):
                apP_fc[v, h] = 1
                
        for v in SN:
            pN_fc[v, 0] = 1
            for h in range(hop + 1):
                apN_fc[v, h] = 1
                
        for h in range(hop):
            temppP_fc = defaultdict(lambda: 1)
            temppN_fc = defaultdict(lambda: 1)
            
            for v in ZP_fc[h]:
                W_fc = list(G.neighbors(v))
                ZP_fc[h + 1] += W_fc
                for w in W_fc:
                    temppP_fc[w] *= (1 - pP_fc[v, h] * G[v][w]['weight'])
            
            ZP_fc[h + 1] = list(set(ZP_fc[h + 1]))
            for v in ZP_fc[h + 1]:
                pP_fc[v, h + 1] = (1 - temppP_fc[v]) * (1 - apN_fc[v, h]) * (1 - apP_fc[v, h])
                for tau_f in range(h + 1, hop + 1):
                    apP_fc[v, tau_f] = apP_fc[v, h] + pP_fc[v, h + 1]
                    
            for v in ZN_fc[h]:
                W_fc = list(G.neighbors(v))
                ZN_fc[h + 1] += W_fc
                for w in W_fc:
                    temppN_fc[w] *= (1 - pN_fc[v, h] * G[v][w]['weight'])
            
            ZN_fc[h + 1] = list(set(ZN_fc[h + 1]))
            for v in ZN_fc[h + 1]:
                pN_fc[v, h + 1] = temppP_fc[v] * (1 - temppN_fc[v]) * (1 - apN_fc[v, h]) * (1 - apP_fc[v, h])
                for tau_f in range(h + 1, hop + 1):
                    apN_fc[v, tau_f] = apN_fc[v, h] + pN_fc[v, h + 1]
                    
        for u in com_and_fs:
            effect_fc += apN_fc[u, hop]

        return effect_fc

    @staticmethod
    def simulate_propagation(G_sim, positive_seeds, negative_seeds, max_hop):
        """
        Simulates information propagation with given positive and negative seeds.
        Returns the set of nodes negatively activated.
        """
        pos_activated = set(positive_seeds)        
        neg_activated = set(negative_seeds)        
        current_pos_frontier = set(positive_seeds) 
        current_neg_frontier = set(negative_seeds) 
        
        for h in range(max_hop):
            new_pos_frontier = set()
            new_neg_frontier = set()
            
            # Positive propagation first
            for u in current_pos_frontier:
                for w in G_sim.neighbors(u):
                    if w not in pos_activated and w not in neg_activated:
                        prob = G_sim[u][w]['weight']
                        if random.random() < prob:
                            pos_activated.add(w)
                            new_pos_frontier.add(w)
                            
            # Negative propagation second
            for u in current_neg_frontier:
                for w in G_sim.neighbors(u):
                    if w not in pos_activated and w not in neg_activated and w not in new_pos_frontier:
                        prob = G_sim[u][w]['weight']
                        if random.random() < prob:
                            neg_activated.add(w)
                            new_neg_frontier.add(w)
                            
            current_pos_frontier = new_pos_frontier
            current_neg_frontier = new_neg_frontier
            
            if not current_pos_frontier and not current_neg_frontier:
                break
                
        return neg_activated

    @staticmethod
    def get_activated_node_count(seed_set, G, SN, max_hop=2, simulations=3):
        """
        Calculates the average number of negatively activated nodes by running multiple simulations.
        """
        total_activated = 0
        for _ in range(simulations):
            neg_set = DPADVEvaluator.simulate_propagation(G, seed_set, SN, max_hop)
            total_activated += len(neg_set)
        return total_activated / simulations

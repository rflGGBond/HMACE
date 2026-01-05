import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import heapq
import networkx as nx
import copy
from typing import List, Dict, Any
from collections import defaultdict

from ..core import graph_ops, evaluator, evolution, merger
from .community import Community
from ..utils.types import CommunityAction, MetaAction, MetaObservation, CommunitySummary
import random

class PCMCCEnvironment:
    def __init__(self, graph_path: str, sn_nodes: List[int], total_budget: int, num_communities: int, is_directed: bool = False):
        self.graph_path = graph_path
        self.sn_nodes = sn_nodes
        self.total_budget = total_budget
        self.initial_num_communities = num_communities
        self.is_directed = is_directed
        self.hop = 2 # Default hop count
        self.initial_alpha = 12.0 # Default search space reduction factor
        
        # Load Graph
        self.G = nx.DiGraph() if is_directed else nx.Graph()
        with open(graph_path, 'r') as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 3:
                    n, m, w = int(parts[0]), int(parts[1]), float(parts[2])
                    self.G.add_edge(n, m, weight=w)
        
        # Initialize Spaces
        self._init_spaces()
        
        # Initialize Communities
        self.communities: Dict[int, Community] = {}
        self._init_communities()
        
        # Global State
        self.current_gen = 0
        self.global_dpadv_history = []
        self.parameter_history: List[Dict[str, Any]] = [] # [{'params': {...}, 'global_score': float}]
        self.global_best_seed = []
        self.global_best_dpadv = float('inf')
        
        # Parallel Execution Context
        self.manager = multiprocessing.Manager()
        # Legacy lists - commented out to prevent index errors with dynamic community merging
        # self.shared_islands = self.manager.list()
        # self.shared_islands_effect = self.manager.list()
        # self.locks = self.manager.list()
        # self.begin_flag = self.manager.list([0])
        # self.max_community_end_flags = self.manager.list()
        # self.com_gen_acc = self.manager.list([0] * num_communities)
        
        # Merge suggestions pending execution
        self.pending_merge_suggestions: List[tuple] = []

    def set_merge_suggestions(self, suggestions: List[tuple]):
        """
        Stores merge suggestions from Meta-Agent to be executed in the next step.
        """
        self.pending_merge_suggestions = suggestions

    def _convert_index(self, islands):
        """
        Helper to flatten population indices for shared memory.
        Refactored from convert_Index_10.
        """
        res_1 = {} # [i, j, N, X] -> flat_idx
        res_2 = {} # [i, j, N] -> flat_idx
        res_3 = {} # [i, j] -> flat_idx
        count_1 = 0
        count_2 = 0
        count_3 = 0

        # islands structure: [community][subpop][individual][gene]
        # In our case, self.population is [community][subpop][individual] (list of nodes)
        
        for i in range(len(islands)):
            for j in range(len(islands[i])):
                res_3[i, j] = count_3
                count_3 += 1

                for N in range(len(islands[i][j])):
                    res_2[i, j, N] = count_2
                    count_2 += 1

                    for X in range(len(islands[i][j][N])):
                        res_1[i, j, N, X] = count_1
                        count_1 += 1
        
        return res_1, res_2, res_3

    def _init_spaces(self):
        # 1. Fitness Space
        self.fitness_space = []
        cur_un_ergodic = copy.deepcopy(self.sn_nodes)
        hop_f = 0
        hop = 2 # Hardcoded for now, should be param
        
        while hop_f <= hop:
            cur_ergodic = copy.deepcopy(cur_un_ergodic)
            self.fitness_space += cur_ergodic
            if hop_f == hop:
                break
            else:
                cur_un_ergodic = []
                for u in cur_ergodic:
                    for v in self.G.neighbors(u):
                        cur_un_ergodic.append(v)
                cur_un_ergodic = list(set(cur_un_ergodic) - set(self.fitness_space))
                hop_f += 1
        self.fitness_space = list(set(self.fitness_space) - set(self.sn_nodes))

        # 2. Search Space
        self.search_space = []
        cur_un_ergodic = copy.deepcopy(self.fitness_space)
        hop_s = 0
        while hop_s <= hop:
            cur_ergodic = copy.deepcopy(cur_un_ergodic)
            self.search_space += cur_ergodic
            if hop_s == hop:
                break
            else:
                cur_un_ergodic = []
                for u in cur_ergodic:
                    for v in self.G.neighbors(u):
                        cur_un_ergodic.append(v)
                cur_un_ergodic = list(set(cur_un_ergodic) - set(self.search_space + self.sn_nodes))
                hop_s += 1
                
        # 3. Graph Subgraph (Gs)
        all_nodes = list(set(self.search_space + self.sn_nodes))
        self.Gs = self.G.subgraph(all_nodes).copy()
        
        # 4. Search Space Reduction
        # Use initial_alpha parameter instead of hardcoded value
        # Use Degree Centrality (In+Out) as heuristic
        self.search_space = heapq.nlargest(
            min(len(self.search_space), int(self.initial_alpha * self.total_budget)), 
            self.search_space, 
            key=lambda x: self.Gs.degree(x)
        )

    def _init_communities(self):
        # Community Division
        parts = graph_ops.detect_communities(self.Gs, self.initial_num_communities)
        
        # Calculate N_prob (needed for budget assignment)
        N_prob = evaluator.DPADVEvaluator.calculate_negative_probability(
            self.Gs, self.sn_nodes, self.fitness_space, hop=2
        )
        
        # Calculate N_prob sums for each community to assign budgets
        community_probs = []
        for part in parts:
            prob_sum = 0
            for node in part:
                # Sum N_prob for t=1 to hop (here hop=2)
                for t in range(1, 3):
                    prob_sum += N_prob.get((node, t), 0)
            community_probs.append(prob_sum)
            
        total_prob = sum(community_probs)
        
        # Assign Budgets based on N_prob weight
        budgets = []
        if total_prob > 0:
            remaining_budget = self.total_budget
            for i in range(len(parts) - 1):
                # Proportional assignment
                b = int(round(self.total_budget * (community_probs[i] / total_prob)))
                b = max(1, b) # Ensure at least 1 seed
                budgets.append(b)
                remaining_budget -= b
            budgets.append(max(1, remaining_budget)) # Assign rest to last
        else:
            # Fallback to even distribution
            base = self.total_budget // len(parts)
            budgets = [base] * len(parts)
            budgets[-1] += self.total_budget % len(parts)

        for i, part in enumerate(parts):
            self.communities[i] = Community(i, part, budgets[i])
            
            # Initialize random seed set for the community
            # Ensure we have enough candidates
            candidates = list(set(self.search_space).intersection(set(part)))
            if len(candidates) < budgets[i]:
                # Fallback: take from general search space if community is too small
                candidates = self.search_space 
            
            initial_seed = random.sample(candidates, k=min(budgets[i], len(candidates)))
            
            # Calculate initial DPADV
            score = evaluator.DPADVEvaluator.calculate_fitness(
                initial_seed, self.Gs, self.sn_nodes, self.fitness_space, hop=2
            )
            self.communities[i].update_best_solution(initial_seed, score)

    def _execute_merges(self):
        """
        Executes the pending merge suggestions by merging Community objects.
        This is a simplified implementation that replaces the complex legacy merger.py
        logic which relied on specific population structures not fully present here.
        """
        if not self.pending_merge_suggestions:
            return

        print(f"Executing Meta-Agent Merges: {self.pending_merge_suggestions}")
        
        # Track which communities are being merged to avoid double-merging
        merged_ids = set()
        new_communities = {}
        
        # Generate new ID starting after the highest current ID
        next_id = max(self.communities.keys()) + 1
        
        for merge_group in self.pending_merge_suggestions:
            # Filter out invalid IDs or already merged ones
            valid_group = [
                cid for cid in merge_group 
                if cid in self.communities and cid not in merged_ids
            ]
            
            if len(valid_group) < 2:
                continue
                
            # Mark as merged
            for cid in valid_group:
                merged_ids.add(cid)
            
            # Create New Community
            # 1. Merge Nodes
            new_nodes = set()
            new_budget = 0
            combined_seed = []
            
            # Helper to average parameters
            params_sum = {'cr1': 0, 'cr2': 0, 'beta': 0, 'alpha': 0}
            
            for cid in valid_group:
                com = self.communities[cid]
                new_nodes.update(com.state.nodes)
                new_budget += com.state.budget
                combined_seed.extend(com.state.current_seed_set)
                
                params_sum['cr1'] += com.state.cr1
                params_sum['cr2'] += com.state.cr2
                params_sum['beta'] += com.state.beta
                params_sum['alpha'] += com.state.alpha
            
            # 2. Create Object
            new_budget = int(new_budget) # Ensure integer
            new_com = Community(next_id, list(new_nodes), new_budget)
            
            # 3. Set Parameters (Average)
            count = len(valid_group)
            new_com.state.cr1 = params_sum['cr1'] / count
            new_com.state.cr2 = params_sum['cr2'] / count
            new_com.state.beta = params_sum['beta'] / count
            new_com.state.alpha = params_sum['alpha'] / count
            
            # 4. Set Seed (Truncate to budget if needed)
            # Ensure unique nodes in seed
            unique_seed = list(set(combined_seed))
            if len(unique_seed) > new_budget:
                # Keep the best ones? For now, random or just first N
                unique_seed = unique_seed[:new_budget]
            elif len(unique_seed) < new_budget:
                # Fill with random from new_nodes
                candidates = list(new_nodes - set(unique_seed))
                needed = new_budget - len(unique_seed)
                if candidates:
                    unique_seed.extend(random.sample(candidates, min(len(candidates), needed)))
            
            # Recalculate fitness for the new seed
            score = evaluator.DPADVEvaluator.calculate_fitness(
                unique_seed, self.Gs, self.sn_nodes, self.fitness_space, hop=2
            )
            new_com.update_best_solution(unique_seed, score)
            
            new_communities[next_id] = new_com
            next_id += 1
            
        # Apply changes to self.communities
        # Remove old
        for cid in merged_ids:
            del self.communities[cid]
            
        # Add new
        self.communities.update(new_communities)
        
        # Clear pending
        self.pending_merge_suggestions = []
        
        if new_communities:
            print(f"Merge Complete. Created {len(new_communities)} new communities. Total communities: {len(self.communities)}")

    def step(self):
        """
        Executes one generation of evolution.
        """
        self.current_gen += 1
        
        # 0. Check and Execute Merges (if any pending from Meta-Agent)
        if self.pending_merge_suggestions:
            self._execute_merges()

        # 1. Parallel Evolution (Simulated Single-Threaded with Real Logic)
        print(f"Env: Executing step {self.current_gen}...")
        
        # Pre-calculate shared metrics needed for evolution
        # (In real MP, these are passed via shared memory)
        N_prob = evaluator.DPADVEvaluator.calculate_negative_probability(
            self.Gs, self.sn_nodes, self.fitness_space, hop=2
        )
        
        # Calculate P_score (Positive Score) for sampling
        P_score = evaluator.DPADVEvaluator.calculate_positive_score(
            self.Gs, self.fitness_space, self.search_space, 2, N_prob
        )
        
        # Sort P_score to get top nodes globally (descending order)
        sorted_p_score = sorted(P_score.items(), key=lambda x: x[1], reverse=True)
        
        for com_id, com in self.communities.items():
            # Update community with top scoring nodes
            com.set_top_k_nodes(sorted_p_score)
            
            current_seed = com.state.current_seed_set
            if not current_seed: continue
            
            budget = com.state.budget
            # Construct a small population for evolution (if not maintaining persistent pop)
            # For this step, we treat 'current_seed' as the best individual S1
            # and generate a temporary 'SI' to cross with.
            
            # 1. Generate SI (Mutation of S1)
            SI = copy.deepcopy(current_seed)
            # Simple mutation: replace 1 random node
            if self.search_space:
                 candidates = list(set(self.search_space) - set(SI))
                 if candidates:
                     idx = random.randint(0, len(SI)-1)
                     SI[idx] = random.choice(candidates)
            
            # 2. Crossover (S1 + SI) -> Offspring
            # Use the pure function from core.evolution
            
            S1_input = copy.deepcopy(current_seed)
            
            # Use community-specific parameters (alpha, beta, cr1, cr2)
            S1 = evolution.crossover_and_mutate(
                S1_input, SI, com.state.budget, 
                com.state.cr1, com.state.cr2, 
                self.search_space, P_score,
                alpha=com.state.alpha
            )
            
            # 3. Local Search
            # Use full local search logic matching PCMCC
            com_and_fs = set(com.state.nodes).union(set(self.fitness_space))
            gama_com = self.search_space
            
            S1_new = evolution.local_search(
                S1, self.G, com_and_fs, self.hop, N_prob, gama_com
            )
            
            # If changed, update and re-evaluate
            if S1_new != S1:
                S1 = S1_new
                effectS1 = evaluator.DPADVEvaluator.calculate_fitness(
                    S1, self.G, self.sn_nodes, com_and_fs, self.hop
                )
            
            # Check for improvement against Community's current best
            # Note: We are comparing 'effectS1' (local fitness approximation) vs 'current_dpadv' (global or previous best)
            # In a distributed setting, this comparison should ideally be consistent. 
            # Assuming 'calculate_fitness' returns consistent DPADV metric.
            
            # Recalculate fitness for S1 just to be sure we have the metric
            new_fitness = evaluator.DPADVEvaluator.calculate_fitness(
                S1, self.G, self.sn_nodes, com_and_fs, self.hop
            )

            if new_fitness < com.state.current_dpadv:
                com.update_best_solution(S1, new_fitness)
                com.reset_stagnation()
            else:
                com.increment_stagnation()

            # Update metrics for observation
            com.calculate_metrics([S1, SI], self.Gs) # Use current pop sample

        # 2. Update Global State
        # Find global best
        best_com_id = -1
        best_dpadv = float('inf')
        
        for com_id, com in self.communities.items():
            if com.state.current_dpadv < best_dpadv:
                best_dpadv = com.state.current_dpadv
                best_com_id = com_id
        
        if best_com_id != -1:
             # In a real scenario, global best is combination of all community seeds.
             # Here we simplify.
             self.global_best_dpadv = best_dpadv
             self.global_dpadv_history.append(best_dpadv)

    def get_global_observation(self):
        """
        Aggregates state to form MetaObservation.
        """
        # --- Calculate Community Closeness ---
        # Call logic from merger.py as requested
        # 1. Prepare list of community nodes
        com_nodes_map = {cid: com.state.nodes for cid, com in self.communities.items()}
        com_ids = list(self.communities.keys())
        
        # 2. Calculate Pairwise Scores using merger.calculate_connection_strength
        closeness_map = defaultdict(lambda: defaultdict(float))
        
        # Optimize: Iterate only pairs that likely have edges? 
        # Or iterating edges is faster than iterating all pairs O(N^2)
        # Using the edge iteration method is still faster for sparse graphs, 
        # but we want to invoke the specific function.
        
        # Approach A: Iterate edges and sum (Fast, logic inside merger.py function would be bypassed if we do it here)
        # Approach B: Call merger.calculate_connection_strength(G, nodes_i, nodes_j) for all pairs.
        # Since we want to "call merger.py", let's try to be smart. 
        # We can detect neighbors first, then call the function for precise scoring.
        
        # Step 1: Fast neighbor detection
        node_to_com = {}
        for cid, nodes in com_nodes_map.items():
            for n in nodes:
                node_to_com[n] = cid
                
        neighbors = defaultdict(set)
        for u, v in self.Gs.edges():
            c1 = node_to_com.get(u)
            c2 = node_to_com.get(v)
            if c1 is not None and c2 is not None and c1 != c2:
                neighbors[c1].add(c2)
                neighbors[c2].add(c1)
        
        # Step 2: Call merger.py for identified neighbors
        for c1, neighbor_set in neighbors.items():
            for c2 in neighbor_set:
                if c1 < c2: # Avoid double calc
                    # CALL THE MERGER FUNCTION
                    score = merger.calculate_connection_strength(self.Gs, com_nodes_map[c1], com_nodes_map[c2])
                    closeness_map[c1][c2] = score
                    closeness_map[c2][c1] = score
        
        summaries = []
        for com_id, com in self.communities.items():
            # Get calculated closeness
            closeness = dict(closeness_map[com_id])
            
            # Update community's neighbor list while we are at it
            com.update_neighbors(list(closeness.keys()))
            
            # Calculate Improvement Rate (Delta DPADV / Delta T)
            # Use history: dpadv_history
            imp_rate = 0.0
            history = com.state.dpadv_history
            if len(history) >= 2:
                # Calculate drop from oldest recorded to current
                # history[-1] is current (usually), history[0] is oldest in window
                delta_f = history[0] - history[-1]
                delta_t = len(history) - 1 # Generations elapsed in window
                if delta_t > 0:
                    imp_rate = delta_f / delta_t
            
            summaries.append(CommunitySummary(
                community_id=com_id,
                budget=com.state.budget,
                best_dpadv=com.state.current_dpadv,
                improvement_rate=imp_rate, 
                diversity=com.state.diversity_score,
                boundary_risk=len(com.state.boundary_nodes) / max(1, len(com.state.nodes)),
                closeness_info=closeness
            ))
            
        return MetaObservation(
            current_generation=self.current_gen,
            current_global_dpadv=self.global_best_dpadv,
            global_dpadv_history=self.global_dpadv_history,
            community_summaries=summaries,
            merge_history=[], # TODO: maintain merge history
            parameter_history=self.parameter_history
        )

    def apply_community_action(self, community_id: int, action: CommunityAction):
        """
        Applies the action from a Community Agent, including strict candidate evaluation.
        """
        com = self.communities.get(community_id)
        if not com: return
        
        # 1. Update Parameters (Mode A)
        if action.parameters:
            com.update_parameters(action.parameters)
        
        # 2. Candidate Generation (Mode B) - Try-Evaluate-Revert Logic
        if action.candidate_seed_set:
            # Validate constraints (e.g., size match)
            if len(action.candidate_seed_set) != com.state.budget:
                print(f"Agent {community_id} provided seed set of wrong size. Ignored.")
                return

            # Validate node existence
            for node in action.candidate_seed_set:
                if not self.Gs.has_node(node):
                    print(f"Agent {community_id} proposed non-existent node {node}. Ignored.")
                    return

            # Construct Global Candidate: Combine new candidate with OTHER communities' current bests
            global_candidate_seed = []
            for cid, other_com in self.communities.items():
                if cid == community_id:
                    global_candidate_seed.extend(action.candidate_seed_set)
                else:
                    global_candidate_seed.extend(other_com.state.current_seed_set)
            
            # Calculate Global DPADV for this candidate configuration
            new_global_dpadv = evaluator.DPADVEvaluator.calculate_fitness(
                global_candidate_seed, self.Gs, self.sn_nodes, self.fitness_space, hop=2
            )
            
            # Compare with current global best
            if new_global_dpadv < self.global_best_dpadv:
                print(f"Agent {community_id} found BETTER global solution (DPADV: {self.global_best_dpadv} -> {new_global_dpadv}). Accepted.")
                # Accept: Update Community Best & Global Best
                com.update_best_solution(action.candidate_seed_set, new_global_dpadv) # Note: Local DPADV is approximation here
                self.global_best_dpadv = new_global_dpadv
                self.global_best_seed = global_candidate_seed
            else:
                # Reject: Do nothing (Revert is implicit by not applying)
                print(f"Agent {community_id} candidate rejected (DPADV: {new_global_dpadv} >= {self.global_best_dpadv}).")
                pass

    def apply_meta_action(self, action: MetaAction):
        """
        Applies global decisions from Meta-Agent.
        """
        # 1. Update Global Baselines
        if action.global_baselines:
            print(f"Meta-Agent updating global baselines: {action.global_baselines}")
            for com in self.communities.values():
                com.update_parameters(action.global_baselines, is_global_baseline=True)
            
            # Track Parameter History
            # Store the NEW parameters and the CURRENT global score (as a baseline for their performance)
            # Note: The true evaluation of these parameters happens in the NEXT generation.
            # However, for simplicity in "evolutionary prompt", we pair them with the score at the end of this gen.
            self.parameter_history.append({
                'params': action.global_baselines,
                'global_score': self.global_best_dpadv # Using current best as proxy
            })
            
            # Sort by score (ascending - lower is better)
            self.parameter_history.sort(key=lambda x: x['global_score'])
            # Keep top 10
            if len(self.parameter_history) > 10:
                self.parameter_history = self.parameter_history[:10]
                
        # 2. Update Budgets (Redistribution)
        if action.budget_adjustments:
            print(f"Meta-Agent adjusting budgets: {action.budget_adjustments}")
            for com_id, new_budget in action.budget_adjustments.items():
                if com_id in self.communities:
                    self.communities[com_id].state.budget = int(new_budget) # Ensure integer
                    # Note: Changing budget might require resizing population or seeds in next step
                    # This is a complex operation in real implementation (re-sampling or truncating)
                    
        # 3. Execute Merges (Already handled via set_merge_suggestions, but ensuring consistency)
        if action.merge_suggestions:
            # If not already set by run.py logic, set it here
            if not self.pending_merge_suggestions:
                self.set_merge_suggestions(action.merge_suggestions)

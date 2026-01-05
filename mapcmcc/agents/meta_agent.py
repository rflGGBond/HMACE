from .base import BaseAgent
from ..utils.types import MetaObservation, MetaAction
from ..utils.llm_client import LLMClient
import json
import dataclasses

class MetaAgent(BaseAgent):
    """
    Agent controlling the global parameters and merges.
    """
    def __init__(self, llm_client: LLMClient = None):
        self.llm_client = llm_client or LLMClient()  # Default to mock if not provided

    def get_action(self, observation: MetaObservation) -> MetaAction:
        # 1. Prepare Prompt
        obs_dict = dataclasses.asdict(observation)
        
        # --- Pre-processing: Identify Struggling Communities (Theta Logic) ---
        # Theta Threshold Logic: Rate < theta * (ki / k)
        # We'll calculate this and inject it into the prompt to guide the LLM.
        theta = 1.0 # Global parameter, could be made adjustable
        total_budget = sum(s.budget for s in observation.community_summaries)
        struggling_communities = []
        
        if total_budget > 0:
            for s in observation.community_summaries:
                target_rate = theta * (s.budget / total_budget)
                if s.improvement_rate <= target_rate:
                    struggling_communities.append(s.community_id)
        
        struggling_info_str = f"Communities needing merge (Low Improvement Rate <= Target): {struggling_communities}"
        print(f"Meta-Agent Analysis: {struggling_info_str}")

        # Truncate history to prevent prompt overflow
        if "global_dpadv_history" in obs_dict and isinstance(obs_dict["global_dpadv_history"], list):
            obs_dict["global_dpadv_history"] = obs_dict["global_dpadv_history"][-10:] # Keep only last 10
            
        if "merge_history" in obs_dict and isinstance(obs_dict["merge_history"], list):
             obs_dict["merge_history"] = obs_dict["merge_history"][-5:] # Keep only last 5
        
        # Format Parameter History for prompt
        history_str = ""
        if "parameter_history" in obs_dict and obs_dict["parameter_history"]:
             for entry in obs_dict["parameter_history"]:
                 history_str += f"{{ {entry['params']} }} {{ {entry['global_score']} }}\n"
        else:
             # Default Fallback
             history_str = "{ 'cr1': 0.4, 'cr2': 0.4, 'beta': 2.0, 'alpha': 12.0 } { N/A }\n"

        system_prompt = f"""
        Description of problem and solution properties
        You are the Meta Agent. Your task is to find a set of global parameters ('cr1', 'cr2', 'beta', 'alpha') that results in the lowest possible Global DPADV score (Negative Influence Blocking).
        
        PARAMETER DEFINITIONS:
        - cr1 (0.0-1.0): Crossover Rate 1. Probability of performing crossover. Higher values mean more gene exchange.
        - cr2 (0.0-1.0): Crossover Rate 2. Probability of two-way crossover vs one-way.
        - beta (1.0-10.0): Local Search Intensity. Higher values imply more aggressive local optimization.
        - alpha (1.0-20.0): Search Space Reduction Factor. Determines the pool size of candidate nodes (alpha * budget). Higher values allow wider exploration but slower convergence.

        CRITICAL INSIGHT:
        {struggling_info_str}
        
        In-context examples (population)
        Below are some previous parameter sets and their resulting Global DPADV scores. The sets are arranged in descending order based on their scores, where lower values are better.

        {history_str}

        Task instructions
        Please follow the instruction step-by-step to generate new global parameters and coordinate communities:
        
        1. Select two parameter sets from the above history.
        2. Crossover the two sets to generate new parameters.
        3. Mutate the parameters (explore slightly different values).
        4. Keep the generated parameters as the new 'global_baselines'.
        
        ADDITIONAL TASKS (Parallel to Evolution):
        5. Suggest Merges: If communities are struggling (listed above) or have high connection overlap, add them to 'merge_suggestions'.
        6. Redistribute Budget: If needed, move budget to struggling communities in 'budget_adjustments'.

        OUTPUT RULES:
        1. Return ONLY valid JSON.
        2. Format:
        {{
            "reasoning": "...",
            "global_baselines": {{ "cr1": float, "cr2": float, "beta": float, "alpha": float }},
            "budget_adjustments": {{ "id": delta }},
            "merge_suggestions": [[id1, id2]]
        }}
        """
        
        user_prompt = f"Current Global Observation: {json.dumps(obs_dict, default=str)}\n\nRespond with valid JSON only."
        
        # 2. Call LLM
        try:
            response_str = self.llm_client.get_completion(system_prompt, user_prompt, temperature=0.75)
            print(f"Meta-Agent Response: {response_str}")  # 输出Meta Agent的原始响应
            response_json = json.loads(response_str)
            
            # 3. Parse Response to Action
            # Ensure keys exist
            budget_adjustments = response_json.get("budget_adjustments", {})
            # Convert string keys back to int if necessary (JSON keys are always strings)
            if budget_adjustments:
                budget_adjustments = {int(k): v for k, v in budget_adjustments.items()}
            
            # --- Hybrid Merge Strategy: Filter by Heuristic Score (from merger.py logic) ---
            raw_suggestions = response_json.get("merge_suggestions", [])
            valid_suggestions = []
            
            # Build a lookup for community summaries to access closeness info
            com_lookup = {s.community_id: s for s in observation.community_summaries}
            
            print(f"Meta-Agent Raw Suggestions: {raw_suggestions}")
            
            for pair in raw_suggestions:
                if not isinstance(pair, list) or len(pair) != 2:
                    continue
                
                id1, id2 = int(pair[0]), int(pair[1])
                
                # Check 1: Communities exist
                if id1 not in com_lookup or id2 not in com_lookup:
                    print(f"Merge rejected: Community {id1} or {id2} not found.")
                    continue
                
                # Check 2: Heuristic Score (Connection Strength)
                # Logic derived from mapcmcc/core/merger.py: merge_score based on edge weights
                # closeness_info in observation is exactly this sum of edge weights
                
                score_1_to_2 = com_lookup[id1].closeness_info.get(id2, 0.0)
                score_2_to_1 = com_lookup[id2].closeness_info.get(id1, 0.0)
                
                # Use the max or average, usually symmetric for undirected, but good to be safe
                heuristic_score = max(score_1_to_2, score_2_to_1)
                
                # Threshold: Must have SOME connection (score > 0)
                # You can increase this threshold to be stricter (e.g., > 1.0 or > mean_weight)
                if heuristic_score > 0:
                    valid_suggestions.append((id1, id2))
                    print(f"Merge Accepted: {id1}-{id2} (Score: {heuristic_score:.4f})")
                else:
                    print(f"Merge Rejected: {id1}-{id2} (Score: {heuristic_score:.4f} - Too low/No connection)")
            
            action = MetaAction(
                global_baselines=response_json.get("global_baselines", {}),
                budget_adjustments=budget_adjustments,
                merge_suggestions=valid_suggestions
            )
            return action
            
        except Exception as e:
            print(f"LLM Error in MetaAgent: {e}. Fallback to default.")
            return MetaAction(
                budget_adjustments={},
                global_baselines={},
                merge_suggestions=[]
            )

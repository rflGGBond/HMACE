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
        history_str = "Parameter History (Params -> Score):\n"
        if "parameter_history" in obs_dict and obs_dict["parameter_history"]:
             for entry in obs_dict["parameter_history"]:
                 history_str += f"{{ {entry['params']} }} -> {entry['global_score']}\n"
        else:
             # Default Fallback
             history_str += "{ 'cr1': 0.4, 'cr2': 0.4, 'beta': 2.0, 'alpha': 12.0 } -> N/A\n"
             
        # Format Merge History
        if "merge_history" in obs_dict and obs_dict["merge_history"]:
            history_str += "\nRecent Merges:\n"
            for m in obs_dict["merge_history"]:
                history_str += f"- Merged {m}\n"

        system_prompt = f"""
        You are the Meta Agent in the MAPCMCC evolutionary algorithm. 
        Your task is to coordinate the global optimization process by tuning parameters or MERGING communities to minimize the Global DPADV score by coordinating multiple communities.
        
        PARAMETER DEFINITIONS:
        - cr1 (0.0-1.0): Crossover Rate 1. Probability of performing crossover.
        - cr2 (0.0-1.0): Crossover Rate 2. Probability of two-way crossover.
        - beta (1.0-10.0): Local Search Intensity.
        - alpha (1.0-20.0): Search Space Reduction Factor.
        
        CRITICAL INSIGHT:
        {struggling_info_str}
        These communities are improving too slowly based on their budget allocation. 
        PRIORITIZE merging these communities with their strongly connected neighbors (check 'closeness_info') to pool resources.
        
        In-context examples:
        {history_str}
        
        TASK INSTRUCTIONS:
        1. PARAMETER OPTIMIZATION:
           - Analyze the parameter history.
           - Select high-performing sets, crossover and mutate them to generate new 'global_baselines'.
           
        2. COMMUNITY MERGING:
           - Review the 'Struggling Communities' list above.
           - Check 'closeness_info' in the observation to find strongly connected neighbors.
           - PROPOSE MERGES for struggling communities to pool resources and escape local optima.
           - Add pairs to 'merge_suggestions' (e.g., [[0, 2]]).
           
        3. BUDGET REDISTRIBUTION:
           - Move budget to communities that need it if necessary.
           - 'budget_adjustments' should be a dictionary where keys are community IDs and values are the adjustment amounts (positive to add, negative to reduce).
        
        OUTPUT RULES:
        1. Return ONLY valid JSON.
        2. Parameters in 'global_baselines' MUST be rounded to exactly 2 decimal places (e.g., 0.45, 5.00).
        3. Format:
        {{
            "reasoning": "concise explanation",
            "global_baselines": {{ "cr1": float, "cr2": float, "beta": float, "alpha": float }},
            "budget_adjustments": {{ "community_id": delta_amount }},
            "merge_suggestions": [[id1, id2]]
        }}
        """
        
        user_prompt = f"Current Global Observation: {json.dumps(obs_dict, default=str)}\n\nRespond with valid JSON only."
        
        # 2. Call LLM
        try:
            response_str = self.llm_client.get_completion(system_prompt, user_prompt, temperature=0.5)
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

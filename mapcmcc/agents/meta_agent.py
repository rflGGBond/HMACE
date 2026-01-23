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

        # --- Identify High Danger Communities ---
        danger_communities = []
        critical_danger_communities = []
        for s in observation.community_summaries:
            d_score = getattr(s, 'danger_score', 0.0) # Safety getattr
            if d_score >= 0.6:
                critical_danger_communities.append(f"{s.community_id} (Score: {d_score:.2f})")
            elif d_score >= 0.3:
                danger_communities.append(f"{s.community_id} (Score: {d_score:.2f})")
        
        danger_info_str = ""
        if danger_communities or critical_danger_communities:
            danger_info_str = "DANGER ALERTS:\n"
            if critical_danger_communities:
                danger_info_str += f"- CRITICAL DANGER (Level 2 - MUST MERGE/RESTRUCTURE): {critical_danger_communities}\n"
            if danger_communities:
                danger_info_str += f"- WARNING (Level 1 - Monitor/Tune): {danger_communities}\n"
        else:
            danger_info_str = "No communities currently in danger zone.\n"
        
        print(f"Meta-Agent Danger Analysis: {danger_info_str.strip()}")

        # Get valid IDs for prompt grounding
        valid_ids = [s.community_id for s in observation.community_summaries]
        num_communities = len(valid_ids)
        valid_ids_str = f"AVAILABLE COMMUNITY IDs: {valid_ids}"

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
             history_str += "{ 'cr1': 0.3, 'cr2': 0.3, 'beta': 2.0, 'alpha': 12.0 } -> N/A\n"
             
        # Format Merge History
        if "merge_history" in obs_dict and obs_dict["merge_history"]:
            history_str += "\nRecent Merges:\n"
            for m in obs_dict["merge_history"]:
                history_str += f"- Merged {m}\n"

        # --- Dynamic Task Instructions ---
        if num_communities == 1:
            task_instructions = """
        TASK INSTRUCTIONS:
        1. PARAMETER OPTIMIZATION (SOLE FOCUS):
        - You are now in the GLOBAL EVOLUTION PHASE with a SINGLE community.
        - Your ONLY goal is to fine-tune the parameters to squeeze out the last bit of performance.
        - Analyze the parameter history deeply.
        - Select high-performing sets, crossover and mutate them to generate new 'global_baselines'.
        
        2. COMMUNITY MERGING:
        - DISABLED. There is only one community left. Return an empty list [].
        
        3. BUDGET REDISTRIBUTION:
        - DISABLED. Only one community. Return an empty dictionary {}.
            """
        else:
            task_instructions = f"""
        TASK INSTRUCTIONS:
        1. PARAMETER OPTIMIZATION:
        - Analyze the parameter history.
        - Select high-performing sets, crossover and mutate them to generate new 'global_baselines'.
        - ACTION C (Forced Perturbation): If CRITICAL DANGER communities exist, you MUST set aggressive parameters (e.g., Alpha > 20.0, CR2 > 0.8, Beta < 5.0) to force "Jump Candidates" that explore boundary nodes.
           
        2. COMMUNITY MERGING:
        - Review the 'AVAILABLE COMMUNITY IDs' and 'Struggling Communities' list above.
        - Check 'closeness_info' in the observation to find strongly connected neighbors.
        - PROPOSE MERGES for struggling communities to pool resources and escape local optima.
        - Add pairs to 'merge_suggestions' (e.g., [[0, 2]]).
        - WARNING: You MUST ONLY propose merges between IDs listed in 'AVAILABLE COMMUNITY IDs'. Do not hallucinate IDs.
           
        3. BUDGET REDISTRIBUTION:
        - Move budget to communities that need it if necessary.
        - 'budget_adjustments' should be a dictionary where keys are community IDs and values are the adjustment amounts (positive to add, negative to reduce).
        - CONSTRAINT: The sum of all values in 'budget_adjustments' MUST be 0.
            """

        system_prompt = f"""
        You are the Meta Agent in the MAPCMCC evolutionary algorithm. 
        Your task is to coordinate the global optimization process by tuning parameters or MERGING communities to minimize the Global DPADV score by coordinating multiple communities.
        
        PARAMETER DEFINITIONS:
        - cr1 (0.0-1.0): Crossover Rate 1. Probability of performing crossover.
        - cr2 (0.0-1.0): Crossover Rate 2. Probability of two-way crossover.
        - beta (1.0-20.0): Local Search Intensity.
        - alpha (1.0-30.0): Search Space Reduction Factor.
        
        CRITICAL INSIGHT:
        {valid_ids_str}
        These are valid communities.
        
        {struggling_info_str}
        {danger_info_str}
        
        These communities are improving too slowly based on their budget allocation. 
        PRIORITIZE merging these communities (especially CRITICAL DANGER ones) with their strongly connected neighbors (check 'closeness_info') to pool resources.
        
        In-context examples:
        {history_str}
        
        {task_instructions}
        
        OUTPUT RULES:
        1. Return ONLY valid JSON.
        2. Parameters in 'global_baselines' MUST be rounded to exactly 2 decimal places (e.g., 0.45, 5.00).
        3. Format:
        {{
            "reasoning": "concise explanation",
            "global_baselines": {{ "cr1": float, "cr2": float, "beta": float, "alpha": float }},
            "budget_adjustments": {{ "1": 10, "2": -10 }},
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
                valid_adjustments = {}
                for k, v in budget_adjustments.items():
                    try:
                        valid_adjustments[int(k)] = int(v)
                    except ValueError:
                        print(f"Warning: Invalid community ID '{k}' or value '{v}' in budget_adjustments. Skipping.")
                        continue
                budget_adjustments = valid_adjustments

                # Force Zero-Sum Constraint
                total_delta = sum(budget_adjustments.values())
                if total_delta != 0 and budget_adjustments:
                    print(f"Warning: Budget adjustments sum to {total_delta} (not 0). Normalizing...")
                    # Adjust the first valid community to balance the equation
                    first_key = next(iter(budget_adjustments))
                    budget_adjustments[first_key] -= total_delta
                    print(f"Normalized: Adjusted community {first_key} by {-total_delta}.")
            
            # --- Hybrid Merge Strategy: Filter by Heuristic Score (from merger.py logic) ---
            raw_suggestions = response_json.get("merge_suggestions", [])
            
            # HARD ENFORCEMENT: No merges if only 1 community exists
            if num_communities < 2:
                if raw_suggestions:
                    print(f"Meta-Agent Enforcement: Merges disabled for single community. Ignored suggestions: {raw_suggestions}")
                raw_suggestions = []

            valid_suggestions = []
            
            # Build a lookup for community summaries to access closeness info
            com_lookup = {s.community_id: s for s in observation.community_summaries}
            
            print(f"Meta-Agent Raw Suggestions: {raw_suggestions}")
            
            for pair in raw_suggestions:
                if not isinstance(pair, list) or len(pair) != 2:
                    continue
                
                id1, id2 = int(pair[0]), int(pair[1])
                
                # Check 1: Communities exist
                missing = []
                if id1 not in com_lookup: missing.append(id1)
                if id2 not in com_lookup: missing.append(id2)
                
                if missing:
                    print(f"Merge rejected: Community {missing} not found (Request: {id1}-{id2}).")
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

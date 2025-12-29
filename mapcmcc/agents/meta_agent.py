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
        
        # Truncate history to prevent prompt overflow
        if "global_dpadv_history" in obs_dict and isinstance(obs_dict["global_dpadv_history"], list):
            obs_dict["global_dpadv_history"] = obs_dict["global_dpadv_history"][-10:] # Keep only last 10
            
        if "merge_history" in obs_dict and isinstance(obs_dict["merge_history"], list):
             obs_dict["merge_history"] = obs_dict["merge_history"][-5:] # Keep only last 5
        
        system_prompt = """
        You are the Meta Agent in the MAPCMCC evolutionary algorithm and a strict JSON generator.
        Your goal is to optimize the Global DPADV (Negative Influence Blocking) by coordinating multiple communities.
        
        Your responsibilities:
        1. Set Global Baselines for parameters ('cr1', 'cr2', 'beta', 'alpha').
        2. Redistribute Budget: Transfer budget between communities (optional).
        3. Suggest Merges: Propose merging two communities if they overlap significantly or perform poorly.
        GOAL: Optimize Global DPADV by coordinating communities.
        
        ACTIONS:
        1. "global_baselines": Set defaults for 'cr1', 'cr2', 'beta', 'alpha'.
        2. "budget_adjustments": Transfer budget (e.g., {"0": -5, "1": 5}). Keys are community IDs (strings).
        3. "merge_suggestions": List of pairs to merge (e.g., [[0, 2]]).
        
        OUTPUT RULES:
        1. Return ONLY valid JSON.
        2. NO markdown.
        3. NO explanations outside JSON.
        4. "reasoning" MUST be concise (max 20 words).
        
        EXAMPLE OUTPUT:
        {
            "reasoning": "Community 0 is stuck, merging with Community 2.",
            "global_baselines": { "cr1": 0.4, "cr2": 0.4, "beta": 2.0, "alpha": 12.0 },
            "budget_adjustments": { "0": -5, "1": 5 },
            "merge_suggestions": [[0, 2]]
        }
        """
        
        user_prompt = f"Current Global Observation: {json.dumps(obs_dict, default=str)}"
        
        # 2. Call LLM
        try:
            response_str = self.llm_client.get_completion(system_prompt, user_prompt)
            print(f"LLM Response: {response_str}")  # 输出LLM的原始响应
            response_json = json.loads(response_str)
            
            # 3. Parse Response to Action
            # Ensure keys exist
            budget_adjustments = response_json.get("budget_adjustments", {})
            # Convert string keys back to int if necessary (JSON keys are always strings)
            if budget_adjustments:
                budget_adjustments = {int(k): v for k, v in budget_adjustments.items()}
                
            action = MetaAction(
                global_baselines=response_json.get("global_baselines", {}),
                budget_adjustments=budget_adjustments,
                merge_suggestions=response_json.get("merge_suggestions", [])
            )
            return action
            
        except Exception as e:
            print(f"LLM Error in MetaAgent: {e}. Fallback to default.")
            return MetaAction(
                budget_adjustments={},
                global_baselines={},
                merge_suggestions=[]
            )

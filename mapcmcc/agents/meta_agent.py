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
        self.llm_client = llm_client or LLMClient() # Default to mock if not provided

    def get_action(self, observation: MetaObservation) -> MetaAction:
        # 1. Prepare Prompt
        obs_dict = dataclasses.asdict(observation)
        
        system_prompt = """
        You are the Meta Agent in the MAPCMCC evolutionary algorithm.
        Your goal is to optimize the Global DPADV (Negative Influence Blocking) by coordinating multiple communities.
        
        Your responsibilities:
        1. Set Global Baselines for parameters ('cr1', 'cr2', 'beta', 'alpha').
        2. Redistribute Budget: Transfer budget between communities (optional).
        3. Suggest Merges: Propose merging two communities if they overlap significantly or perform poorly.
        
        Input Format: A JSON object describing the global state and summaries of each community.
        
        IMPORTANT: Output ONLY the JSON object. Do not include any explanation, markdown formatting, or code blocks.
        
        Output Format Example:
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

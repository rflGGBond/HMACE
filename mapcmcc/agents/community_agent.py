from .base import BaseAgent
from ..utils.types import CommunityObservation, CommunityAction
from ..utils.llm_client import LLMClient
import json
import dataclasses

class CommunityAgent(BaseAgent):
    """
    Agent controlling a single community.
    Uses LLM to make decisions on parameters and candidate seeds.
    """
    def __init__(self, agent_id: str, llm_client: LLMClient = None):
        self.agent_id = agent_id
        self.llm_client = llm_client or LLMClient() # Default to mock if not provided

    def get_action(self, observation: CommunityObservation) -> CommunityAction:
        # 1. Prepare Prompt
        obs_dict = dataclasses.asdict(observation)
        # Convert complex objects to string if needed, or rely on JSON serializer
        # (Top-K nodes might need simplification to save tokens)
        
        system_prompt = """
        You are an intelligent Community Agent in the MAPCMCC evolutionary algorithm.
        Your goal is to optimize the 'DPADV' (Negative Influence Blocking) for your specific community.
        
        You have two modes of operation:
        1. Parameter Adjustment (Mode A): Tune 'cr1', 'cr2' (crossover rates), 'beta' (local search strength), 'alpha'.
        2. Candidate Generation (Mode B): Propose a specific list of node IDs ('candidate_seed_set') to replace the current seed.
        
        Input Format: A JSON object describing the current state of your community.
        
        IMPORTANT: Output ONLY the JSON object. Do not include any explanation, markdown formatting, or code blocks.
        
        Output Format Example:
        {
            "reasoning": "Performance is stagnant, increasing mutation rates.",
            "action_type": "adjust_parameters" or "propose_candidate",
            "parameters": { "cr1": 0.5, "cr2": 0.5, "beta": 3.0, "alpha": 10.0 },
            "candidate_seed_set": null
        }
        """
        
        user_prompt = f"Current Observation: {json.dumps(obs_dict, default=str)}"
        
        # 2. Call LLM
        try:
            response_str = self.llm_client.get_completion(system_prompt, user_prompt)
            response_json = json.loads(response_str)
            
            # 3. Parse Response to Action
            action = CommunityAction()
            
            if response_json.get("action_type") == "adjust_parameters":
                action.parameters = response_json.get("parameters")
            
            elif response_json.get("action_type") == "propose_candidate":
                action.candidate_seed_set = response_json.get("candidate_seed_set")
                
            return action
            
        except Exception as e:
            print(f"LLM Error in CommunityAgent {self.agent_id}: {e}. Fallback to default.")
            return CommunityAction() # Return empty action (do nothing)


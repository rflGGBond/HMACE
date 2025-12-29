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
        
        # Truncate history to prevent prompt overflow
        if "dpadv_history" in obs_dict and isinstance(obs_dict["dpadv_history"], list):
            obs_dict["dpadv_history"] = obs_dict["dpadv_history"][-10:] # Keep only last 10 entries
            
        # Truncate boundary info if too large (optional, but recommended)
        if "boundary_info" in obs_dict and isinstance(obs_dict["boundary_info"], dict):
             # Just keep a summary or limit keys if needed. For now, we trust it's not huge or we just summarize size.
             # obs_dict["boundary_info"] = f"Boundary Info with {len(obs_dict['boundary_info'])} items"
             pass

        system_prompt = """
        You are an intelligent Community Agent in the MAPCMCC evolutionary algorithm and a strict JSON generator.
        Your goal is to optimize the 'DPADV' (Negative Influence Blocking) for your specific community.
        
        GOAL: Optimize 'DPADV' (Blocking Influence) for your community.
        
        MODES:
        A. "adjust_parameters": Tune 'cr1', 'cr2' (0.0-1.0), 'beta' (1.0-10.0), 'alpha' (1.0-20.0).
        B. "propose_candidate": Propose a list of integer node IDs to be the new seed set. (Size MUST match 'budget')
        
        INPUT: JSON state of your community.
        
        OUTPUT RULES:
        1. Return ONLY valid JSON.
        2. NO markdown (no ```json).
        3. NO explanations outside the JSON.
        4. "action_type" MUST be "adjust_parameters" OR "propose_candidate".
        5. "reasoning" MUST be a single concise sentence (max 20 words). Keep it brief.
        6. "candidate_seed_set" size MUST equal the 'budget' value in input.
        
        EXAMPLE OUTPUT:
        {
            "reasoning": "Performance is stagnant, increasing mutation rates.",
            "action_type": "adjust_parameters",
            "parameters": { "cr1": 0.5, "cr2": 0.5, "beta": 3.0, "alpha": 10.0 },
            "candidate_seed_set": null
        }
        """
        
        user_prompt = f"Current Observation: {json.dumps(obs_dict, default=str)}"
        
        # 2. Call LLM
        try:
            response_str = self.llm_client.get_completion(system_prompt, user_prompt)
            print(f"LLM Response: {response_str}")  # 输出LLM的原始响应
            response_json = json.loads(response_str)
            
            # 3. Parse Response to Action
            action = CommunityAction()
            
            if response_json.get("action_type") == "adjust_parameters":
                action.parameters = response_json.get("parameters")
            
            elif response_json.get("action_type") == "propose_candidate":
                candidates = response_json.get("candidate_seed_set")
                if candidates and isinstance(candidates, list):
                    # Auto-fix: Truncate if too long
                    if len(candidates) > observation.budget:
                        print(f"DEBUG: Truncating candidate set from {len(candidates)} to {observation.budget}")
                        candidates = candidates[:observation.budget]
                    action.candidate_seed_set = candidates
                else:
                     action.candidate_seed_set = candidates # Pass through if None or invalid type, let validation handle it
                
            return action
            
        except Exception as e:
            # Enhanced error logging with raw hex output for invisible characters check
            import binascii
            raw_hex = binascii.hexlify(response_str.encode('utf-8', errors='ignore')).decode() if 'response_str' in locals() else "N/A"
            print(f"LLM Error in CommunityAgent {self.agent_id}: {e}.")
            if 'response_str' in locals():
                print(f"DEBUG: Failed Raw Response (Hex): {raw_hex}")
            print("Fallback to default.")
            return CommunityAction()  # Return empty action (do nothing)

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
        # 1. Prepare Observation
        obs_dict = dataclasses.asdict(observation)
        # print(f"Community Agent {self.agent_id} Observation: {obs_dict}")
        # print(f"Community Agent {self.agent_id} Budget: {observation.budget}")
        
        # Truncate history to prevent prompt overflow
        if "dpadv_history" in obs_dict and isinstance(obs_dict["dpadv_history"], list):
            obs_dict["dpadv_history"] = obs_dict["dpadv_history"][-10:] # Keep only last 10 entries
            
        # Truncate boundary info if too large
        if "boundary_info" in obs_dict and isinstance(obs_dict["boundary_info"], dict):
            b_info = obs_dict["boundary_info"]
            # If neighbor_ids list is too long, truncate it
            if "neighbor_ids" in b_info and isinstance(b_info["neighbor_ids"], list):
                if len(b_info["neighbor_ids"]) > 50:
                    b_info["neighbor_ids"] = b_info["neighbor_ids"][:50]
                    b_info["neighbor_ids_truncated"] = True
                    b_info["total_neighbors"] = len(obs_dict["boundary_info"]["neighbor_ids"]) # Store original length
            
            # If boundary_nodes list exists (future proofing) and is too long
            if "boundary_nodes" in b_info and isinstance(b_info["boundary_nodes"], list):
                if len(b_info["boundary_nodes"]) > 50:
                    b_info["boundary_nodes"] = b_info["boundary_nodes"][:50]
                    b_info["boundary_nodes_truncated"] = True

        obs_json_str = json.dumps(obs_dict, default=str)
        
        try:
            # --- STEP 1: DECIDE ACTION TYPE (Low Temperature for Stability) ---
            step1_system_prompt = f"""
            You are an intelligent Community Agent in the MAPCMCC evolutionary algorithm.
            
            GOAL: Minimize 'DPADV' (Blocking Influence) for your community.
            
            TASK: Analyze the current state and decide which action mode to take:
            A. "adjust_parameters": Tune evolutionary parameters if performance is stagnant or needs fine-tuning.
            B. "propose_candidate": Propose a new seed set if exploration is needed.
            
            OUTPUT RULES:
            1. Return ONLY valid JSON.
            2. Output format: {{ "reasoning": "...", "action_type": "..." }}
            3. "action_type" MUST be strictly either "adjust_parameters" or "propose_candidate". Do not output "both".
            4. "reasoning" MUST be a single concise sentence (max 20 words).
            """
            step1_user_prompt = f"Current Observation: {obs_json_str}\n\nDecide action type. Respond with valid JSON."
            
            response_step1_str = self.llm_client.get_completion(step1_system_prompt, step1_user_prompt, temperature=0.8)
            print(f"Community Agent {self.agent_id} Step 1 Response: {response_step1_str}")
            step1_json = json.loads(response_step1_str)
            
            action_type = step1_json.get("action_type")
            reasoning = step1_json.get("reasoning", "")
            
            # Handle "both" hallucination by prioritizing candidate proposal
            if action_type and "both" in action_type.lower():
                print(f"Agent {self.agent_id} returned 'both'. Defaulting to 'propose_candidate'.")
                action_type = "propose_candidate"
            
            # Initialize action
            action = CommunityAction()
            
            # --- STEP 2: GENERATE CONTENT ---
            
            if action_type == "adjust_parameters":
                # Mode A: Parameter Tuning
                param_temp = 0.75
                
                step2_system_prompt = f"""
                You decided to 'adjust_parameters'.
                
                TASK: Tune 'cr1', 'cr2' (0.0-1.0), 'beta' (1.0-20.0), 'alpha' (1.0-30.0).

                PARAMETER DEFINITIONS:
                - cr1 (0.0-1.0): Crossover Rate 1. Probability of performing crossover. Higher values mean more gene exchange.
                - cr2 (0.0-1.0): Crossover Rate 2. Probability of two-way crossover vs one-way.
                - beta (1.0-20.0): Local Search Intensity. Higher values imply more aggressive local optimization.
                - alpha (1.0-30.0): Search Space Reduction Factor. Determines the pool size of candidate nodes (alpha * budget). Higher values allow wider exploration but slower convergence.
                
                OUTPUT RULES:
                1. Return ONLY valid JSON.
                2. Output format: {{ "parameters": {{ "cr1": ..., "cr2": ..., "beta": ..., "alpha": ... }} }}
                3. Parameters MUST be rounded to exactly 2 decimal places (e.g., 0.45, 5.00).
                4. NO comments (like // ... or /* ... */) inside the JSON. Standard JSON does not support comments.
                """
                step2_user_prompt = f"Current Observation: {obs_json_str}\n\nReasoning: {reasoning}\n\nGenerate parameters. Respond with valid JSON."
                
                response_step2_str = self.llm_client.get_completion(step2_system_prompt, step2_user_prompt, temperature=param_temp)
                print(f"Community Agent {self.agent_id} Step 2 (Mode A) Response: {response_step2_str}")
                step2_json = json.loads(response_step2_str)
                
                action.parameters = step2_json.get("parameters")

            elif action_type == "propose_candidate":
                # Mode B: High/Adaptive Temperature
                
                # Format Population History for the prompt
                history_str = ""
                if "solution_history" in obs_dict and obs_dict["solution_history"]:
                     for sol in obs_dict["solution_history"]:
                         history_str += f"{{ {sol['seed']} }} {{ {sol['score']} }}\n"
                else:
                     # Fallback if no history yet
                     history_str = f"{{ {observation.current_seed_set} }} {{ {observation.current_dpadv} }}\n"

                step2_system_prompt = f"""
                Description of problem and solution properties
                You are given a list of top potential nodes with scores: {observation.top_k_score_nodes}. Your task is to find a seed set of size {observation.budget}, with the lowest possible DPADV score (blocking influence), that minimizes the negative influence.

                In-context examples (population)
                Below are some previous seed sets and their DPADV scores. The sets are arranged in descending order based on their scores, where lower values are better.

                {history_str}

                Task instructions
                Please follow the instruction step-by-step to generate a new seed set:
                1. Analyze the historical successful seed sets (In-context examples) to identify effective blocking nodes.
                2. Select high-potential nodes from the provided {observation.top_k_score_nodes} list.
                3. Synthesize these insights to construct a single, superior seed set of size {observation.budget}.
                4. Ensure the set is diverse and strategically positioned to minimize DPADV.
                
                Directly give me the final generated seed set in JSON format: {{ "candidate_seed_set": [id1, id2, ...] }}
                """
                step2_user_prompt = f"Current Observation: {obs_json_str}\n\nReasoning: {reasoning}\n\nGenerate candidate seed set. Respond with valid JSON."
                
                response_step2_str = self.llm_client.get_completion(step2_system_prompt, step2_user_prompt, temperature=0.7)
                print(f"Community Agent {self.agent_id} Step 2 (Mode B) Response: {response_step2_str}")
                step2_json = json.loads(response_step2_str)
                
                candidates = step2_json.get("candidate_seed_set")
                if candidates and isinstance(candidates, list):
                    if len(candidates) > observation.budget:
                        print(f"Truncating candidate set from {len(candidates)} to {observation.budget}")
                        candidates = candidates[:observation.budget]
                    action.candidate_seed_set = candidates
                else:
                     action.candidate_seed_set = candidates

            else:
                print(f"Unknown action type: {action_type}")
                
            return action
            
        except Exception as e:
            print(f"LLM Error in CommunityAgent {self.agent_id}: {e}.")
            print("Fallback to default.")
            return CommunityAction()  # Return empty action (do nothing)

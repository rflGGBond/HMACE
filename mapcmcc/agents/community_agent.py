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
        
        # --- DANGER ANALYSIS (Level 0/1/2) ---
        danger_score = obs_dict.get("danger_score", 0.0)
        danger_level = 0
        if danger_score >= 0.6:
            danger_level = 2 # Critical
        elif danger_score >= 0.3:
            danger_level = 1 # Warning
            
        danger_context = ""
        if danger_level == 1:
            danger_context = f"""
            WARNING: DANGER LEVEL 1 DETECTED (Score: {danger_score:.2f}).
            The community is showing signs of stagnation and closure.
            REQUIRED ACTION:
            - You MUST adopt a more AGGRESSIVE exploration strategy.
            - Prioritize 'adjust_parameters' to increase exploration (higher cr2, higher alpha) OR 'propose_candidate' with BOUNDARY INJECTION.
            """
        elif danger_level == 2:
            danger_context = f"""
            CRITICAL: DANGER LEVEL 2 DETECTED (Score: {danger_score:.2f}).
            The community is critically stagnant. Meta-Agent may intervene soon.
            REQUIRED ACTION:
            - Maximize exploration immediately.
            - If proposing candidates, you MUST include boundary nodes to break the structure.
            """

        try:
            # --- STEP 1: DECIDE ACTION TYPE (Low Temperature for Stability) ---
            step1_system_prompt = f"""
            You are an intelligent Community Agent in the MAPCMCC evolutionary algorithm.
            
            GOAL: Minimize 'DPADV' (Blocking Influence) for your community.
            
            {danger_context}
            
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
                
                # --- Danger-based Parameter Correction (Rule-based Override) ---
                if danger_score > 0 and action.parameters:
                    # Constants
                    CR2_MAX = 1.0
                    BETA_MIN = 1.0
                    ALPHA_MAX = 30.0
                    
                    # Scaling Factors (eta)
                    ETA_1 = 0.5  # For cr2 (0-1 scale)
                    ETA_2 = 5.0  # For beta (1-20 scale)
                    ETA_3 = 10.0 # For alpha (1-30 scale)
                    
                    # Original values
                    cr2 = float(action.parameters.get("cr2", 0.5))
                    beta = float(action.parameters.get("beta", 5.0))
                    alpha = float(action.parameters.get("alpha", 10.0))
                    
                    # Apply Rules
                    # cr2 <- min(cr2_max, cr2 + eta1 * Danger)
                    new_cr2 = min(CR2_MAX, cr2 + ETA_1 * danger_score)
                    
                    # beta <- max(beta_min, beta - eta2 * Danger)
                    new_beta = max(BETA_MIN, beta - ETA_2 * danger_score)
                    
                    # alpha <- min(alpha_max, alpha + eta3 * Danger)
                    new_alpha = min(ALPHA_MAX, alpha + ETA_3 * danger_score)
                    
                    # Update Action
                    action.parameters["cr2"] = round(new_cr2, 2)
                    action.parameters["beta"] = round(new_beta, 2)
                    action.parameters["alpha"] = round(new_alpha, 2)
                    
                    if danger_level > 0:
                        print(f"Agent {self.agent_id}: Danger Correction Applied (Score: {danger_score:.2f})")
                        print(f"  cr2: {cr2} -> {new_cr2:.2f} | beta: {beta} -> {new_beta:.2f} | alpha: {alpha} -> {new_alpha:.2f}")

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

                # Boundary Injection Context
                boundary_context = ""
                if "boundary_info" in obs_dict and obs_dict["boundary_info"]:
                     b_nodes = obs_dict["boundary_info"].get("boundary_nodes", [])
                     if b_nodes:
                         boundary_context = f"Available BOUNDARY NODES (Key for breaking closed structures): {b_nodes}"

                step2_system_prompt = f"""
                ### Role & Objective
                You are an expert Evolutionary Strategy Agent. Your goal is to select a seed set of size {observation.budget} to MINIMIZE the DPADV score (blocking negative influence).

                ### Context Data
                1. **High Potential Candidates**: {observation.top_k_score_nodes} (Nodes with high local centrality).
                2. **System Status**:
                {danger_context}
                {boundary_context}

                ### Historical Knowledge (Population)
                Past seed sets and their scores (lower is better):
                {history_str}

                ### Strategic Instructions
                1. **Analyze Patterns**: Identify effective blocking nodes from the historical successful sets.
                2. **Check Danger Signal**:
                   - If **CRITICAL DANGER** or **STAGNATION** is detected: You MUST prioritize **Boundary Nodes** (from the list above) to break local optima. Do not rely solely on the Top-K list.
                   - If **Normal**: Focus on refining the high-performing nodes from the Top-K list: {observation.top_k_score_nodes}.
                3. **Selection**: Construct a single, superior seed set of size {observation.budget}. Mix high-scoring nodes (Exploitation) with boundary nodes (Exploration) if needed.

                ### Output Requirement
                Return ONLY the JSON object:
                {{ "candidate_seed_set": [id1, id2, ...] }}
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
                    
                    # --- Conditional Boundary Injection Enforcement ---
                    # If Danger >= 1 AND Boundary Nodes available AND LLM missed them
                    boundary_nodes = obs_dict.get("boundary_info", {}).get("boundary_nodes", [])
                    if danger_level >= 1 and boundary_nodes and candidates:
                        has_boundary = any(node in boundary_nodes for node in candidates)
                        if not has_boundary:
                            import random
                            # Force Injection
                            # Pick a boundary node (random for now, could use scores if available)
                            forced_node = random.choice(boundary_nodes)
                            # Pick a replacement index
                            replace_idx = random.randint(0, len(candidates) - 1)
                            
                            print(f"Agent {self.agent_id}: Boundary Injection Enforced (LLM missed it). Replaced {candidates[replace_idx]} with {forced_node}.")
                            candidates[replace_idx] = forced_node
                            
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

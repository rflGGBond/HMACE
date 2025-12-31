import os
import sys
import time
import argparse
from typing import List, Dict

# Add the parent directory to sys.path to allow imports if running as script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mapcmcc.environment.env import PCMCCEnvironment
from mapcmcc.agents.community_agent import CommunityAgent
from mapcmcc.agents.meta_agent import MetaAgent
from mapcmcc.utils.types import CommunityObservation, MetaObservation
from mapcmcc.utils.llm_client import LLMClient
from utils.select_SN import select_SN

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run MAPCMCC")
    parser.add_argument("--graph_name", type=str, default="facebook", help="Name of the graph file (without extension)")
    parser.add_argument("--total_budget", type=int, default=50, help="Total budget")
    parser.add_argument("--num_communities", type=int, default=16, help="Number of communities")
    parser.add_argument("--max_gen", type=int, default=20, help="Maximum number of generations")
    parser.add_argument("--t_comm", type=int, default=5, help="Communication interval")
    
    # LLM Arguments
    parser.add_argument("--llm_provider", type=str, default="mock", choices=["mock", "local", "openai"], help="LLM Provider")
    parser.add_argument("--llm_model", type=str, default="qwen3-max", help="Model name (or path for local)")
    parser.add_argument("--api_key", type=str, default=None, help="API Key for OpenAI")
    parser.add_argument("--model_root", type=str, default="../../models", help="Root directory for local models")

    args = parser.parse_args()

    # Configuration
    GRAPH_NAME = args.graph_name
    GRAPH_PATH = f"../graph/{GRAPH_NAME}.txt"
    SN_NODES = select_SN(GRAPH_NAME, 50) # Placeholder, should load from select_SN
    TOTAL_BUDGET = args.total_budget
    NUM_COMMUNITIES = args.num_communities
    MAX_GEN = args.max_gen
    T_COMM = args.t_comm # Communication interval
    
    # Initialize LLM Client
    print(f"Initializing LLM Client ({args.llm_provider} - {args.llm_model})...")
    llm_client = LLMClient(
        provider=args.llm_provider,
        model=args.llm_model,
        api_key=args.api_key,
        model_root=args.model_root
    )

    # Initialize Environment
    print("Initializing MAPCMCC Environment...")
    env = PCMCCEnvironment(GRAPH_PATH, SN_NODES, TOTAL_BUDGET, NUM_COMMUNITIES)
    
    # Initialize Agents
    community_agents = {}
    for com_id in env.communities:
        community_agents[com_id] = CommunityAgent(
            agent_id=f"ComAgent_{com_id}",
            llm_client=llm_client
        )
    
    # MetaAgent is currently missing in the repo or imported incorrectly, 
    # but assuming it exists or will be fixed:
    try:
        meta_agent = MetaAgent(llm_client=llm_client) 
    except Exception as e:
        print(f"Warning: Failed to initialize MetaAgent: {e}")
        meta_agent = None
    
    print("Starting Evolution Loop...")
    start_time = time.time()
    
    for gen in range(1, MAX_GEN + 1):
        print(f"\n--- Generation {gen} ---")
        
        # 1. Standard Evolution Step (PCMCC)
        env.step()
        
        # 2. Agent Interaction (Every T_comm generations)
        if gen % T_COMM == 0:
            print(">>> Triggering Multi-Agent Interaction")
            
            # A. Community Agents
            for com_id, agent in community_agents.items():
                # Get Observation
                # obs = env.communities[com_id].get_observation(...)
                # For now, mock observation construction since Env logic is skeletal
                obs = CommunityObservation(
                    community_id=com_id,
                    current_generation=gen,
                    global_stage="exploration",
                    budget=env.communities[com_id].state.budget,
                    current_dpadv=0.5,
                    dpadv_history=[],
                    diversity_score=0.1,
                    top_k_score_nodes=[],
                    current_seed_set=[],
                    boundary_info={},
                    global_dpadv=0.4
                )
                
                # Get Action (LLM/Rule-Based)
                action = agent.get_action(obs)
                
                # Apply Action
                env.apply_community_action(com_id, action)
                
            # B. Meta Agent
            if meta_agent:
                obs = env.get_global_observation()
                # Mock
                meta_obs = MetaObservation(
                    current_generation=gen,
                    current_global_dpadv=0.4,
                    global_dpadv_history=[],
                    community_summaries=[],
                    merge_history=[]
                )
                
                meta_action = meta_agent.get_action(meta_obs)
                
                # 2.1 Apply Meta-Agent Suggestions to Environment
                # Specifically handling Merge Suggestions which need to override/guide the standard merge logic
                if meta_action.merge_suggestions:
                    print(f"Meta-Agent suggests merging: {meta_action.merge_suggestions}")
                    env.set_merge_suggestions(meta_action.merge_suggestions)
                
                env.apply_meta_action(meta_action)
            
        # 3. Check Convergence
        # if env.check_convergence(): break
        
    end_time = time.time()
    print(f"\nEvolution Finished. Total Time: {end_time - start_time:.2f}s")
    print(f"Best Global DPADV: {env.global_best_dpadv}")

    # Calculate and print Negatively Activated Nodes
    from mapcmcc.core.evaluator import DPADVEvaluator
    if env.global_best_seed:
        print("Calculating final activated nodes...")
        neg_activated_count = DPADVEvaluator.get_activated_node_count(env.global_best_seed, env.Gs, env.sn_nodes)
        print(f"Negatively Activated Nodes: {neg_activated_count}")
    else:
        print("Warning: No global best seed set found.")

if __name__ == "__main__":
    main()

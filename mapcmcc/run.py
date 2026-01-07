import os
import sys
import time
import argparse
import matplotlib.pyplot as plt
from typing import List, Dict
import datetime

# Add the parent directory to sys.path to allow imports if running as script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mapcmcc.environment.env import PCMCCEnvironment
from mapcmcc.agents.community_agent import CommunityAgent
from mapcmcc.agents.meta_agent import MetaAgent
from mapcmcc.utils.types import CommunityObservation, MetaObservation
from mapcmcc.utils.llm_client import LLMClient
from mapcmcc.utils.select_SN import select_SN
from mapcmcc.core.evaluator import DPADVEvaluator
import random
import numpy as np
import torch

class LoggerWriter:
    """
    A simple class to redirect stdout to both terminal and a log file.
    """
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush() # Ensure it's written immediately

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# Define graph types based on PCMCC reference
DIRECTED_GRAPHS = {"email-Eu-core", "Email-EuAll", "p2p-Gnutella31", "soc-Epinions1"}
UNDIRECTED_GRAPHS = {"facebook", "HR", "BA3000", "ER3000", "RG3000", "WS3000"}

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run MAPCMCC")
    parser.add_argument("--graphs", type=str, nargs='+', default=["facebook"], help="List of graph names to run (e.g. facebook email-Eu-core)")
    parser.add_argument("--total_budget", type=int, nargs='+', default=[20, 110, 200], help="List of total budgets (k values)")
    parser.add_argument("--num_communities", type=int, default=16, help="Number of communities")
    parser.add_argument("--max_gen", type=int, default=20, help="Maximum number of generations")
    parser.add_argument("--t_comm", type=int, default=5, help="Communication interval")
    parser.add_argument("--mc_runs", type=int, default=100, help="Number of Monte Carlo runs for evaluation")
    parser.add_argument("--repeats", type=int, default=5, help="Number of repeats for each experiment")
    
    # LLM Arguments
    parser.add_argument("--llm_provider", type=str, default="local", choices=["mock", "local", "openai"], help="LLM Provider")
    parser.add_argument("--llm_model", type=str, default="qwen3-max", help="Model name (or path for local)")
    parser.add_argument("--api_key", type=str, default=None, help="API Key for OpenAI")
    parser.add_argument("--model_root", type=str, default="../../models", help="Root directory for local models")

    args = parser.parse_args()
    
    # Set seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # --- Setup Logging ---
    log_dir = "../results/logs/MAPCMCC"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"mapcmcc_{args.graphs[0]}_{args.llm_model}_{timestamp}.log"
    log_path = os.path.join(log_dir, log_filename)
    
    # Redirect stdout to LoggerWriter
    sys.stdout = LoggerWriter(log_path)
    
    print(f"Logging output to: {log_path}")
    print("-" * 50)

    # Configuration Constants
    K_VALUES = args.total_budget
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

    for GRAPH_NAME in args.graphs:
        print(f"\n##########################################")
        print(f"Processing Graph: {GRAPH_NAME}")
        print(f"##########################################\n")
        
        # Determine if directed
        is_directed = False
        if GRAPH_NAME in DIRECTED_GRAPHS:
            is_directed = True
            print(f"Graph '{GRAPH_NAME}' identified as DIRECTED.")
        elif GRAPH_NAME in UNDIRECTED_GRAPHS:
            is_directed = False
            print(f"Graph '{GRAPH_NAME}' identified as UNDIRECTED.")
        else:
            print(f"Warning: Graph '{GRAPH_NAME}' type unknown. Defaulting to UNDIRECTED.")
            
        GRAPH_PATH = f"../graph/{GRAPH_NAME}.txt"
        
        try:
            SN_NODES = select_SN(GRAPH_NAME, 50, is_directed=is_directed)
        except Exception as e:
            print(f"Error selecting SN nodes for {GRAPH_NAME}: {e}")
            continue

        results_coicm = []
        results_mcicm = []

        for k in K_VALUES:
            print(f"\n==========================================")
            print(f"Starting Run for {GRAPH_NAME} with Budget (k) = {k}")
            print(f"==========================================\n")

            current_k_coicm_list = []
            current_k_mcicm_list = []

            for r in range(args.repeats):
                print(f"\n--- Repeat {r+1}/{args.repeats} ---")

                # Set independent seed for this run
                current_seed = 42 + k + r * 1000
                random.seed(current_seed)
                np.random.seed(current_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(current_seed)
                torch.manual_seed(current_seed)

                # Initialize Environment
                print("Initializing MAPCMCC Environment...")
                env = PCMCCEnvironment(GRAPH_PATH, SN_NODES, k, NUM_COMMUNITIES, is_directed=is_directed)
                
                # Initialize Agents
                community_agents = {}
                for com_id in env.communities:
                    community_agents[com_id] = CommunityAgent(
                        agent_id=f"ComAgent_{com_id}",
                        llm_client=llm_client
                    )
                
                # MetaAgent
                try:
                    meta_agent = MetaAgent(llm_client=llm_client) 
                except Exception as e:
                    print(f"Warning: Failed to initialize MetaAgent: {e}")
                    meta_agent = None
                
                print("Starting Evolution Loop...")
                start_time = time.time()
                
                gen = 1
                while True:
                    print(f"\n--- Generation {gen} ---")
                    print(f"Current Communities: {len(env.communities)}")
                    
                    # 1. Standard Evolution Step (PCMCC)
                    # We pass agent_active=False so that heuristic merges are ALWAYS checked/executed in the step.
                    # This ensures the generation is "fully completed" (including standard merges) 
                    # before the Agent is called to observe the result.
                    env.step(agent_active=False)
                    
                    # Sync Agents with Environment (Handle Merges)
                    current_community_ids = set(env.communities.keys())
                    agent_ids = set(community_agents.keys())
                    
                    # Remove agents for deleted communities
                    for cid in agent_ids - current_community_ids:
                        print(f"Removing Agent for merged/deleted community {cid}")
                        del community_agents[cid]
                        
                    # Add agents for new communities
                    for cid in current_community_ids - agent_ids:
                        print(f"Initializing Agent for new community {cid}")
                        community_agents[cid] = CommunityAgent(
                            agent_id=f"ComAgent_{cid}",
                            llm_client=llm_client
                        )
                    
                    # 2. Agent Interaction (Every T_comm generations)
                    if gen % T_COMM == 0:
                        print("\n>>> Triggering Multi-Agent Interaction")
                        
                        # A. Community Agents
                        for com_id, agent in community_agents.items():
                            # Get Real Observation
                            obs_dict = env.communities[com_id].get_observation(
                                current_gen=gen,
                                global_stage="exploration", # Simplified stage logic
                                global_dpadv=env.global_best_dpadv
                            )
                            # Convert dict to Dataclass
                            obs = CommunityObservation(**obs_dict)
                            
                            # Get Action (LLM/Rule-Based)
                            action = agent.get_action(obs)
                            
                            # Apply Action
                            env.apply_community_action(com_id, action)
                            
                        # B. Meta Agent
                        if meta_agent:
                            # Get Real Global Observation
                            obs = env.get_global_observation()
                            
                            meta_action = meta_agent.get_action(obs)
                            
                            # 2.1 Apply Meta-Agent Suggestions to Environment
                            if meta_action.merge_suggestions:
                                print(f"Meta-Agent suggests merging: {meta_action.merge_suggestions}")
                                env.set_merge_suggestions(meta_action.merge_suggestions)
                            
                            env.apply_meta_action(meta_action)
                        
                    # 3. Check Convergence
                    if env.check_termination(MAX_GEN): 
                        break
                    
                    print(f"Generation {gen} Best DPADV: {env.global_best_dpadv}")
                    gen += 1
                    
                end_time = time.time()
                print(f"\nEvolution Finished for {GRAPH_NAME}, k={k}, repeat={r+1}. Total Time: {end_time - start_time:.0f}s")
                print(f"Best Global DPADV: {env.global_best_dpadv}")

                # Calculate and print Negatively Activated Nodes
                neg_activated_count_coicm = 0
                neg_activated_count_mcicm = 0
                
                if env.global_best_seed:
                    print(f"Calculating final activated nodes (Monte Carlo runs: {args.mc_runs})...")
                    
                    # COICM
                    neg_activated_count_coicm = DPADVEvaluator.get_activated_node_count(
                        env.global_best_seed, env.Gs, env.sn_nodes, runs=args.mc_runs, model='COICM'
                    )
                    print(f"Negatively Activated Nodes (COICM, k={k}, repeat={r+1}): {neg_activated_count_coicm:.0f}")
                    
                    # MCICM
                    neg_activated_count_mcicm = DPADVEvaluator.get_activated_node_count(
                        env.global_best_seed, env.Gs, env.sn_nodes, runs=args.mc_runs, model='MCICM'
                    )
                    print(f"Negatively Activated Nodes (MCICM, k={k}, repeat={r+1}): {neg_activated_count_mcicm:.0f}")
                    
                else:
                    print("Warning: No global best seed set found.")
                
                current_k_coicm_list.append(int(round(neg_activated_count_coicm)))
                current_k_mcicm_list.append(int(round(neg_activated_count_mcicm)))

            # Average results
            avg_coicm = int(round(sum(current_k_coicm_list) / len(current_k_coicm_list))) if current_k_coicm_list else 0
            avg_mcicm = int(round(sum(current_k_mcicm_list) / len(current_k_mcicm_list))) if current_k_mcicm_list else 0
            
            print(f"Negatively Activated Nodes (COICM, k={k}): {avg_coicm:.0f}")
            print(f"Negatively Activated Nodes (MCICM, k={k}): {avg_mcicm:.0f}")

            results_coicm.append(avg_coicm)
            results_mcicm.append(avg_mcicm)

        # Plotting Results
        print(f"\nGenerating Plots for {GRAPH_NAME}...")
        
        # 1. COICM Plot
        try:
            output_fig_dir_coicm = f"../results/COICM/MAPCMCC/"
            if not os.path.exists(output_fig_dir_coicm):
                os.makedirs(output_fig_dir_coicm)
            
            plt.figure(figsize=(6, 6))
            plt.plot(K_VALUES, results_coicm, marker='o', linestyle='--', label=GRAPH_NAME, color='salmon')
            
            for x, y in zip(K_VALUES, results_coicm):
                plt.text(x, y, f'{y:.0f}', ha='center', va='bottom')
                
            plt.title(f'COICM {GRAPH_NAME}')
            plt.xlabel('k')
            plt.ylabel('Negatively Activated Nodes')
            plt.xticks(K_VALUES)
            plt.tight_layout()
            
            plot_path_coicm = os.path.join(output_fig_dir_coicm, f'COICM_{GRAPH_NAME}_{args.llm_model}.png')
            plt.savefig(plot_path_coicm)
            plt.close()
            print(f"Saved COICM plot to {plot_path_coicm}")
            
        except Exception as e:
            print(f"Error plotting COICM: {e}")

        # 2. MCICM Plot
        try:
            output_fig_dir_mcicm = f"../results/MCICM/MAPCMCC/"
            if not os.path.exists(output_fig_dir_mcicm):
                os.makedirs(output_fig_dir_mcicm)
            
            plt.figure(figsize=(6, 6))
            plt.plot(K_VALUES, results_mcicm, marker='o', linestyle='--', label=GRAPH_NAME, color='skyblue')
            
            for x, y in zip(K_VALUES, results_mcicm):
                plt.text(x, y, f'{y:.0f}', ha='center', va='bottom')
                
            plt.title(f'MCICM {GRAPH_NAME}')
            plt.xlabel('k')
            plt.ylabel('Negatively Activated Nodes')
            plt.xticks(K_VALUES)
            plt.tight_layout()
            
            plot_path_mcicm = os.path.join(output_fig_dir_mcicm, f'MCICM_{GRAPH_NAME}_{args.llm_model}.png')
            plt.savefig(plot_path_mcicm)
            plt.close()
            print(f"Saved MCICM plot to {plot_path_mcicm}")
            
        except Exception as e:
            print(f"Error plotting MCICM: {e}")

if __name__ == "__main__":
    main()
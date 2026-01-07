import networkx as nx
import os
import sys
import random

def get_unique_filepath(directory, filename):
    """
    Ensure the filename is unique in the directory.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    name, ext = os.path.splitext(filename)
    full_path = os.path.join(directory, filename)
    
    if not os.path.exists(full_path):
        return full_path
        
    counter = 1
    while True:
        new_filename = f"{name}_{counter}{ext}"
        full_path = os.path.join(directory, new_filename)
        if not os.path.exists(full_path):
            return full_path
        counter += 1

def save_network(G, filepath):
    """
    Save the network to a TXT file with 'u v weight' format.
    Weights are randomly chosen from {0.01, 0.05, 0.2}.
    """
    try:
        with open(filepath, 'w') as f:
            # Format: u v weight
            possible_weights = [0.01, 0.05, 0.2]
            for u, v in G.edges():
                w = random.choice(possible_weights)
                f.write(f"{u} {v} {w}\n")
        print(f"Successfully saved network to: {filepath}")
    except Exception as e:
        print(f"Error saving network: {e}")

def get_params_for_edges(n, m_target, network_type):
    """
    Calculate model parameters to approximate the target number of edges.
    """
    if network_type == 'scale_free': # Barabasi-Albert
        # Total edges E approx m * n (for large n)
        # m = E / n
        m = int(round(m_target / n))
        if m < 1: m = 1
        if m >= n: m = n - 1
        return {'m': m}
        
    elif network_type == 'regular':
        # E = n * d / 2  => d = 2 * E / n
        d = int(round(2 * m_target / n))
        # d must be less than n
        if d >= n: d = n - 1
        # n * d must be even
        if (n * d) % 2 != 0:
            # Adjust d to make n*d even
            # Try d+1 or d-1
            if d + 1 < n:
                d += 1
            else:
                d -= 1
        if d < 0: d = 0 # Disconnected
        return {'d': d}
        
    elif network_type == 'small_world': # Watts-Strogatz
        # E = n * k / 2 => k = 2 * E / n
        k = int(round(2 * m_target / n))
        if k >= n: k = n - 1
        # k must be even in some implementations, but nx.watts_strogatz_graph requires k (int)
        # Each node is connected to k nearest neighbors in ring topology
        # So k should be even? Documentation says "Each node is joined with its k nearest neighbors in a ring topology."
        # It doesn't strictly enforce even, but standard WS model usually implies even k for symmetry.
        # But let's check nx documentation logic: "k (int) – Each node is joined with its k nearest neighbors in a ring topology."
        return {'k': k, 'p': 0.1} # Default p
        
    elif network_type == 'random': # Erdos-Renyi (gnm)
        return {'m': m_target}
        
    return {}

def main():
    random.seed(42) # Ensure reproducibility
    print("=== Network Generation Tool ===")
    print("1. Scale-Free Network (Barabasi-Albert)")
    print("2. Regular Network")
    print("3. Small-World Network (Watts-Strogatz)")
    print("4. Random Network (Erdos-Renyi)")
    
    try:
        type_choice = input("Select network type (1-4): ").strip()
        if type_choice not in ['1', '2', '3', '4']:
            print("Invalid selection.")
            return

        n_str = input("Enter number of nodes (N): ").strip()
        n = int(n_str)
        if n <= 0:
            print("Number of nodes must be positive.")
            return

        edge_input = input("Enter number of edges (integer or 'auto'): ").strip().lower()
        
        m_target = None
        if edge_input == 'auto':
            # Default to average degree of ~4
            m_target = n * 2
        else:
            try:
                m_target = int(edge_input)
                if m_target < 0:
                    print("Number of edges must be non-negative.")
                    return
            except ValueError:
                print("Invalid input for edges. Using auto mode.")
                m_target = n * 2

        # Generate Graph
        G = None
        prefix = ""
        
        if type_choice == '1': # Scale-Free
            params = get_params_for_edges(n, m_target, 'scale_free')
            print(f"Generating Scale-Free network with N={n}, m={params['m']}...")
            G = nx.barabasi_albert_graph(n, params['m'])
            prefix = "BA"
            
        elif type_choice == '2': # Regular
            params = get_params_for_edges(n, m_target, 'regular')
            print(f"Generating Regular network with N={n}, d={params['d']}...")
            try:
                G = nx.random_regular_graph(params['d'], n)
            except nx.NetworkXError as e:
                print(f"Error creating regular graph: {e}")
                return
            prefix = "RG"
            
        elif type_choice == '3': # Small-World
            params = get_params_for_edges(n, m_target, 'small_world')
            print(f"Generating Small-World network with N={n}, k={params['k']}, p={params['p']}...")
            try:
                G = nx.watts_strogatz_graph(n, params['k'], params['p'])
            except nx.NetworkXError as e:
                print(f"Error creating small-world graph: {e}")
                return
            prefix = "WS"
            
        elif type_choice == '4': # Random
            params = get_params_for_edges(n, m_target, 'random')
            print(f"Generating Random network with N={n}, M={params['m']}...")
            G = nx.gnm_random_graph(n, params['m'])
            prefix = "ER"

        if G is None:
            print("Failed to generate network.")
            return

        # Prepare storage
        # Path to PCMCC/graph relative to this script
        # This script is in /PCMCC/datasets/
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir) # PCMCC
        data_dir = os.path.join(project_root, 'graph')
        
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            
        filename = f"{prefix}{n}.txt"
        filepath = get_unique_filepath(data_dir, filename)
        
        print(f"Network generated. Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
        save_network(G, filepath)
    
            
    except ValueError as e:
        print(f"Input Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()

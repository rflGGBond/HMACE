import os

graph_dir = "/data/lfr/HMACE/graph"
files = [f for f in os.listdir(graph_dir) if f.endswith(".txt")]
files.sort()

print(f"{'Graph Name':<25} | {'Nodes':<10} | {'Edges':<10}")
print("-" * 50)

for filename in files:
    filepath = os.path.join(graph_dir, filename)
    nodes = set()
    edge_count = 0
    try:
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    u, v = parts[0], parts[1]
                    nodes.add(u)
                    nodes.add(v)
                    edge_count += 1
        print(f"{filename:<25} | {len(nodes):<10} | {edge_count:<10}")
    except Exception as e:
        print(f"{filename:<25} | Error: {e}")

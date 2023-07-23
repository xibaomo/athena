import networkx as nx

# Create a directed graph (You can create your own graph here)
G = nx.DiGraph()

# Add nodes to the graph
G.add_nodes_from(["Node A", "Node B", "Node C", "Node D"])

# Add weighted edges to the graph
G.add_edge("Node A", "Node B", weight=5)
G.add_edge("Node A", "Node C", weight=3)
G.add_edge("Node B", "Node D", weight=7)
G.add_edge("Node C", "Node D", weight=2)
G.add_edge("Node D", "Node A", weight=4)  # Adding a path from 'Node D' back to 'Node A'

# Find all paths from 'Node A' to 'Node A' (including self-loops)
all_paths = list(nx.all_simple_paths(G, source="Node A", target="Node D",cutoff=None))

# Print all paths
print("All paths from 'Node A' to 'Node A':")
for path in all_paths:
    print(" -> ".join(path))


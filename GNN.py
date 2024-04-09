from torch_geometric.datasets import Reddit

# Load the Reddit dataset
dataset = Reddit(root='./data/Reddit')

# Print dataset information
print("Dataset: Reddit")
print("===================================")
print("Number of graphs:", len(dataset))
print("Number of features per node:", dataset.num_features)
print()

# Analyze properties of each graph in the dataset
for i in range(len(dataset)):
    data = dataset[i]
    print(f"Graph {i+1}:")
    print("Number of nodes:", data.num_nodes)
    print("Number of edges:", data.num_edges)
    print("Has self-loops:", data.has_self_loops())
    print("Has isolated nodes:", data.has_isolated_nodes())
    print("Is undirected:", data.is_undirected())
    print()

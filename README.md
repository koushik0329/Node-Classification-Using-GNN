# Node-Classification-Using-GNN

**Graph Neural Networks** (GNNs) are a class of neural networks designed to operate on graph-structured data. Unlike traditional neural networks that operate on data in the form of grid (images) or sequential data (text), GNNs are tailored for data represented in the form of graphs, which consist of nodes connected by edges. GNNs are used in predicting nodes, edges, and graph-based tasks.

Node classification using Graph Neural Networks (GNNs) is a task where the goal is to predict labels or categories for each node in a graph based on the graph's structure and node attributes. This task is commonly applied to various types of graphs, such as social networks, citation networks, and biological networks, where nodes represent entities and edges represent relationships or interactions between them.

**GNN.py** is used to analyze the properties of each graph used in the dataset and can also know the dataset information

**gat.py** contains code for training and evaluating a Graph Attention Network (GAT) model for node classification on the Cora citation network dataset using PyTorch Geometric. The GAT model is implemented in PyTorch and leverages the GATConv layer from PyTorch Geometric for message passing and feature transformation. The code includes data loading, model definition, training, and evaluation steps.

**gcn.py** contains code for training and evaluating a Graph Convolutional Network (GCN) model for node classification on the Cora citation network dataset using PyTorch Geometric. The GCN model is implemented in PyTorch and utilizes the GCNConv layer from PyTorch Geometric for message passing and feature transformation. The code includes data loading, model definition, training, and evaluation steps.

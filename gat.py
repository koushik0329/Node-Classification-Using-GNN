import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GATConv
from torch_geometric.data import DataLoader

# Step 1: Load Dataset
dataset = Planetoid(root='./data/Planetoid', name='Cora')

# Step 2: Define GAT Model
class GAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads):
        super(GAT, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=num_heads, concat=True, dropout=0.6)
        self.conv2 = GATConv(hidden_dim * num_heads, output_dim, heads=1, concat=False, dropout=0.6)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Step 3: Train Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GAT(dataset.num_features, hidden_dim=8, output_dim=dataset.num_classes, num_heads=8).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
criterion = nn.NLLLoss()

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(dataset[0].to(device))
    loss = criterion(out, dataset[0].y.to(device))
    loss.backward()
    optimizer.step()

# Step 4: Evaluate Model
model.eval()
with torch.no_grad():
    pred = model(dataset[0].to(device))
    pred = pred.argmax(dim=1)
    accuracy = (pred == dataset[0].y.to(device)).sum().item() / len(dataset[0].y)

print(f'Test Accuracy: {accuracy}')

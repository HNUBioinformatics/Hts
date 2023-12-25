import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, GCNConv, JumpingKnowledge, global_max_pool

class GIN_drug(torch.nn.Module):
    def __init__(self, nn1, dim):
        super().__init__()
        self.nn1 = nn1
        self.dim = dim
        self.JK = JumpingKnowledge('cat')
        self.convs_drug = torch.nn.ModuleList()
        self.bns_drug = torch.nn.ModuleList()

        self.conv = GCNConv(225, 100)

        self.dropout = nn.Dropout(0.3)

        for i in range(self.nn1):
            if i:
                block = nn.Sequential(nn.Linear(self.dim, self.dim), nn.ReLU(),
                                      nn.Linear(self.dim, self.dim))
            else:
                block = nn.Sequential(nn.Linear(75, self.dim), nn.ReLU(), nn.Linear(self.dim, self.dim))
            conv = GINConv(block)
            bn = torch.nn.BatchNorm1d(self.dim)

            self.convs_drug.append(conv)
            self.bns_drug.append(bn)

    def forward(self, drug_feature, drug_adj, ibatch):
        x, edge_index, batch = drug_feature, drug_adj, ibatch
        list = []
        for i in range(self.nn1):
            x = F.relu(self.dropout(self.convs_drug[i](x, edge_index)))

            # x = self.dropout(x)

            x = self.bns_drug[i](x)

            # x = self.dropout(x)

            list.append(x)

            # x = self.dropout(x)

        exp = self.JK(list)
        drug = global_max_pool(exp, batch)

        return drug

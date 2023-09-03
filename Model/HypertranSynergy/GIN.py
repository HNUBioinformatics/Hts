import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, JumpingKnowledge, global_max_pool, GCNConv

class GNN_drug(torch.nn.Module):
    def __init__(self, layer_drug, dim_drug):
        super().__init__()
        self.layer_drug = layer_drug
        self.dim_drug = dim_drug
        self.JK = JumpingKnowledge('cat')
        self.convs_drug = torch.nn.ModuleList()
        self.bns_drug = torch.nn.ModuleList()

        self.conv = GCNConv(225, 100)

        self.dropout = nn.Dropout(0.3)

        for i in range(self.layer_drug):
            if i:
                block = nn.Sequential(nn.Linear(self.dim_drug, self.dim_drug), nn.ReLU(),
                                      nn.Linear(self.dim_drug, self.dim_drug))
            else:
                block = nn.Sequential(nn.Linear(75, self.dim_drug), nn.ReLU(), nn.Linear(self.dim_drug, self.dim_drug))
            conv = GINConv(block)
            bn = torch.nn.BatchNorm1d(self.dim_drug)

            self.convs_drug.append(conv)
            self.bns_drug.append(bn)

    def forward(self, drug_feature, drug_adj, ibatch):
        x, edge_index, batch = drug_feature, drug_adj, ibatch
        x_drug_list = []
        for i in range(self.layer_drug):
            x = F.relu(self.dropout(self.convs_drug[i](x, edge_index)))

            # x = self.dropout(x)

            x = self.bns_drug[i](x)

            # x = self.dropout(x)

            x_drug_list.append(x)

            # x = self.dropout(x)

        node_representation = self.JK(x_drug_list)
        x_drug = global_max_pool(node_representation, batch)

        return x_drug
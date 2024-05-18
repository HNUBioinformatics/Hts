
import torch
import torch.nn as nn
from torch_geometric.nn import HypergraphConv, GCNConv, global_max_pool, global_mean_pool
import sys

sys.path.append('')
from Model.HypertranSynergy.GIN import *
from Model.HypertranSynergy.trans import *
from Model.utils import reset

#number of your drug/ cell line
drug_num = 
cline_num = 


class Initialize(nn.Module):
    def __init__(self, d_dim,  c_dim, o_dim):
        super(Initialize, self).__init__()
        self.JK1 = GIN_drug(2,d_dim)
        self.l1 = nn.Linear(c_dim, 128)
        self.batch_cell1 = nn.BatchNorm1d(128)
        self.l2 = nn.Linear(128, o_dim)
        self.dropout = nn.Dropout(0.7)
        self.act = nn.ReLU()

    def forward(self, drug_feature, drug_adj, ibatch, gexpr_data):

        x_drug = self.JK1(drug_feature, drug_adj, ibatch)
        x_cellline = torch.tanh(self.fc_cell1(gexpr_data))
        x_cellline = self.batch_cell1(x_cellline)
        x_cellline = self.drop_out(x_cellline)
        x_cellline = self.act(self.fc_cell2(x_cellline))
        return x_drug, x_cellline

class CIE(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CIE, self).__init__()
        self.drop_out = nn.Dropout(0.75)
        self.act = nn.LeakyReLU(0.2)

        self.conv3 = HypergraphConv(100, out_channels)
        self.MH = MultiHeadAttention(2, 100, 0.5, 0.5, 1e-5)
        self.FW = FeedForward(100, 256, 0.5, 'sigmoid', 1e-5)

    def forward(self, x, edge):
        x = self.MH(x, None)
        x = self.FW(x)
        x = self.MH(x, None)
        x = self.FW(x)
        x = self.act(self.drop_out(self.conv3(x, edge)))

        return x


class Decoder(torch.nn.Module):
    def __init__(self, in_channels):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // 2)
        self.batch1 = nn.BatchNorm1d(in_channels // 2)
        self.fc2 = nn.Linear(in_channels // 2, in_channels // 4)
        self.batch2 = nn.BatchNorm1d(in_channels // 4)
        self.fc3 = nn.Linear(in_channels // 4, 1)
        self.drop_out = nn.Dropout(0.3)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, graph_embed, druga_id, drugb_id, cellline_id):
        h = torch.cat((graph_embed[druga_id, :], graph_embed[drugb_id, :], graph_embed[cellline_id, :]), 1)
        h = self.act(self.fc1(h))
        h = self.batch1(h)
        h = self.drop_out(h)
        h = self.act(self.fc2(h))
        h = self.batch2(h)
        h = self.drop_out(h)
        h = self.fc3(h)
        return h.squeeze(dim=1)


class Hts(torch.nn.Module):
    def __init__(self, initialize, cie, decoder):
        super(Hts, self).__init__()
        self.initialize = initialize
        self.cie = cie
        self.decoder = decoder
        self.drug_rec_weight = nn.Parameter(torch.rand(512, 512))
        self.cline_rec_weight = nn.Parameter(torch.rand(512, 512))
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.initialize)
        reset(self.cie)
        reset(self.decoder)

    def forward(self, drug_feature, drug_adj, ibatch, gexpr_data, adj, druga_id, drugb_id, cellline_id):
        drug_embed, cellline_embed = self.initialize(drug_feature, drug_adj, ibatch, gexpr_data)
        merge_embed = torch.cat((drug_embed, cellline_embed), 0)
        graph_embed = self.cie(merge_embed, adj)
        drug_emb, cline_emb = graph_embed[:drug_num], graph_embed[drug_num:]       
        res = self.decoder(graph_embed, druga_id, drugb_id, cellline_id)
        return res, rec_drug, rec_cline

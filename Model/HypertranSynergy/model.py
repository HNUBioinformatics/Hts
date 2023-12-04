import torch
import torch.nn as nn
from torch_geometric.nn import HypergraphConv, GCNConv, global_max_pool, global_mean_pool
from torch_geometric.nn import GINConv,GATConv,JumpingKnowledge
from GIN import *
import sys
from trans import *
from Model.HypertranSynergy.GIN import *
from Model.HypertranSynergy.trans import *

sys.path.append('')
from Model.utils import reset

drug_num = 38
cline_num = 32


class Initialize(nn.Module):
    def __init__(self, dim_drug, dim_cellline, output, use_GMP=True):
        super(Initialize, self).__init__()
        self.JK1 = GNN_drug(2,dim_drug)
        self.fc_cell1 = nn.Linear(dim_cellline, 128)
        self.batch_cell1 = nn.BatchNorm1d(128)
        self.fc_cell2 = nn.Linear(128, output)

        self.reset_para()
        self.act = nn.ReLU()

        self.dropout = nn.Dropout(0.3)

    def reset_para(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        return

    def forward(self, drug_feature, drug_adj, ibatch, gexpr_data):
        x_drug = self.JK1(drug_feature, drug_adj, ibatch)
        x_cellline = torch.tanh(self.fc_cell1(gexpr_data))
        x_cellline = self.batch_cell1(x_cellline)
        x_cellline = self.act(self.fc_cell2(x_cellline))
        return x_drug, x_cellline


class CIE(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CIE, self).__init__()

        self.conv3 = HypergraphConv(128, out_channels)
        self.act = nn.ReLU()

        self.MH = MultiHeadAttention(2, 128, 0.5, 0.5, 1e-5)
        self.FW = FeedForward(128, 256, 0.5, 'sigmoid', 1e-5)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x, edge):
        x = self.MH(x, None)
        x = self.FW(x)
        x = self.MH(x, None)
        x = self.FW(x)

        x = self.act(self.dropout(self.conv3(x, edge)))
        return x


class Decoder(torch.nn.Module):
    def __init__(self, in_channels):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // 2)
        self.batch1 = nn.BatchNorm1d(in_channels // 2)
        self.fc2 = nn.Linear(in_channels // 2, in_channels // 4)
        self.batch2 = nn.BatchNorm1d(in_channels // 4)
        self.fc3 = nn.Linear(in_channels // 4, 1)

        self.conv1 = nn.Conv2d(in_channels//4,1,kernel_size=1)

        self.reset_parameters()
        self.drop_out = nn.Dropout(0.3)
       
        self.act = nn.Tanh()
        #self.act = nn.PReLU()
        #self.act = nn.LeakyReLU(negative_slope=0.2)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, graph_embed, druga_id, drugb_id, cellline_id):
        h1 = torch.cat((graph_embed[druga_id, :], graph_embed[drugb_id, :], graph_embed[cellline_id, :]), 1)
        h = self.act(self.fc1(h1))
        h = self.batch1(h)
        h = self.drop_out(h)
        h = self.act(self.fc2(h))
        h = self.batch2(h)
        h = self.drop_out(h)

        h = self.fc3(h)
        #h = self.conv1(h)

        return torch.sigmoid(h.squeeze(dim=1))


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
        rec_drug = torch.sigmoid(torch.mm(torch.mm(drug_emb, self.drug_rec_weight), drug_emb.t()))
        rec_cline = torch.sigmoid(torch.mm(torch.mm(cline_emb, self.cline_rec_weight), cline_emb.t()))
        res = self.decoder(graph_embed, druga_id, drugb_id, cellline_id)
        tsne = self.decoder(graph_embed, druga_id, drugb_id, cellline_id)
        return res, rec_drug, rec_cline, tsne

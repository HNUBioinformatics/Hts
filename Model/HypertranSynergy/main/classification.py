
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

#number of your drug/celll line
drug_num = 
cline_num = 


class Initialize(nn.Module):
    def __init__(self, d_dim, c_dim, o_dim):
        super(Initialize, self).__init__()
        self.JK1 = GIN_drug(2,d_dim)
        self.l1 = nn.Linear(c_dim, 128)
        self.batch_cell1 = nn.BatchNorm1d(128)
        self.l2 = nn.Linear(128, o_dim)

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

    def forward(self, d_ft, d_adj, batch, c_ft):
        d_ft = self.JK1(d_ft, d_adj, batch)
        c_ft = torch.tanh(self.l1(c_ft))
        c_ft = self.batch_cell1(c_ft)
        c_ft = self.act(self.l2(c_ft))
        return d_ft, c_ft


class CIE(torch.nn.Module):
    def __init__(self, num_ft_i, num_ft_o):
        super(CIE, self).__init__()

        self.conv = HypergraphConv(128, num_ft_o)
        self.act = nn.ReLU()

        self.MH = MultiHeadAttention(2, 128, 0.5, 0.5, 1e-5)
        self.FW = FeedForward(128, 256, 0.5, 'sigmoid', 1e-5)

        self.dp = nn.Dropout(0.3)

    def forward(self, x, edge):
        x = self.MH(x, None)
        x = self.FW(x)
        x = self.MH(x, None)
        x = self.FW(x)

        x = self.act(self.dp(self.conv(x, edge)))
        return x


class Decoder(torch.nn.Module):
    def __init__(self, num_feat_i):
        super(Decoder, self).__init__()
        self.l1 = nn.Linear(num_feat_i, num_feat_i // 2)
        self.batch1 = nn.BatchNorm1d(num_feat_i // 2)
        self.l2 = nn.Linear(num_feat_i // 2, num_feat_i // 4)
        self.batch2 = nn.BatchNorm1d(num_feat_i // 4)
        self.l3 = nn.Linear(num_feat_i // 4, 1)

        self.conv1 = nn.Conv2d(num_feat_i//4,1,kernel_size=1)

        self.reset_parameters()
        self.dp = nn.Dropout(0.3)
       
        self.act = nn.Tanh()
        #self.act = nn.PReLU()
        #self.act = nn.LeakyReLU(negative_slope)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, embs, entity1, entity2, entity3):
        h1 = torch.cat((embs[entity1, :], embs[entity2, :], embs[entity3, :]), 1)
        h = self.act(self.l1(h1))
        h = self.batch1(h)
        h = self.dp(h)
        h = self.act(self.l2(h))
        h = self.batch2(h)
        h = self.dp(h)

        h = self.l3(h)
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

    def forward(self, d_ft, d_adj, batch, c_ft, h_adj, entity1, entity2, entity3):
        embs_d, embs_c = self.initialize(d_ft, d_adj, batch, c_ft)
        embs_dc = torch.cat((embs_d, embs_c), 0)
        embs_hg = self.cie(embs_dc, h_adj)
        drug_emb, cline_emb = embs_hg[:drug_num], embs_hg[drug_num:]
        fie_d = torch.sigmoid(torch.mm(torch.mm(drug_emb, self.drug_rec_weight), drug_emb.t()))
        fie_c = torch.sigmoid(torch.mm(torch.mm(cline_emb, self.cline_rec_weight), cline_emb.t()))
        res = self.decoder(embs_hg, entity1, entity2, entity3)
        return res, fie_d, fie_c


import math
import torch.nn.functional as fn
from contranorm import *

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, size, dp1, dp2, ne):
        super(MultiHeadAttention, self).__init__()

        self.heads = heads
        self.h_size = int(size / heads)
        self.t_size = self.heads * self.h_size

        self.q = nn.Linear(size, self.t_size)
        self.k = nn.Linear(size, self.t_size)
        self.v = nn.Linear(size, self.t_size)

        self.dense = nn.Linear(size, size)

        self.dp1 = nn.Dropout(dp1)
        self.dp2 = nn.Dropout(dp2)
        
        self.LayerNorm = nn.LayerNorm(size, eps=ne)
        self.LayerNorm1 = ContraNorm(size, scale=0.0, dual_norm=False, pre_norm=False, temp=1.0, learnable=False,
                                     positive=False, identity=False)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.heads, self.h_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1)

    def forward(self, input_tensor, attention_mask):
        mixed_query_layer = self.q(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        noise = torch.randn_like(attention_scores).to(attention_scores)
        attention_scores = attention_scores + noise * 40


        attention_scores = attention_scores / math.sqrt(self.h_size)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)


        attention_probs = self.dp2(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.t_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.dp1(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        hidden_states = self.LayerNorm1(hidden_states + input_tensor)

        return hidden_states


class FeedForward(nn.Module):

    def __init__(self, size, inner_size, dp, act, ne):
        super(FeedForward, self).__init__()
        self.l1 = nn.Linear(size, inner_size)
        self.act = self.get_hidden_act(act)

        self.l2 = nn.Linear(inner_size, size)

        self.LayerNorm = ContraNorm(size, scale=0.0, dual_norm=False, pre_norm=False, temp=1.0, learnable=False,
                                    positive=False, identity=False)
        self.LayerNorm1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

        self.dp = nn.Dropout(dp)

    def get_hidden_act(self, act):
        ACT2FN = {
            "gelu": self.gelu,
            "relu": fn.relu,
            "swish": self.swish,
            "tanh": torch.tanh,
            "leakyrelu": nn.LeakyReLU(negative_slope=0.2),
            "sigmoid": nn.ELU(alpha=0.1, inplace=False),
        }
        return ACT2FN[act]

    def gelu(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def swish(self, x):
        return x * torch.sigmoid(x)

    def forward(self, input):
        tr_t = self.l1(input)
        tr_t = self.act(tr_t)

        tr_t = self.l2(tr_t)
        tr_t = self.dp(tr_t)
        tr_t = self.LayerNorm1(tr_t + input)
        tr_t = self.LayerNorm(tr_t + input)

        return tr_t

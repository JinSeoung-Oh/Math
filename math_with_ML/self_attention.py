## Need more understand

## dot-product attention

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c)
    sum_exp = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y

class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.input_dim = input_dim
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=2)
        
    def forward(self, x):
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / (self.input_dim ** 0.5)  #https://pytorch.org/docs/stable/generated/torch.bmm.html
        attention = self.softmax(scores)
        weighted = torch.bmm(attention, values)
        return weighted


## Scaled dot-product attention
# https://github.com/JinSeoung-Oh/Etc/blob/main/Transformers_with_SDPA.py
# https://paperswithcode.com/method/scaled
def ScaledDotProductAttention(query, key, value, mask):
    matmul_qk = np.matmul(query, np.transpose(key))
    dk = np.shape(key)[-1]
    sacled_atten = matmul_qk/np.sqrt(dk)

    if mask is not None:
        scaled_attention += (mask * -1e9)
    attention_weights = nn.Softmax(dim=-1)
    sacled_dot_product_attention = np.matmul(attention_weights, value)

    return sacled_dot_product_attention

## Multi-head attention
# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/fec78a687210851f055f792d45300d27cc60ae41/transformer/SubLayers.py#L9
class MultiHeadAttention(nn.Module):
    from transformer.Modules import ScaledDotProductAttention
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.attentipn = ScaledDotProductAttention(temperature=d_k ** 0.5)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        residual = q
        
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q,k,v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)
        q,attn = self.attention(q,k,v,mask=mask)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual
        q = self.layer_norm(q)

        return q, attn
        
# Local attention

# Additive attention

# Cosine attention

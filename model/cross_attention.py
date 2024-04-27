import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        
        # linear transformations for query, key, and value
        self.Q = nn.Linear(query_dim, query_dim)
        self.K = nn.Linear(key_dim, key_dim)
        self.V = nn.Linear(value_dim, value_dim)
        
        # scale factor for dot product attention
        self.scale_factor = 1.0 / (query_dim ** 0.5)
        
        # linear transformation for output
        self.out_linear = nn.Linear(value_dim, query_dim)
        
    def forward(self, query, key, value, mask=None):
        # linear transformation for query, key, and value
        query = self.Q(query)
        key = self.K(key)
        value = self.V(value)
        
        # split the query, key, and value into multiple heads
        query = self.split_heads(query)
        key = self.split_heads(key)
        value = self.split_heads(value)
        
        # compute scaled dot product attention
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale_factor
        if mask is not None:
            attn_scores.masked_fill_(mask == 0, float('-inf'))  # apply mask
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, value)
        
        # concatenate attention outputs from all heads and linearly transform
        attn_output = attn_output.transpose(1, 2).contiguous().view(query.size(0), -1, value.size(-1))

        attn_output = attn_output.view(query.size(0), -1)
        attn_output = self.out_linear(attn_output)
        
        return attn_output
    
    def split_heads(self, x):
        batch_size, seq_len, dim = x.size()
        head_dim = dim // self.num_heads
        x = x.view(batch_size, seq_len, self.num_heads, head_dim)
        return x.transpose(1, 2)
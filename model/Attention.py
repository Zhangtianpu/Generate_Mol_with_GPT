import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, length, input_dim, head_dim, num_head, drop_rate=0.5, bias=False):
        super(SelfAttention, self).__init__()
        self.input_dim = input_dim
        self.head_dim = head_dim
        self.num_head = num_head
        self.context_length = length

        self.q_layer = nn.Linear(in_features=input_dim,
                                 out_features=self.head_dim * self.num_head,
                                 bias=bias)
        self.k_layer = nn.Linear(in_features=input_dim,
                                 out_features=self.head_dim * self.num_head,
                                 bias=bias)
        self.v_layer = nn.Linear(in_features=input_dim,
                                 out_features=self.head_dim * self.num_head,
                                 bias=bias)
        self.register_buffer('mask',torch.triu(torch.ones(self.context_length,self.context_length),diagonal=1))

        # self.layernorm = nn.LayerNorm(self.head_dim * self.num_head)
        self.concat_layer = nn.Linear(
            in_features=self.head_dim * self.num_head,
            out_features=self.head_dim * self.num_head, bias=bias)
        self.drop_layer = nn.Dropout(p=drop_rate)

    def forward(self, input):
        # input:[batch,length,embedding_dim]
        batch, length, _ = input.size()
        # Q:[batch,num_head,length,dim]
        Q = self.q_layer(input).reshape(batch, length, self.num_head, self.head_dim).permute(0, 2, 1, 3)
        # K_T:[batch,num_head,dim,length]
        K_T = self.k_layer(input).reshape(batch, length, self.num_head, self.head_dim).permute(0, 2, 3, 1)
        # V:[batch,num_head,length,dim]
        V = self.v_layer(input).reshape(batch, length, self.num_head, self.head_dim).permute(0, 2, 1, 3)

        # att_score:[batch,num_head,length,length]
        att_score = torch.matmul(Q, K_T) / np.sqrt(self.head_dim)
        mask_bool=self.mask.bool()[:length,:length]

        att_score = att_score.masked_fill_(mask_bool, torch.tensor(-float("inf")))

        # att_score = F.softmax(att_score/V.shape[-1]**0.5, dim=-1)
        att_score = F.softmax(att_score, dim=-1)

        att_score=self.drop_layer(att_score)
        # output:[batch,length,num_head,dim]
        output = torch.matmul(att_score, V).transpose(1,2)

        # output:[batch,length,num_head*dim]
        # output = torch.cat([output[:,:, i, :] for i in range(self.num_head)], dim=-1)
        output=output.contiguous().view(batch,length,self.num_head*self.head_dim)
        output = self.concat_layer(output)

        # output = self.layernorm(output)

        return output

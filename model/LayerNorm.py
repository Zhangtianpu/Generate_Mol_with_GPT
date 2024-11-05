import torch
import torch.nn as nn

class Layer_norm(nn.Module):
    def __init__(self,emb_dim):
        super().__init__()
        self.eps=1e-5
        self.scale=nn.Parameter(torch.ones(emb_dim))
        self.shift=nn.Parameter(torch.zeros(emb_dim))

    def forward(self,inputs):
        mean=inputs.mean(dim=-1,keepdim=True)
        var=inputs.var(dim=-1,keepdim=True,unbiased=False)
        norm_inputs=(inputs-mean)/torch.sqrt(var+self.eps)
        return self.scale*norm_inputs+self.shift

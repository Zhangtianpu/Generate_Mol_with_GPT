import torch
import torch.nn as nn
from model import Attention,FeedFroward_module
from model.LayerNorm import Layer_norm


class transfomer_block(nn.Module):
    def __init__(self, context_length, wordEmbedding_dim, head_dim, num_head, drop_rate, bias=True):
        super(transfomer_block, self).__init__()
        self.attention_layer = Attention.SelfAttention(length=context_length,
                                                       input_dim=wordEmbedding_dim,
                                                       head_dim=head_dim,
                                                       num_head=num_head,
                                                       drop_rate=drop_rate,
                                                       bias=bias)
        self.forward_layer=FeedFroward_module.feedforward(wordEmbedding_dim)
        self.norm_layer1=Layer_norm(wordEmbedding_dim)
        self.norm_layer2=Layer_norm(wordEmbedding_dim)
        self.drop_shortcut=nn.Dropout(drop_rate)



    def forward(self, inputs):

        shortcut=inputs
        x=self.norm_layer1(inputs)
        x=self.attention_layer(x)
        x=self.drop_shortcut(x)
        x=shortcut+x

        shortcut=x
        x=self.norm_layer2(x)
        x=self.forward_layer(x)
        x=self.drop_shortcut(x)
        x=x+shortcut

        return x

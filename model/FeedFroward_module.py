import numpy as np
import torch
import torch.nn as nn


class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, inputs):
        return 0.5 * inputs * (
                1 + torch.tanh(torch.sqrt(torch.tensor(2 / np.pi)) * (inputs + 0.04475 * torch.pow(inputs, 3))))


class feedforward(nn.Module):
    def __init__(self, wordEmbedding_dim):
        super(feedforward, self).__init__()

        self.forward_layer = nn.Sequential(nn.Linear(in_features=wordEmbedding_dim, out_features=4 * wordEmbedding_dim),
                                           GELU(),
                                           nn.Linear(in_features=4 * wordEmbedding_dim, out_features=wordEmbedding_dim))
        # self.forward_layer = nn.Sequential(nn.Conv1d(in_channels=wordEmbedding_dim,
        #                                              out_channels=4 * wordEmbedding_dim,
        #                                              kernel_size=1,
        #                                              stride=1,
        #                                              padding=0, bias=True),
        #                                    GELU(),
        #                                    nn.Conv1d(in_channels=4 * wordEmbedding_dim,
        #                                              out_channels=wordEmbedding_dim,
        #                                              kernel_size=1,
        #                                              stride=1,
        #                                              padding=0, bias=True)
        #                                    )

    def forward(self, inputs):
        outputs = self.forward_layer(inputs)
        return outputs

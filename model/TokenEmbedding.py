import torch
import torch.nn as nn


class PositionEmbedding(nn.Module):
    def __init__(self, context_length, positionEmbedding_dim,device):
        super(PositionEmbedding,self).__init__()
        self.context_length=context_length
        self.device=device
        self.positionEmbed_layer = nn.Embedding(context_length, positionEmbedding_dim)

    def forward(self,inputs):
        batch,tokens=inputs.size()
        positionInfo = self.positionEmbed_layer(torch.arange(tokens).to(self.device))
        return positionInfo


class WordEmbedding(nn.Module):
    def __init__(self, vocab_len, wordEmbedding_dim):
        super(WordEmbedding,self).__init__()

        self.wordEmbed_layer = nn.Embedding(vocab_len, wordEmbedding_dim)

    def forward(self, inputs):
        embedded_inputs = self.wordEmbed_layer(inputs)
        return embedded_inputs

class Embedding(nn.Module):
    def __init__(self,vocab_len,context_length,wordEmbedding_dim,positionEmbedding_dim,device):
        super(Embedding,self).__init__()
        assert wordEmbedding_dim == positionEmbedding_dim
        self.wordEmbeddingLayer=WordEmbedding(vocab_len,wordEmbedding_dim)
        self.positionEmbeddingLayer=PositionEmbedding(context_length,positionEmbedding_dim,device)

    def forward(self,inputs):
        #[batch,words,embedding_dim]
        word_embedding_re=self.wordEmbeddingLayer(inputs)
        position_embedding_re=self.positionEmbeddingLayer(inputs)
        embedding_result=word_embedding_re+position_embedding_re
        return embedding_result

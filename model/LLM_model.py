import torch
import torch.nn as nn
from model import TokenEmbedding, TransformerBlock
from model.LayerNorm import Layer_norm


class llm_model(nn.Module):
    def __init__(self, vocab_len, context_length, wordEmbedding_dim, positionEmbedding_dim, n_layers, head_dim,
                 num_head, device, drop_rate=0.5):
        super(llm_model, self).__init__()

        self.token_layer = TokenEmbedding.Embedding(vocab_len, context_length, wordEmbedding_dim, positionEmbedding_dim,
                                                    device)

        self.transfomer_layres = nn.Sequential(
            *[
                TransformerBlock.transfomer_block(context_length, wordEmbedding_dim, head_dim,num_head, drop_rate, bias=True)
                for i in range(n_layers)
            ])

        self.drop_layer = nn.Dropout(p=drop_rate)
        self.norm_layer=Layer_norm(wordEmbedding_dim)
        self.final_layer = nn.Linear(in_features=wordEmbedding_dim, out_features=vocab_len,bias=False)
        # self.final_layer=nn.Embedding(num_embeddings=wordEmbedding_dim,embedding_dim=vocab_len)

    def forward(self, inputs):
        #input:[batch,tokens]
        #embedded_inputs:[batch,tokens,embedding_dim]
        embedded_inputs = self.token_layer(inputs)
        drop_inputs = self.drop_layer(embedded_inputs)
        #trans_output:[batch,tokens,embedding_dim]
        trans_output = self.transfomer_layres(drop_inputs)
        trans_output=self.norm_layer(trans_output)
        #output:[batch,tokens,vocab]
        output = self.final_layer(trans_output)

        return output

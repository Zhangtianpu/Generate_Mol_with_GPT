import re
import os

import Utils
import torch

class SmilesTokenzier:
    def __init__(self,vocab):
        self.str_to_int=vocab
        self.int_to_str={id:text for text,id in vocab.items()}

    def encode(self,single_smile,return_type='pt'):

        if single_smile  in['<|UNK|>','<|endoftext|>']:
            return [self.str_to_int[single_smile]]
        pattern = "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        regex = re.compile(pattern)
        tokens=regex.findall(single_smile.strip())

        index=[]

        for token in tokens:
            if token in self.str_to_int:
                index.append(self.str_to_int[token])
            else:
                index.append(self.str_to_int['<|UNK|>'])
        if return_type =='pt':
            index=torch.LongTensor(index)
        return index

    def decode(self,tokens):
        if type(tokens) == torch.Tensor:
            tokens=tokens.numpy()
        word_list=[]
        for token in tokens:
            word_list.append(self.int_to_str[token])

        return "".join(word_list)



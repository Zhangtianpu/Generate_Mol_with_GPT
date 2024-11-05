import re
import os
from collections import Counter
import Utils

if __name__ == '__main__':
    # zinc_standard_agent|MOSES|GUACAMOL
    MOSES_folder = f'./raw_data/MOSES/raw/'
    data_filename = 'smiles.csv'
    MOSES_path = os.path.join(MOSES_folder, data_filename)

    GUACAMOL_folder = f'./raw_data/GUACAMOL/raw/'
    GUACAMOL_path = os.path.join(GUACAMOL_folder, data_filename)

    smiles = []
    for data_path in [MOSES_path, GUACAMOL_path]:
        with open(data_path, 'r') as f:
            lines = f.readlines()
            data_name = data_path.split('/')[-3]
            if data_name == 'MOSES':
                # the first line is title
                sub_smiles = [line.split(',')[0].strip() for line in lines[1:]]
            if data_name == 'zinc_standard_agent':
                sub_smiles = [line.strip() for line in lines]
            if data_name == 'GUACAMOL':
                sub_smiles = [line.strip() for line in lines]

        smiles.extend(sub_smiles)

    pattern = "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)

    all_tokens = []
    for smile in smiles:
        all_tokens.extend(regex.findall(smile.strip()))
    token_counter = Counter(all_tokens)

    vocab = {token: idx for idx, (token, _) in enumerate(token_counter.items(), start=0)}

    # add unknown token
    vocab.update({'UNK': len(vocab), '<|endoftext|>': len(vocab) + 1})

    vocab_save_folder = "./vocab"
    vocab_name = 'vocab.json'
    Utils.save_vocab(vocab, vocab_save_folder, vocab_name)

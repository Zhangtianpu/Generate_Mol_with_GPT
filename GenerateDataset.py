import os
import numpy as np
import Utils
from SmilesTokenizer import SmilesTokenzier
import torch



def split_train_val_dataset(smiles, train_ratio=0.9, random_seed=123):
    length = len(smiles)
    np.random.seed(random_seed)
    random_index = np.random.permutation(length)

    train_end = int(length * train_ratio)
    train_index = random_index[:train_end]
    val_index = random_index[train_end:]

    train_data = smiles[train_index]
    val_data = smiles[val_index]

    return train_data, val_data


def generate_dataset(data, tokenizer, max_length, stride, padding='<|endoftext|>'):
    input_ids = []
    target_ids = []
    # data = Utils.load_data(data_folder, train_or_val)

    token_ids = []
    for smile in data:
        token_ids.extend(tokenizer.encode(smile))
        token_ids.extend(tokenizer.encode(padding))

    for i in range(0, len(token_ids) - max_length, stride):
        input_chunk = token_ids[i:i + max_length]
        target_chunk = token_ids[i + 1:i + max_length + 1]

        input_ids.append(input_chunk)
        target_ids.append(target_chunk)

    return input_ids, target_ids

def generate_dataset_1(data, tokenizer, padding='<|endoftext|>'):
    input_ids = []
    target_ids = []
    # data = Utils.load_data(data_folder, train_or_val)
    encoded_pad=tokenizer.encode(padding)
    token_ids = []
    max_length=0
    for smile in data:
        encoded_smile=tokenizer.encode(smile)
        token_ids.append(encoded_smile)
        if max_length<len(encoded_smile):
            max_length=len(encoded_smile)




    for ids in token_ids:
        input=ids
        padding_length=max_length-len(input)
        padding=torch.IntTensor(encoded_pad*padding_length)
        end_padding=torch.IntTensor(encoded_pad)
        input=torch.cat([input,padding,end_padding],dim=0)

        input_ids.append(input[:-1])
        target_ids.append(input[1:])

    return input_ids, target_ids


if __name__ == '__main__':
    train_ratio = 0.9
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
    train_data, val_data = split_train_val_dataset(np.array(smiles), train_ratio=train_ratio)
    print(f"train_data:{train_data.shape}")
    print(f"val_data:{val_data.shape}")

    vocab_folder = './vocab'
    vocab_name = 'vocab.json'
    vocab = Utils.load_vocab(folder=vocab_folder, name=vocab_name)
    tokenizer = SmilesTokenzier(vocab)

    # train_input, train_target = generate_dataset(train_data, tokenizer, max_length=256, stride=256)
    # val_input, val_target = generate_dataset(val_data, tokenizer, max_length=256, stride=256)
    train_input,train_target=generate_dataset_1(train_data,tokenizer)
    val_input,val_target=generate_dataset_1(val_data,tokenizer)
    save_folder = './processed_dataset'
    train_data_name = 'train_data.h5'
    val_data_name = 'val_data.h5'
    # Utils.save_data(train_data,save_folder,train_data_name)
    # Utils.save_data(val_data,save_folder,val_data_name)
    Utils.save_h5(data_dict={"input": train_input, "target": train_target}, save_folder=save_folder,
                  save_name=train_data_name)
    Utils.save_h5(data_dict={"input": val_input, "target": val_target}, save_folder=save_folder,
                  save_name=val_data_name)

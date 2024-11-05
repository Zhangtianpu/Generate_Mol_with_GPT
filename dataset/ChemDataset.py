import torch
import os
from torch.utils.data import Dataset, DataLoader
from SmilesTokenizer import SmilesTokenzier
from torch.utils.data.distributed import DistributedSampler
import Utils


class ChemDataset(Dataset):
    def __init__(self, data_folder, train_or_val='train_data.pkl'):
        super(ChemDataset, self).__init__()

        data = Utils.load_h5(data_folder, train_or_val)
        self.input_ids = data['input']
        self.target_ids = data['target']

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, item):
        return torch.LongTensor(self.input_ids[item]), torch.LongTensor(self.target_ids[item])


def chemDatasetLoader(data_folder,
                      batch_size,
                      shuffle=False,
                      train_or_val='train_data.pkl',
                      drop_last=True):
    dataset = ChemDataset(data_folder, train_or_val=train_or_val)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    return dataloader


def multi_gpu_chemDatasetLoader(data_folder,
                                batch_size,
                                shuffle=False,
                                train_or_val='train_data.pkl',
                                drop_last=True):
    dataset = ChemDataset(data_folder, train_or_val=train_or_val)
    dist_sampler = DistributedSampler(dataset, shuffle=shuffle)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            sampler=dist_sampler,
                            drop_last=drop_last)
    return dataloader


if __name__ == '__main__':
    data_folder = '/soft/home/zhaojw.bjhy/SHARE_TO_ALL/To_TianPu/generate_mol_with_gpt/dataset'
    train_data_name = 'train_data.pkl'
    val_data_name = 'val_data.pkl'

    vocab = Utils.load_vocab('../vocab', 'vocab.json')
    tokenizer = SmilesTokenzier(vocab)
    trainLoader = chemDatasetLoader(data_folder=data_folder,
                                    batch_size=8,
                                    shuffle=False,
                                    train_or_val=train_data_name)
    for input, target in trainLoader:
        print(input, target)
        break

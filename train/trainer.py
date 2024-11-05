import Utils
from SmilesTokenizer import SmilesTokenzier
from dataset.ChemDataset import multi_gpu_chemDatasetLoader
import torch
from model.LLM_model import llm_model
from TrainModel import train_model_smiple
import torch.distributed as dist
import torch.multiprocessing as mp
import argparse
import torch.nn as nn
import numpy as np
import yaml



def setup_MGPU(rank, world_size):
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def run_train(rank, args):
    args.rank = rank
    setup_MGPU(args.rank, args.num_gpus)
    torch.cuda.set_device(args.rank)

    vocab = Utils.load_vocab(folder=args.vocab_folder, name=args.vocab_name)

    tokenizer = SmilesTokenzier(vocab)

    trainLoader = multi_gpu_chemDatasetLoader(data_folder=args.processed_data_folder,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              train_or_val=args.train_data_name)

    valLoader = multi_gpu_chemDatasetLoader(data_folder=args.processed_data_folder,
                                            batch_size=args.batch_size,
                                            shuffle=False,
                                            train_or_val=args.val_data_name)

    BASE_CONFIG = {
        "vocab_size": len(vocab),
        "context_length": args.context_length,
        "drop_rate": 0.0,
        "bias": True
    }
    model_config = {
        "gpt2_124M": {'emb_dim': 768, "n_layers": 12, "n_heads": 12, "head_dim": 64},
        "gpt2_335M": {'emb_dim': 1024, 'n_layers': 24, "n_heads": 16, "head_dim": 64},
        "gpt2_774M": {"emb_dim": 1280, 'n_layers': 36, "n_heads": 20, "head_dim": 64},
        "gpt2_1.5B": {"emb_dim": 1600, 'n_layers': 48, 'n_heads': 25, 'head_dim': 64},
        "custom": {"emb_dim": 256, 'n_layers': 8, 'n_heads': 8, 'head_dim': 32}
    }

    CHOOSE_MODEL = args.choose_model
    if CHOOSE_MODEL=='custom':
        yaml_path='../model_params/custom_params.yml'
        with open(yaml_path, 'r') as f:
            yaml_config=yaml.safe_load(f)
        custom_model_parameter=yaml_config['model_parameter']
        BASE_CONFIG.update(custom_model_parameter)
    else:
        BASE_CONFIG.update(model_config[CHOOSE_MODEL])

    llm = llm_model(vocab_len=BASE_CONFIG['vocab_size'],
                    context_length=BASE_CONFIG['context_length'],
                    wordEmbedding_dim=BASE_CONFIG['emb_dim'],
                    positionEmbedding_dim=BASE_CONFIG['emb_dim'],
                    n_layers=BASE_CONFIG['n_layers'],
                    head_dim=BASE_CONFIG['head_dim'], num_head=BASE_CONFIG['n_heads'],
                    device=args.rank, drop_rate=BASE_CONFIG['drop_rate']).cuda()
    llm = nn.SyncBatchNorm.convert_sync_batchnorm(llm)

    total_params = sum(p.numel() for p in llm.parameters())
    print(total_params)
    total_size_bytes = total_params * 4
    total_size_mb = total_size_bytes / (1024 * 1024)
    print("Total size of model:%.2f MB" % total_size_mb)

    # print("Check Train Loader")
    # for x, y in trainLoader:
    #     print(x.shape, y.shape)
    torch.manual_seed(123)

    optimizer = torch.optim.AdamW(llm.parameters(), lr=0.0004, weight_decay=0.1)

    num_epochs = 10

    train_loss, val_loss, token_seen = train_model_smiple(llm, trainLoader, valLoader, optimizer, args.rank,
                                                          num_epochs, eval_freq=1000, eval_iter=500,
                                                          start_context='c', tokenizer=tokenizer,
                                                          save_model_path=args.save_model_path)
    cleanup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="generate mol with llm")
    """
    parameters 
    """

    parser.add_argument('--train_data_name', type=str, default='train_data.h5')
    parser.add_argument('--val_data_name', type=str, default='val_data.h5')
    parser.add_argument('--context_length', type=int, default=147)
    parser.add_argument('--vocab_folder', type=str, default='../vocab')
    parser.add_argument('--vocab_name', type=str, default='vocab.json')

    """
    training parameters
    """
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--num_gpus', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--save_model_path', type=str,
                        default='../trained_model/mol_llm.pkl')

    """
    model's parameters
    """
    parser.add_argument('--choose_model', type=str, default='custom', help="gpt2_124M|gpt2_335M|gpt2_774M|gpt2_1.5B|custom")
    # data_folder = '/soft/home/zhaojw.bjhy/SHARE_TO_ALL/To_TianPu/generate_mol_with_gpt/dataset'
    # train_data_name = 'train_data.pkl'
    # val_data_name = 'val_data.pkl'
    args = parser.parse_args()

    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    if args.num_gpus == 1:
        run_train(rank=0, args=args)
    else:
        torch.multiprocessing.set_start_method('spawn')
        mp.spawn(run_train, nprocs=args.num_gpus, args=(args,))

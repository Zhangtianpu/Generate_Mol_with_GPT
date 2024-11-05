import os
import pickle
import json
from torch.nn import functional as F
import torch
import torch.nn as nn
import h5py
from rdkit import Chem
from rdkit.Chem import Draw
def save_vocab(vocab,save_folder,save_name):
    save_path=os.path.join(save_folder,save_name)
    with open(save_path,'w') as f:
        json.dump(vocab,f)

def load_vocab(folder,name):
    save_path=os.path.join(folder,name)
    with open(save_path,'r') as f:
        vocab=json.load(f)
    return vocab

def save_data(data,save_folder,save_name):
    if os.path.exists(save_folder) is False:
        os.makedirs(save_folder)
    save_path=os.path.join(save_folder,save_name)
    with open(save_path,'wb') as f:
        pickle.dump(data,f)

def save_h5(data_dict,save_folder,save_name):
    """
    :param data_dict:{"input":[],"target":[]}
    :param save_folder:
    :param save_name:
    :return:
    """
    save_path=os.path.join(save_folder,save_name)

    with h5py.File(save_path,'w') as hdf:
        # hdf.attrs.create(name='data',data=data.encode('ascii'),dtype=dtype)
        hdf.create_dataset('input',data=data_dict['input'])
        hdf.create_dataset('target',data=data_dict['target'])
    print("save h5 success fully")

def load_h5(folder,name):
    file_path=os.path.join(folder,name)
    data=h5py.File(file_path,'r')
    return data


def load_data(save_folder,save_name):
    save_path=os.path.join(save_folder,save_name)
    with open(save_path,'rb') as f:
        data=pickle.load(f)
    return data

def save_model(model,save_path):
    torch.save(model.state_dict(),save_path)
    print('save successfully')

def load_model(model,save_path):
    model_parameters=torch.load(save_path)
    model.load_state_dict(model_parameters)

def calc_loss_batch(input,target,model,device):
    input=input.to(device)
    target=target.to(device)
    output=model(input)
    loss=F.cross_entropy(output.flatten(0,1),target.flatten())
    return loss

def calc_loss_loader(data_loader,model,device,num_batch=None):
    total_loss=0
    if num_batch is None:
        num_batch=len(data_loader)
    else:
        num_batch=min(num_batch,len(data_loader))

    for i, (input,target) in enumerate(data_loader):
        if i < num_batch:
            batch_loss=calc_loss_batch(input,target,model,device)
            total_loss+=batch_loss.item()
        else:
            break
    return total_loss/num_batch

def text_to_ids(smile,tokenizer):
    tokens=tokenizer.encode(smile)
    encode_text_tensor=torch.tensor(tokens).unsqueeze(0)
    return encode_text_tensor

def ids_to_text(ids,tokenizer):
    """
    :param ids: Tensor [batch,tokens]
    :param tokenizer:
    :return: list,[[text1],[text2]...]
    """
    total_text=[]
    #ids:[batch,tokens]
    batch,_=ids.shape
    for i in range(batch):
        text=tokenizer.decode(ids[i].tolist())
        total_text.append(text)
    return total_text

def idslist_to_text(idslist,tokenizer):
    """
    :param idslist: list[Tensor]
    :param tokenizer:
    :return: list,[[text1],[text2]...]
    """
    total_text=[]
    #ids:[batch,tokens]

    for ids in idslist:
        text=ids_to_text(ids,tokenizer)[0]
        total_text.append(text)
    return total_text


def generate_based_GAN(MolGAN,idx,max_new_tokens,context_size,temperature=0.0,top_K=None,eos_id=None):
    MolGAN.eval()
    for _ in range(max_new_tokens):
        idx_cond=idx[:,-context_size:]
        # tensor([[-55.1197, -48.4754, -60.2548,  ..., -61.0108, -59.1533, -53.3415]],
        #        device='cuda:1')
        with torch.no_grad():
            logits=MolGAN(idx_cond)
        logits = logits[:, -1, :]

        if top_K is not None:
            #topk_logits:[batch,top_K],topK_pos:[batch,top_K]
            topk_logits,topk_pos=torch.topk(logits,top_K)
            logits=torch.where(condition=logits<topk_logits[:,-1],input=torch.tensor(float('-inf')).to(logits.device),other=logits)

        if temperature>0.0:
            logits=logits/temperature
            probs=torch.softmax(logits,dim=-1)
            idx_next=torch.multinomial(input=probs,num_samples=1)
        else:
            #idx_next:[batch,1]
            idx_next=torch.argmax(logits,dim=-1,keepdim=True)
        if idx_next==eos_id:
            break
        #idx:[batch,tokens+1]
        idx=torch.cat([idx,idx_next],dim=1)
    return idx


def generate_based_GAN_random_noise(MolGAN,base_config,max_new_tokens):
    MolGAN.eval()
    fake_input=torch.randn(size=(1,max_new_tokens,base_config['emb_dim'])).cuda()
    logits=MolGAN(fake_input)
    idx=torch.argmax(logits,dim=-1)

    return idx


def generate(model,idx,max_new_tokens,context_size,temperature=0.0,top_K=None,eos_id=None):
    #tensor([[  32, 3797,  739,  257,  262]], device='cuda:1')
    # tensor([[32, 3797, 739, 257, 262, 685]], device='cuda:1')
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond=idx[:,-context_size:]
        # tensor([[-55.1197, -48.4754, -60.2548,  ..., -61.0108, -59.1533, -53.3415]],
        #        device='cuda:1')
        with torch.no_grad():
            logits=model(idx_cond)
        logits=logits[:,-1,:]

        if top_K is not None:
            #topk_logits:[batch,top_K],topK_pos:[batch,top_K]
            topk_logits,topk_pos=torch.topk(logits,top_K)
            logits=torch.where(condition=logits<topk_logits[:,-1],input=torch.tensor(float('-inf')).to(logits.device),other=logits)

        if temperature>0.0:
            logits=logits/temperature
            probs=torch.softmax(logits,dim=-1)
            idx_next=torch.multinomial(input=probs,num_samples=1)
        else:
            #idx_next:[batch,1]
            idx_next=torch.argmax(logits,dim=-1,keepdim=True)
        if idx_next==eos_id:
            break
        #idx:[batch,tokens+1]
        idx=torch.cat([idx,idx_next],dim=1)
    return idx
def generate_for_numbers(numbers:int,model,idx,max_new_tokens,context_size,temperature=0.0,top_K=None,eos_id=None):
    """
    :param numbers:
    :param model:
    :param idx:
    :param max_new_tokens:
    :param context_size:
    :param temperature:
    :param top_K:
    :param eos_id:
    :return: list[torch.tensor] [tokens1,tokens2...]
    """
    total_generated_ids=[]
    for i in range(numbers):
        generated_ids=generate(model,idx,max_new_tokens,context_size,temperature,top_K,eos_id)
        total_generated_ids.append(generated_ids)

    return total_generated_ids

def generate_for_numbers_with_GAN(numbers:int,model,base_config,max_new_tokens):
    """
    :param numbers:
    :param model:
    :param base_config:
    :param max_new_tokens:
    :return: list[torch.tensor] [tokens1,tokens2...]
    """
    total_generated_ids=[]
    for i in range(numbers):
        generated_ids=generate_based_GAN_random_noise(model,base_config,max_new_tokens)
        total_generated_ids.append(generated_ids)

    return total_generated_ids

def show_single_from_smile(single_smile):
    mol=Chem.MolFromSmiles(single_smile)
    Draw.ShowMol(mol)

def show_from_smiles(smiles):
    type_none=0
    validate_mol=0
    for smile in smiles:
        smile = smile.replace('<|endoftext|>', '')
        mol=Chem.MolFromSmiles(smile)
        if mol is None:
            type_none+=1
        else:
            print(smile)
            validate_mol+=1
    print(f"the number of validate smiles has {validate_mol}")
    print(f"the number of invalidate smiles has {type_none}")
    print(f"the rate of validation of smiles is {validate_mol/len(smiles)}")
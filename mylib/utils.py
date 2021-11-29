import torch
import random
import numpy as np
import os
from os import X_OK, path as osp
import csv 
import argparse

from torch.optim import optimizer

class StepCounter:
    def __init__(self, max_step, cur_step=0):
        self.max_step = max_step
        self.cur_step = cur_step
  
    def step(self, stride=1):
        self.cur_step += stride
    
    def reset(self):
        self.max_step = 0
        self.cur_step = 0



class MyLRScheduler:
    def __init__(self, sc:StepCounter, decay_fn, optimizer, lr_min, lr_max, warmup=0.1):
        self.decay_fn = decay_fn
        self.optimizer = optimizer
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.warmup = warmup
        self.sc = sc

    def step(self, stride=1):
        self.sc.step(stride=stride)
        self.decay_fn(self.optimizer, self.sc.cur_step, self.sc.max_step, self.lr_min, self.lr_max, self.warmup)
    
    def reset(self):
        self.sc.reset()
        self.decay_fn(self.optimizer, self.sc.cur_step, self.sc.max_step, self.lr_min, self.lr_max, self.warmup)

def get_device():
    return torch.device('cuda')

def set_seed(seed=20211129):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def makedirs(path):
    if not osp.exists(path):
        os.makedirs(path)

def load_csv( path, delimiter='\t',include_header=True) : 
    """
    load datas from csv_file, 
    return: header(list),  datas(list of list)
    """
    header = None
    data  = list()
    with open( path, encoding='utf-8') as csvfile:
        reader = csv.reader( csvfile, delimiter=delimiter ) 
        for row in reader : 
            if header is None : 
                header = row
                if not include_header:
                    continue
            data.append( row ) 
    return header, data

def write_csv(data, location): 
    """
    write data to location, in csv format
    data: list of list
    """
    with open( location, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer( csvfile ) 
        writer.writerows( data) 
    print( "Wrote {}".format( location ) ) 
    return

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, help='best corr_spearman on dev set, pytorch_model checkpoint')
    parser.add_argument('--train_file', type=str, choices=['combine', 'train'], default='combine', help='choose the datasets the model trains on')
    parser.add_argument('--loss', type=str, choices=['opt2', 'opt3', 'opt3s'], help='self-guided contrastive loss func')
    parser.add_argument('--regloss', type=str, choices=['param', 'hidden'], default='param', help='regularization term')
    parser.add_argument('--lamb', type=float, default=0.1, help='weight of regularization term')
    parser.add_argument('--temp', type=float, help='cosine similarity scaling factor ')
    parser.add_argument('--sampler', type=str, choices=['uniform', 'weighted'], default=None)
    parser.add_argument('--sample_weight', type=float, nargs='*')
    parser.add_argument('--do_train', action='store_true', default=False, help='whether to train model')
    parser.add_argument('--do_dev', action='store_true', default=False, help='whether to evaluate model after each training epoch on dev set')
    parser.add_argument('--do_test', action='store_true', default=False, help='whether to evaluate model after whole training was done, loading the checkpoint \
        which has best dev-set performance')
    parser.add_argument('--lr', type=float)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--epoch_num', type=int)
    parser.add_argument('--max_sent_len', type=int, default=128)
    parser.add_argument('--seed', type=int, nargs='+')
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--warmup', type=float, default=0.1)
    global args
    args = parser.parse_args()
    return args

def load_sentence_pair(file_path):
    """
    load sentence pair
    return list of [sent1, sent2]
    """
    datas = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        if len(lines[0].strip().split('\t')) == 2:
            return [
                line.strip().split('\t') for line in lines
            ]
        else:
            return [
                    line.strip().split('\t') for line in [
                    '\t'.join(line.strip().split('\t')[5:7]) for line in lines
                ]
            ]
            
def load_single_sentence(file_path):
    """
    load single sentence
    return list of sentence
    """
    datas = load_sentence_pair(file_path)
    new_datas = []
    for data in datas:
        new_datas.extend(data)
    return new_datas

def load_eval_datas(file_path):
    """
    load sentence pair with label
    return list of [label, sent1, sent2]
    """
    datas = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split('\t')
            datas.append(line[4:7])
    return datas

def infer_hidden_size(model_name):
    """
    infer hidden_size and layers_num from model_name or model checkpoint
    return hiddensize, layernum
    """
    import json

    if not osp.exists(osp.join(model_name, 'config.json')):
        if 'base' in model_name:
            return 768, 12
        elif 'large' in model_name:
            return 1024, 24

    with open(osp.join(model_name, 'config.json'), 'r') as f:
        cfg = json.load(f)
        hidden = cfg['hidden_size']
        layer = cfg['num_hidden_layers']
    return hidden, layer

def warmup_cosine_decay(optimizer, current_step, max_step, lr_min, lr_max, warmup=0.1):
    """
    first linearn warmup and then cosine learning rate decay
    """
    from math import cos, pi
    warmup_step = int(max_step * warmup)
    if current_step < warmup_step:
        lr = lr_max * (current_step / warmup_step)
    else:
        lr = (lr_max - lr_min) * (1 + cos(
            pi * (current_step - warmup_step) / (max_step - warmup_step)
        )) / 2 + lr_min
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def move_to_device(input:dict, device):
    """
    moving tokenizer's output to device
    """
    return {
        key : value.to(device) for key, value in input.items()
    }
# print(load_sentence_pair('datasets\Stsbenchmark\sts-dev.csv')[:5])
# print(load_sentence_pair('datasets\Stsbenchmark\sts-dev.csv')[-5:])
from torch.utils.data import DataLoader
test_loader = DataLoader(
    load_eval_datas('datasets\Stsbenchmark\sts-test.csv'),
    batch_size=2, shuffle=False        
)
for X in test_loader:
    print(X)
    break
# train_datas = load_sentence_pair('datasets\Stsbenchmark\sts-test.csv')
# train_loader = DataLoader(
#     train_datas, batch_size=2, shuffle=True
# )

# print(load_eval_datas('datasets\Stsbenchmark\sts-test.csv')[:2])
# print(load_eval_datas('datasets\Stsbenchmark\sts-test.csv')[-2:])
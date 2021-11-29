from torch.utils.data.dataloader import DataLoader
from mylib.utils import (
    StepCounter,
    MyLRScheduler,
    move_to_device,
    parse_args, 
    set_seed, 
    load_sentence_pair, load_single_sentence, load_eval_datas,    
    infer_hidden_size,
    get_device,
    warmup_cosine_decay
)
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F
from torch import nn, utils, optim
import os
import numpy as np
from os import path as osp
from myModule import (
    UniformSampler,
    WeightedSampler,
    SGLossOpt2,
    SGLossOpt3,
    SGLossOpt3Simplified,
    RegLoss,
    RegHiddenLoss,
    TotalLoss,
    SelfGuidedContraModel
)
from scipy.stats import pearsonr, spearmanr


SAMPLER_CLASS= {
    'uniform' : UniformSampler,
    'weighted' : WeightedSampler
}

SGLOSS_CLASS = {
    'opt2' : SGLossOpt2,
    'opt3' : SGLossOpt3,
    'opt3s' : SGLossOpt3Simplified
}

REGLOSS_CLASS = {
    'param' : RegLoss,
    'hidden' : RegHiddenLoss
}

TRAIN_FILE = {
    'train' : 'datasets\Stsbenchmark\sts-train.csv',
    'combine' : 'datasets\Stsbenchmark\sts-all.csv'
}

args = parse_args()

device = get_device()
temp = args.temp
lamb = args.lamb
model_name = args.model_name

hidden_size, layer_num = infer_hidden_size(model_name)

regloss = REGLOSS_CLASS[args.regloss]()
sgloss = SGLOSS_CLASS[args.loss](temp=temp)

if args.sampler == 'weighted':
    sample_weight_tensor = torch.FloatTensor(args.sample_weight)
    sampler = SAMPLER_CLASS['weighted'](sample_weight_tensor)
else:
    sampler = SAMPLER_CLASS['uniform']

totalloss = TotalLoss(sgloss, sampler, regloss, lamb)
tokenizer = AutoTokenizer.from_pretrained(model_name)


pearsonr_list_all, spearmanr_list_all = [], []
test_pearsonr_over_seed, test_spearmanr_over_seed = [], []
for seed in args.seed:
    set_seed(seed)
    model = SelfGuidedContraModel(model_name, totalloss, hidden_size)
    train_loader = DataLoader(
        load_single_sentence(TRAIN_FILE[args.train_file]), batch_size=args.batch_size, shuffle=True
    )

    dev_loader, test_loader = None, None

    if args.do_dev:
        dev_loader = DataLoader(
            load_eval_datas('datasets\Stsbenchmark\sts-dev.csv'),
            batch_size=args.batch_size, shuffle=False
        )

    if args.do_test:
        test_loader = DataLoader(
            load_eval_datas('datasets\Stsbenchmark\sts-test.csv'),
            batch_size=args.batch_size, shuffle=False        
        )


    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.9))
    step_counter = StepCounter(max_step=args.epoch_num * len(train_loader), cur_step=0)
    lr_sche = MyLRScheduler(
        sc=step_counter,
        decay_fn=warmup_cosine_decay,
        optimizer=optimizer,
        lr_min=args.lr*0.01,
        lr_max=args.lr,
        warmup=args.warmup
    )

    pearsonr_list, spearmanr_list = [], []
    best_spearmanr = -1
    for epoch in range(args.epoch_num):
        model.train()
        for batch_id, X in enumerate(train_loader):
            lr_sche.step()
            X = tokenizer(
                list(X),
                truncation=True,
                padding=True,
                max_length=args.max_sent_len,
                return_tensors='pt'
            )
            X = move_to_device(X, device)
            loss = model(**X)
            loss.backward()
            optimizer.step()
            loss_val = loss.item()
        
        if args.do_dev:
            model.eval()
            bertT = model.bertT
            predict = []
            groundtruth = []
            with torch.no_grad():
                for batch_id, (label, sent1, sent2) in enumerate(dev_loader):
                    sent1 = move_to_device(
                        tokenizer(
                            list(sent1),
                            truncation=True,
                            padding=True,
                            max_length=args.max_sent_len,
                            return_tensors='pt'
                        ),
                        device=device)
                    sent2 = move_to_device(
                        tokenizer(
                            list(sent2),
                            truncation=True,
                            padding=True,
                            max_length=args.max_sent_len,
                            return_tensors='pt'
                        ),
                        device=device)
                    label = label.cpu().numpy().tolist()
                    cls1, cls2 = bertT(**sent1).pooler_output, bertT(**sent2).pooler_output
                    cosine_similarities = F.cosine_similarity(cls1, cls2, dim=-1).cpu().numpy().tolist()     

                    predict.extend(cosine_similarities)
                    groundtruth.extend(label)      
            corr_pearson = pearsonr(predict, groundtruth)[1]
            corr_spearman = spearmanr(predict, groundtruth).correlation 
            if corr_spearman > best_spearmanr:
                best_spearmanr = corr_spearman
                torch.save(bertT.state_dict(), osp.join(args.save_path, 'pytorch_model.bin'))

    if args.do_test:
        model.bertT.load_state_dict(torch.load(osp.join(args.save_path, 'pytorch_model.bin')))
        model.eval()
        bertT = model.bertT
        predict, groundtruth = [], []
        with torch.no_grad():
            for batch_id, (label, sent1, sent2) in enumerate(test_loader):
                sent1 = move_to_device(
                    tokenizer(
                        list(sent1),
                        truncation=True,
                        padding=True,
                        max_length=args.max_sent_len,
                        return_tensors='pt'
                    ),
                    device=device)
                sent2 = move_to_device(
                    tokenizer(
                        list(sent2),
                        truncation=True,
                        padding=True,
                        max_length=args.max_sent_len,
                        return_tensors='pt'
                    ),
                    device=device)
                label = label.cpu().numpy().tolist()
                cls1, cls2 = bertT(**sent1).pooler_output, bertT(**sent2).pooler_output
                cosine_similarities = F.cosine_similarity(cls1, cls2, dim=-1).cpu().numpy().tolist()  
                predict.extend(cosine_similarities)
                groundtruth.extend(label)
        corr_pearson = pearsonr(predict, groundtruth)[1]
        corr_spearman = spearmanr(predict, groundtruth).correlation   
        test_pearsonr_over_seed.append(corr_pearson)
        test_spearmanr_over_seed.append(corr_spearman)

print(f'average spearmanr over {len(args.seed)} seed : {round(np.mean(test_spearmanr_over_seed) * 100, 2)}')
print(f'average pearsonr over {len(args.seed)} seed : {round(np.mean(test_pearsonr_over_seed) * 100, 2)}')

# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--save_path', type=str)
#     parser.add_argument('--train_file', type=str, choices=['combine', 'train'], default='combine')
#     parser.add_argument('--loss', type=str, choices=['opt2', 'opt3', 'opt3s'])
#     parser.add_argument('--regloss', type=str, choices=['param', 'hidden'], default='param')
#     parser.add_argument('--lamb', type=float, default=0.1, help='weight of regularization term')
#     parser.add_argument('--temp', type=float, help='cosine similarity scaling factor ')
#     parser.add_argument('--sampler', type=str, choices=['uniform', 'weighted'], default=None)
#     parser.add_argument('--sample_weight', type=float, nargs='*')
#     parser.add_argument('--do_train', action='store_true', default=False)
#     parser.add_argument('--do_eval', action='store_true', default=False)
#     parser.add_argument('--do_test', action='store_true', default=False)
#     parser.add_argument('--lr', type=float)
#     parser.add_argument('--batch_size', type=int)
#     parser.add_argument('--epoch_num', type=int)
#     parser.add_argument('--max_sent_len', type=int, default=128)
#     parser.add_argument('--seed', type=int, default=20211129)




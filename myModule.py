from re import template
from torch.nn.modules.activation import GELU
from transformers import (
    AutoModel, 
    AutoTokenizer
)
import numpy as np
import random
import torch
from torch import nn, utils, optim
from torch.utils.data import DataLoader
from torch.nn import functional as F
import os
from os import path as osp

"""
Self-Guided Contrastive Learning for BERT Sentence Representations
https://arxiv.org/abs/2106.07345
"""

class UniformSampler(nn.Module):
    """
    Uniformly sample the hidden states of each layer, in other words, average the hidden states of all layers
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, hidden_states):
        """
        hidden_states: the hiddens states of all layers after poolings. [batch_size, layer_num, hidden_dim]
        return: hiddens states after uniform sample. [batch_size, hidden_dim]
        """
        if hidden_states.ndim != 3:
            raise NotImplementedError('hidden_states\' dimensions shoule be 3, including(batch, layer, hidden)')
        
        return torch.mean(hidden_states, dim=1, keepdim=False)

class WeightedSampler(nn.Module):
    """
    Weighted Average over the hidden_states of all layers. 
    """
    def __init__(self, weights:torch.FloatTensor):
        super().__init__()
        self.weights = weights
    
    def forward(self, hidden_states, weights:torch.FloatTensor =None):
        if weights is not None:
            w = self.weights
        else:
            w = weights
        
        if hidden_states.ndim != 3:
            raise NotImplementedError('hidden_states\' dimensions shoule be 3, including(batch, layer, hidden)')

        batch_size, layer_num, hidden_dim = hidden_states.shape
        if layer_num != len(w):
            raise NotImplementedError('layer_num should have same length with weights')
        
        #sum over w == 1, [layer_num]
        w = w / w.sum()

        return torch.sum(
            hidden_states * w.unsqueeze(0).unsqueeze(-1)
        )

class SGLossOpt2(nn.Module):
    """
    Exactly the same loss func with Unsupervised SimCSE
    """
    def __init__(self, temp):
        super().__init__()
        self.temp = temp
    
    def forward(self, cls, hidden):
        """
        both cls and hidden are in same dimension [batch_size, hidden], cls is [cls_token] from BERT_T, hidden is [sampler_out] from BERT_F
        return loss, sim
        """
        # using broadcast to calculate similarities, sim[batch_size, batch_size]
        sim = F.cosine_similarity(cls.unsqueeze(1), hidden.unsqueeze(0), dim=-1) / self.temp
        label = torch.arange(sim.shape[0]).long().to(sim.device)

        return F.cross_entropy(sim, label), sim

class SGLossOpt3(nn.Module):
    """
    Opt3 loss(SG-opt loss) in "Self-Guided Contrastive Learning for BERT Sentence Representations"
    in this optimize objectives, Sampler is not used    
    """
    def __init__(self, temp):
        super().__init__()
        self.temp = temp
    
    def forward(self, cls, hidden):
        """
        cls:[batch_size, hidden_dim]
        hidden:[batch_size, layer_num, hidden_dim]
        return loss, sim
        """        

        if hidden.ndim != 3:
            raise NotImplementedError('hidden_states\' dimensions shoule be 3, including(batch, layer, hidden)')

        batch_size, layer_num, hidden_dim = hidden.shape

        #sim_ci_hik [batch_size, layers]
        #sim_ci_hmn [batch_size, batch_size, layers]
        sim_ci_hik = torch.exp(F.cosine_similarity(cls.unsqueeze(1), hidden, dim=-1)) / self.temp
        sim_ci_hmn = torch.exp(torch.stack([
            F.cosine_similarity(c_i, hidden, -1) for c_i in cls
        ], dim=0)) / self.temp


        #hmn_mask [batch, batch*layers] 
        #sim_ci_hmn reshape [batch, batch*layers]
        hmn_mask = (torch.ones(batch_size, batch_size) - torch.eye(batch_size)).repeat(1, layer_num)
        sim_ci_hmn = sim_ci_hmn.reshape(batch_size, batch_size * layer_num)
        
        sim_after_mask = sim_ci_hmn * hmn_mask

        loss_list = []
        for i in range(batch_size):
            for k in range(layer_num):
                loss_list.append(
                    - torch.log(sim_ci_hik[i, k]) \
                    + torch.log(sim_ci_hik[i, k] + sim_after_mask[i].sum())
                )
        return torch.stack(loss_list).mean()
        

class SGLossOpt3Simplified(nn.Module):
    """
    Simplified Opt3 loss(SG-opt loss) in "Self-Guided Contrastive Learning for BERT Sentence Representations"
    In fact, since I couldn't write the loss function in the paper, I simplified it
    in this optimize objectives, Sampler is not used
    """
    def __init__(self, temp):
        super().__init__()
        self.temp = temp
    
    def forward(self, cls, hidden):
        """
        cls:[batch_size, hidden_dim]
        hidden:[batch_size, layer_num, hidden_dim]
        return loss, sim
        """

        if hidden.ndim != 3:
            raise NotImplementedError('hidden_states\' dimensions shoule be 3, including(batch, layer, hidden)')

        batch_size, layer_num, hidden_dim = hidden.shape
        #这个损失要如何写成矩阵运算的形式？
        #How do I write this loss in terms of matrix operations? Not the for loop.

        #step1 计算损失函数的分子
        #sim_ci_hik [batch_size, layers], 
        #sim_ci_hik[i] 代表第i个句子中，c_i和h_i0 ~ h_il的相似度
        sim_ci_hik = torch.exp(F.cosine_similarity(cls.unsqueeze(1), hidden, dim=-1)) / self.temp

        #step2 计算损失的分母
        #sim_ci_hmn [batch_size, batch_size, layers]
        #sim_ci_hmn[i] 代表第i个句子和其他所有句子所有层的相似度矩阵
        sim_ci_hmn = torch.exp(torch.stack([
            F.cosine_similarity(c_i, hidden, -1) for c_i in cls
        ], dim=0)) / self.temp

        #log(a) + log(b) = log(a*b)
        #对分子而言， sum over batch and layers： sim_ci_hik所有元素相乘
        #对分母而言，分两个步骤，由于简化过，sum over layers = sim_ci_hmn[i].sum() ^ k, 其中i是固定的
        #分母的第二个步骤, sum over batch, 对i进行遍历, prod over (sim_ci_hmn[i].sum() ^ k), i为遍历变量,
        #之后对上述结果做-log即可


        #[batch_size * layers]
        sim_ci_hik = sim_ci_hik.reshape(-1)
        loss1 = - torch.log(sim_ci_hik).sum()

        #[batch_size]
        sum_over_mn = sim_ci_hmn.sum(dim=[1, 2])
        loss2 = torch.log(sum_over_mn).sum()

        return loss1 + loss2

class RegHiddenLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, hidden1, hidden2):
        #input: hidden_states(tuple of tensor)
        #output: loss
        param_list = []
        for h1, h2 in zip(hidden1, hidden2):
            h = (h1 - h2).reshape(-1).pow(2).sum()
            param_list.append(h)
        
        return torch.sum(torch.stack(param_list)).sqrt()

class RegLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, param1, param2):
        #input: model.encoder.parameters()
        #output: loss
        param_list = []
        for part1, part2 in zip(param1, param2):
            param = (part1 - part2).reshape(-1).pow(2).sum()
            param_list.append(param)
        
        return torch.sum(torch.stack(param_list)).sqrt()

class TotalLoss(nn.Module):
    def __init__(self, sgloss, sampler, regloss, lamb=0.1):
        self.sgloss = sgloss
        self.sampler = sampler
        self.regloss = regloss
        self.lamb = 0.1
    
    def forward(self, cls, hiddens, p1, p2):
        """
        cls: [batch, hidden]
        hiddens: [batch, layers, hidden]
        """
        if not isinstance(self.sgloss, (SGLossOpt3, SGLossOpt3Simplified)):
            hiddens = self.sampler(hiddens)
        
        return self.sgloss(cls, hiddens) + self.lamb * self.regloss(p1, p2)


class SelfGuidedContraModel(nn.Module):
    def __init__(self, model_name, total_loss, hidden):
        super.__init__()
        self.bertF = AutoModel.from_pretrained(model_name)
        self.bertT = AutoModel.from_pretrained(model_name)
        self.proj = nn.Sequential(
            nn.Linear(hidden, 4096),
            nn.GELU(),
            nn.Linear(4096, hidden),
            nn.GELU()
        )
        self.loss_fn = total_loss
        self._freeze_param()


    def _freeze_param(self):
        for name, param in self.bertT.encoder.named_parameters():
            if 'layer.0' in name:
                param.requires_grad_(False)
        
        for name, param in self.bertF.encoder.named_parameters():
            param.requires_grad_(False)


    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None, inputs_embeds=None):
       
        #[batch_size, hidden_dim]
        pooler_output = self.bertT(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        ).pooler_output
        pooler_output = self.proj(pooler_output)

        #[batch_size, layers, hidden_dim]
        hiddens = self.bertF(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        ).hidden_states

        hiddens = torch.stack(hiddens, dim=0).mean(-2).transpose(0, 1)
        hiddens = self.proj(hiddens)

        if isinstance(self.loss_fn.regloss, RegLoss):
            return self.loss_fn(
                pooler_output,
                hiddens,
                self.BertF.parameters(),
                self.BertT.parameters()
            )

        elif isinstance(self.loss_fn.regloss, RegHiddenLoss):
            raise NotImplementedError()






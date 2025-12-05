import torch
import math

import  torch.nn as nn
from torch.nn import functional as F

# LN
class LayerNorm(nn.Module):
    def __init__(self,config):
        super.__init__()
        self.weight=nn.Parameter(torch.ones(config.n_embed))   #torch.Tensor
        self.bias=nn.Parameter(torch.zeros(config.n_embed) )  if config.bias_ln else None # 注意LN公式
    
    def forward(self,x):
        return F.layer_norm(x,self.weight.shape,self.weight,self.bias,1e-5)

# FFN
class FFN(nn.Module):
    """ block中的FFN，先升维再降维"""
    def __init__(self,config):
        super().__init__()
        self.up_fc=nn.Linear(config.n_embed,config.n_embed*4,config.bias_FFN)
        self.glue=nn.GELU()
        self.down_fc=nn.Linear(config.n_embed*4,config.n_embed,config.bias_FFN)
        self.dropout=nn.Dropout(config.dropout_FFN)
    def forward(self,x):
        x=self.up_fc(x)
        x=self.glue(x)
        x=self.down_fc(x)
        x=self.dropout(x)
        return x

# casualselfatt
class Casualselfatt(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_embed=config.n_embed
        self.n_head=config.n_head
        self.bias=config.bias_casualselfatt
        self.qkv_fc=nn.Linear(self.n_embed,self.n_embed*3,self.bias)
        self.proj=nn.Linear(self.n_embed,self.n_embed,self.bias)
        
        self.dropout_qkv_fc=nn.Dropout(config.dropout_casualselfatt)
        self.dropout_proj=nn.Dropout(config.dropout_casualselfatt)
        
        self.register_buffer("mask_matrix",torch.tril(torch.ones(config.max_len,config.max_len)).view(1,1,config.max_len,config.max_len))
    
    def forward(self,x):
        batchsize,seq_len,n_embed=x.size()
        qkv=self.qkv_fc(x)
        q,k,v=qkv.split(self.n_embed,dim=2)
        
        # batchsize,n_head,seq_len,per_head_embed
        q=q.view(batchsize,self.n_head,seq_len,n_embed//self.n_head)
        k=k.view(batchsize,self.n_head,seq_len,n_embed//self.n_head)
        v=v.view(batchsize,self.n_head,seq_len,n_embed//self.n_head)
        
        # batchsize,n_head,seq_len,seq_len
        att=q@k.transpose(-1,-2)*(1.0/math.sqrt(n_embed//self.n_head))
        att=att.masked_fill(self.mask_matrix[:,:,:seq_len,:seq_len],float("-inf"))
        
        # batchsize,n_head,seq_len,seq_len
        att=F.softmax(att,dim=-1)
        att=self.dropout_qkv_fc(att)
        
        # batchsize,n_head,seq_len,per_head_embed
        res=att@v
        
        ##########contiguous(),这个连续存储不能忘
        res=res.transpose(1,2).contiguous().view(batchsize,seq_len,n_embed)  # batchsize,seq_len,n_embed
        
        res=self.proj(res)
        res=self.dropout_proj(res)
        return res   
              
     
# Block
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.LN_1=LayerNorm(config)
        self.LN_2=LayerNorm(config)
        self.FFN=FFN(config)
        self.casualselfatt=Casualselfatt(config)
    
    def forward(self,x):
        x=self.LN_1(x)
        x=x+self.Casualselfatt(x)
        x=self.LN_2(x)
        x=x+self.FFN(x)
        
        return x

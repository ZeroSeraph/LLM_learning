import math
import torch
import torch.nn as nn
from torch.nn import functional as F
# FFN
class FFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.up_fc=nn.Linear(config.n_embed,config.n_embed*4,config.bias_ffn)
        self.down_fc=nn.Linear(config.n_embed*4,config.n_embed,config.bias_ffn)
        self.glue=nn.GELU()
        self.dropout=nn.Dropout(config.dropout)
    def forward(self,x):
        x=self.up_fc(x)
        x=self.glue(x)
        x=self.down_fc(x)
        x=self.dropout(x)
        return x
    
# LayerNorm 参考：https://zhuanlan.zhihu.com/p/18446035638
class LayerNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.weight=nn.Parameter(torch.ones(config.n_embed))
        self.bias=nn.Parameter(torch.zeros(config.n_embed)) if config.bias_ln else None
    def forward(self,x): # shape: batchsize,seq_len,n_embed
        x=F.layer_norm(x,x.size(-1),self.weight,self.bias)
        return x
    
# MHA
class MHA(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.qkv_proj=nn.Linear(config.n_embed,config.n_embed*3,config.bias_mha)
        self.proj=nn.Linear(config.n_embed,config.n_embed,config.bias_mha)
        
        self.dropout=nn.Dropout(config.dropout)
        
        self.n_head=config.n_head
        self.n_embed=config.n_embed
        self.per_head_embed=self.n_embed//self.n_head
        
    def forward(self,x,masked_matrix):  
        # masked_matrix由tokenizer生成，mask那些padding token
        # x_shape:batchsize,seq_len,n_embed
        batchsize,seq_len,n_embed=x.size()
        qkv=self.qkv_proj(x)
        q,k,v=qkv.split(n_embed,-1)  # 按照最后一维度划分：n_embed*3=>n_embed,n_embed,n_embed
        
        q=q.view(batchsize,seq_len,self.n_head,self.per_head_embed).transpose(1,2)
        k=k.view(batchsize,seq_len,self.n_head,self.per_head_embed).transpose(1,2)
        v=v.view(batchsize,seq_len,self.n_head,self.per_head_embed).transpose(1,2)
        
        att=q@k.transpose(-1,-2)   # att_shape: batchsize,n_head,seq_len,seq_len
        att=att.masked_fill(masked_matrix,float("-inf"))  # masked_fill(mask, value) -> Tensor
        att=F.softmax(att,dim=-1) 
        att=self.dropout(att)
        
        res=att@v   # res_shape: batchsize,n_head,seq_len,per_head_embed
        res=res.transpose(1,2).congituous().view(batchsize,seq_len,n_embed)
        res=self.proj(res)
        res=self.dropout(res)
              
        return x
        
# Block
class encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.LN_1=LayerNorm(config.n_embed,config.bias)
        self.attn=MHA(config)
        self.LN_2=LayerNorm(config.n_embed,config.bias)
        self.mlp=FFN(config)
    
    def forward(self,x):
        x=x+self.attn(self.LN_1(x))
        x=x+self.mlp(self.LN_2(x))
        return x


        
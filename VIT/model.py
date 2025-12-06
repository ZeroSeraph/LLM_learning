import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass

class PatchEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_size=config.patch_size
        self.n_patches=(config.image_size//self.patch_size)**2
        self.n_embed=config.n_embed
        self.in_channels=config.in_channels
        self.proj=nn.Conv2d(in_channels=self.in_channels,
                            out_channels=self.n_embed,
                            kernel_size=self.patch_size,
                            stride=self.patch_size
                            )
    def forward(self,x):
        # x_shape : batchsize,c,h,w
        x=self.proj(x) # batchsize,n_embed,n_patches**0.5,n_patches**0.5
        x=x.flatten(2) # 对x从shape的第3个维度开始，也就是对h，w合并成一个维度h*w,
        x=x.transpose(1,2) # batchsize,n_embed,n_patches=>batchsize,n_patches,n_embed
        return x

class MHA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head=config.n_head
        self.n_embed=config.n_embed
        self.bias_mha=config.bias_mha
        self.dropout=nn.Dropout(config.dropout)
        self.qkv_proj=nn.Linear(self.n_embed,self.n_embed*3,self.bias_mha)
        self.proj=nn.Linear(self.n_embed,self.n_embed,self.bias_mha)
    def forward(self,x):
        batchsize,seq_len,embed=x.size()
        q,k,v=x.split(self.n_embed,-1)
        
        #   batchsize,n_head,seq_len,per_head_embed
        q=q.view(batchsize,seq_len,self.n_head,self.n_embed//self.n_head).transpose(1,2)
        k=k.view(batchsize,seq_len,self.n_head,self.n_embed//self.n_head).transpose(1,2)
        v=v.view(batchsize,seq_len,self.n_head,self.n_embed//self.n_head).transpose(1,2)
        
        # batchsize,n_head,seq_len,seq_len
        att=q@k.transpose(-1,-2)
        att=F.softmax(att,-1)
        att=self.dropout(att)
        
        # batchsize,n_head,seq_len,per_head_embed
        x=att@v
        x=x.transpose(1,2).contiguous().view(batchsize,seq_len,self.n_embed)
        x=self.proj(x)
        x=self.dropout(x)
        
        return x
        
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.up_fc=nn.Linear(config.n_embed,config.n_embed*4)    
        self.down_fc=nn.Linear(config.n_embed*4,config.n_embed)
        self.gelu=nn.GELU()
        self.dropout=nn.Dropout(config.dropout)
    
    def forward(self,x):
        # x_shape:batchsize,seq_len,n_embed
        x=self.up_fc(x)
        x=self.gelu(x)
        x=self.down_fc(x)
        x=self.dropout(x)
        
        return x  

class LayerNorm(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config=config
        self.weight=nn.Parameter(torch.ones(config.n_embed))
        self.bias=nn.Parameter(torch.zeros(config.n_embed)) if config.bias_ln else None
    
    def forward(self,x):
        return F.layer_norm(x,self.config.n_embed,self.weight,self.bias)

class block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.att=MHA(config)
        self.ln_1=LayerNorm(config)
        self.ln_2=LayerNorm(config)
        self.ffn=MLP(config)
        
    def forward(self,x):
        x=x+self.att(self.ln_1(x))
        x=x+self.ffn(self.ln_2(x))
        
        return x

        # num_embeddings: int,
        # embedding_dim: int,
class vit(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config=config
        self.patch_embedding=PatchEmbedding(self.config)

        self.pos_embed=nn.Embedding((config.image_size//config.patch_size)**2,config.n_embed)
        self.dropout=nn.Dropout(config.dropout)
        
        self.blocks=nn.ModuleList([block(config) for i in range(config.layers)])
        self.norm=LayerNorm(config)
        self.head=nn.Linear(config.n_embed,config.n_class)
        
        self.apply(self._init_weights)
    
    def forward(self,x,traget=None):
        # x_shape:batchsize,c,h,w
        x=self.patch_embedding(x)  # batchsize,n_patches,n_embed
        batchsize,n_patches,n_embed=x.size()
        cls_token=torch.ones(batchsize,1,n_embed)
        x=torch.cat((x,cls_token),dim=1)  # batchsize,1+n_patches,n_embed
        for block in self.blocks:
            x=block(x)
        x=self.norm(x)
        x_embedding=x[:,0,:]  # batchsize,1,n_embed
        x_class_prob=self.head(x_embedding) # batchsize,1,n_class
        
        if traget is not None:
            x_class_prob=x_class_prob.squeeze(1)  # x_class:batchsize,n_class
            loss=F.cross_entropy(x_class_prob,traget.long())  # x_class:batchsize,n_class  traget:batchsize
            class_index=x_class_prob.argmax(dim=-1)
            return x_embedding.squeeze(1),class_index,loss
        else:
            class_index=x_class_prob.squeeze(1).argmax(-1)
            return x_embedding.squeeze(1),class_index
        
    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters())
    
    def _init_weights(self,module):
        if isinstance(module,nn.Linear) or isinstance(module,nn.Conv2d()) or isinstance(module,nn.Embedding()):
            torch.nn.init.normal_(module.weight,mean=0,std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module,LayerNorm):
            nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

            
        
        
import math
import inspect  # 用于自省,检查运行时的对象信息
from dataclasses import dataclass  # 用于简化类的定义,自动生成初始化方法等

import torch
import torch.nn as nn
from torch.nn import functional as F

"""
torch.nn.functional 包含了各种神经网络操作，但都是以函数形式而不是类形式:

激活函数：F.relu(), F.sigmoid(), F.tanh(), F.softmax()
损失函数：F.cross_entropy(), F.mse_loss(), F.l1_loss()
卷积/池化操作：F.conv2d(), F.max_pool2d()
Dropout：F.dropout()
规范化：F.batch_norm(), F.layer_norm()
"""
    
class LayerNorm(nn.Module):
    # nn.Parameter实例的参数会在训练过程中自动计算梯度并自动更新参数
    def __init__(self, ndim, bias):  # 初始化操作（自动完成）
        super.__init__()      # 调用父类的初始化方法
        self.weight = nn.Parameter(torch.ones(ndim))  # 每个维度一个缩放参数，所以权重是ndim维的
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
    
    def forward(self, input):
        return F.layer_norm(input,self.weight.shape,self.weight,self.bias,1e-5)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super.__init__()
        
        # qkv矩阵其实是从一个linear层得来的，然后再split
        self.c_att=nn.Linear(config.n_embed,config.n_embed*3,bias=config.bias_kvq)
        
        # 多头的输出concat后需要做线性变换
        self.c_proj=nn.Linear(config.n_embed,config.n_embed,bias=config.bias_proj)
        
        # dropout正则化:以一定的概率随机将输入张量中的一些元素（神经元的输出）设置为零
        self.att_dropout=nn.Dropout(config.dropout)
        self.proj_dropout=nn.Dropout(config.dropout)
        
        self.n_head=config.n_head
        self.n_embed=config.n_embed
        self.dropout=config.dropout
        
        # 将bias注册为缓冲区的内容，不会被更新
        self.register_buffer("max_mask_matrix",torch.tril(torch.ones(config.max_len,config.max_len))).view(1,1,config.max_len,config.max_len)
        
    def forward(self,x):
        batch_size, seq_len, embed_dim = x.size()
        
        # 按照特征维度对c_att(x)的结果进行split，
        # split用于沿着指定的维度将张量分割成大小相等的块，这里就是沿着012，也就是第3个维度，即输入x的第三个维度特征维度对c_att(x)的结果进行切分
        q,k,v=self.c_att(x).split(self.n_embed,dim=2)
        
        # view操作，将q进行重新拆分，参数的话就是q调整之后的shape，
        # 然后transpose就是调整成为(batch_size,self.n_head,seq_len,embed_dim//self.n_head)
        q=q.view(batch_size,seq_len,self.n_head,embed_dim//self.n_head).transpose(1,2)
        k=k.view(batch_size,seq_len,self.n_head,embed_dim//self.n_head).transpose(1,2)
        v=v.view(batch_size,seq_len,self.n_head,embed_dim//self.n_head).transpose(1,2)
        
        
        # 做注意力计算
        # q batch_size,n_head,seq_len,per_h_ndim
        att=q@k.transpose(-1,-2)*(1.0/math.sqrt(k.size(-1)))
        
        att=att.masked_fill(self.max_mask_matrix[:,:,:seq_len,:seq_len],float("-inf"))
        
        att=F.softmax(att,-1)
        att=self.att_dropout(att)
        
        y=att@v # batch_size,nhead,seq_len,seq_len @ batch_size,nhead,seq_len,per_head_ndim==>batch_size,nhead,seq_len,per_head_ndim
        
        y.transpose(1,2).contiguous().view(batch_size,seq_len,embed_dim)
        y=self.c_proj(y)
        y=self.proj_dropout(y)
        
class MLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.up_fc=nn.Linear(config.n_embed,config.n_embed*4,bias=config.bias)
        self.gelu=nn.GELU()
        self.down_fc=nn.Linear(config.n_embed*4,config.n_embed,bias=config.bias)
        self.dropout=nn.Dropout(config.dropout)
    def forward(self,x):
        x=self.up_fc(x)
        x=self.gelu(x)
        x=self.down_fc(x)
        x=self.dropout(x)
        
        return x

class decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.LN_1=LayerNorm(config.n_embed,config.bias)
        self.attn=CausalSelfAttention(config)
        self.LN_2=LayerNorm(config.n_embed,config.bias)
        self.mlp=MLP(config)
    
    def forward(self,x):
        x=x+self.attn(self.LN_1(x))
        x=x+self.mlp(self.LN_2(x))
        return x

@dataclass  #@dataclass 后，你只需要定义字段和它们的类型注解
class GPTConfig:
    block_size: int=1024  # 字段名: 期望类型 = 默认值
    vocab_size: int=50304
    n_layer:int =12
    n_head:int =12
    n_embed:int=768
    dropout:float=0.0
    bias:bool=True

# class GPTConfig:
#     def __init__(self,config):
#         self.block_size=config.block_size
#         ...

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config=config
        
        self.transformer=nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab,config.n_embed), # 最开始的词嵌入，可学习
            wpe = nn.Embedding(config.block_size, config.n_embed), #最开始的位置嵌入，可学习
            
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([decoder(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embed, bias=config.bias), # 词嵌入和位置嵌入之后还要进行层归一化
        ))
        
        # 最后由于需要预测每个token的概率，所以需要这个
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # 如果左右时对象，那么相等的话其实是引用
        # 在 运行时 始终使用 同一个权重矩阵对象。而不是使用不同的参数矩阵
        self.transformer.wte.weight=self.lm_head.weight
        
        # 初始化所有权重
        self.apply(self._init_weights)

        for pn,p in self.named_parameters():
            if pn.endwith("c_proj.weight"):
                # 进行正态分布初始化,方差更小
                torch.nn.init.normal_(p,mean=0,std=0.02/math.sqrt(2* self.config.n_layer))
    
    def get_num_params(self,non_embedding=True):
        
        # p.numel() 得到p的参数的个数
        n_params=sum(p.numel() for p in self.parameters())
        
        # 因为wte和最终的lm_head共享权重，所以不能减去wte得到参数
        if non_embedding:
            n_params-=self.transformer.wpe.weight.numel()
        return n_params
    
    def _init_weights(self,module):
        if isinstance(module,nn.Linear):
            # 正态分布初始化线性变换层的权重
            torch.nn.init.normal_(module.weight,mean=0,std=0.02)
            if module.bias is not None:
                # bias初始化为0
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module,nn.Embedding):
            torch.nn.init.normal_(module.weight,mean=0,std=0.02)
    
    def forward(self,idx,targets=None):
        device=idx.device
        b,t=idx.size()
        
        pos=torch.arange(0,t,dtype=torch.long,device=device)  # (0,1,2,...t-1)
        
        token_emb=self.transformer.wte(idx)  # b,t=>b,t,n_embed   这个相当于查表操作，t个token，每一个token对应一个embedding
        pos_emb=self.transformer.wpe(pos)  # t=>t,n_embed
        
        # pytorch 广播机制。从尾部维度开始比较，维度大小要么相等，要么其中一个为 1（或该维度不存在）
        x=self.transformer.drop(token_emb+pos_emb)
        for block in self.transformer.h:
            x=block(x)
        x=self.transformer.ln_f(x) 
        
        # train模式，需要计算损失
        if targets is not None:
            logits=self.lm_head(x)  # b,t,n_embed=>b,t,vocab_size
            
            # 需要调整一下，输入调整成b*t,vocab_size，target是b*t
            loss=F.cross_entropy(logits.view(logits.size(0)*logits.size(1),logits.size(-1)),targets.view(targets.size(0)*targets.size(1)),ignore_index=-1)
        else:
            # 只取序列的最后一个token做预测
            logits=self.lm_head(x[:,[-1],:])
            loss=None
        return logits,loss
            
        
        
        
        
        
        
        
        
        
        

        
        
        
            
        
    
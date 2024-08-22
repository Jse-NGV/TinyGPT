"""
project: tinyGPT
author: Jserw
Start: 2024/8/21
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect
import math
import numpy as np
from dataclasses import dataclass

@dataclass
class GPTConfig:
        n_dim = 768
        n_head = 12
        vocab_size = 50304
        block_size = 1024
        n_layers = 12
        dropout = 0.0
        bias = True

class LayerNorm(nn.Module):
    def __init__(self, embd_dim, bias):
        super(LayerNorm, self).__init__()
        self.weights = nn.Parameter(torch.ones(embd_dim))
        self.bias = nn.Parameter(torch.zeros(embd_dim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weights.shape, self.weights, self.bias, 0.00005)

class CasualSelfAttention(nn.Module):
    def __init__(self, config):
        super(CasualSelfAttention, self).__init__()
        self.n_dim = config.n_dim
        self.n_head = config.n_head
        self.block_size = config.block_size
        assert self.n_dim % self.n_head == 0
        self.proj_qkv = nn.ModuleList([nn.Linear(self.n_dim, self.n_dim) for _ in range(3)])
        self.proj_c = nn.Linear(self.n_dim, self.n_dim)
        self.bias = torch.tril(torch.ones(self.block_size, self.block_size).reshape(1, 1, self.block_size, self.block_size))
        self.att_dropout = nn.Dropout(config.dropout)
        self.residual_dropout = nn.Dropout(config.dropout)

    def forward(self, x): # x (batch_size, seq_len, n_dim)
        batch_size, seq_len, n_dim = x.shape
        q, k, v = [f(x) for f in self.proj_qkv] # q,k,v (b, n, d)
        q.reshape(batch_size, seq_len, self.n_head, self.n_dim // self.n_head).transpose(1, 2) # b, h, n, hn
        k.reshape(batch_size, seq_len, self.n_head, self.n_dim // self.n_head).transpose(1, 2) # b, h, n, hn
        v.reshape(batch_size, seq_len, self.n_head, self.n_dim // self.n_head).transpose(1, 2) # b, h, n, hn

        att = (q @ k.transpose(-1, -2)) * (self.n_dim ** -0.5) # b h n n
        att.masked_fill(self.bias[:, :, :seq_len, :seq_len]==0, float('-inf'))
        att = F.softmax(att,dim=-1)
        att = self.att_dropout(att)
        y = att @ v # b h n nh
        y = y.transpose(1,2).reshape(batch_size, seq_len, n_dim) # b n d 不要忘记先transpose再reshape
        y = self.residual_dropout(self.proj_c(y))
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.proj_1 = nn.Linear(config.n_dim, 4*config.n_dim, config.bias)
        self.relu = nn.ReLU()
        self.proj_2 = nn.Linear(4*config.n_dim, config.n_dim, config.bias)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x):
        x = self.proj_1(x)
        x = self.relu(x)
        x = self.proj_2(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super(Block, self).__init__()
        self.att = CasualSelfAttention(config)
        self.ffn = MLP(config)
        self.ln1 = LayerNorm(config.n_dim, config.bias)
        self.ln2 = LayerNorm(config.n_dim, config.bias)

    def forward(self, x):
        # pre_norm
        x = self.att(self.ln1(x)) + x
        x = self.ffn(self.ln2(x)) + x
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super(GPT, self).__init__()
        self.word_embd = nn.Embedding(config.vocab_size, config.n_dim)
        self.pos_enc = nn.Embedding(config.block_size, config.n_dim) # 可训练的位置编码???
        self.layers = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.dropout = nn.Dropout(config.dropout)
        self.ln = LayerNorm(config.n_dim, config.bias)
        self.lm_head = nn.Linear(config.n_dim, config.vocab_size,bias=False)
        self.config = config

        self.pos_enc.weight = self.lm_head.weight

        # 初始化参数
        self.apply(self._init_weights)
        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # 获取参数量
    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        n_params -= self.pos_enc.weight.numel()
        return n_params


    def forward(self, idx, target=None):
        # idx 就是输入的句子 (b, t)
        b,t = idx.size()
        device = idx.device
        assert t <= self.config.block_size # 输入句子长度不能超过最大长度
        pos = torch.arange(0,t,dtype=torch.long, device=device) # t
        tok_emd = self.word_embd(idx) # b t n_dim
        pos_emd = self.pos_enc(pos) # b n_dim
        x = self.dropout(tok_emd + pos_emd) # python 广播机制 b t n_dim
        for layer in self.layers:
            x = layer(x)
        x = self.ln(x)
        if target is not None:
            # train
            logits = self.lm_head(x) # (b ,t ,vocab_size)
            # input (N,C) C是类别个数，target N         这里target(b,t) input-->(b*t, vocab_size) target-->(b*t)
            loss = F.cross_entropy(logits.reshape(-1,logits.size(-1)), target.reshape(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x) #(b, t, vocab_size)
            logits = logits[:,[-1],:] # 取生成的最后一个 (b, 1, vocab_size)
            loss = None
        return logits, loss # train时返回的是所有生成token的概率，推理时返回的是最后一个token的概率


    # to be modified
    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print('loading weights from pretrained gpt: %s' % model_type)
        # n_layers, n_head, n_embd are determined from model_type
        config_args = {
            'gpt2': dict(n_layers=12, n_head=12, n_dim=768), # 124M params
            'gpt2-medium': dict(n_layers=24, n_head=16, n_dim=1024), # 350M params
            'gpt2-large': dict(n_layers=36, n_head=20, n_dim=1280), # 774M params
            'gpt2-xl': dict(n_layers=48, n_head=25, n_dim=1600) # 1558M params
        }[model_type]
        print('vocab_size=50257, block_size=1024, bias=True')
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        config_args['bias'] = True

        # create a from-scratch initialized miniGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model


    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn:p for pn,p in self.named_parameters()}
        param_dict = {pn:p for pn,p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for pn,p in param_dict.items() if p.dim()>=2]
        nodecay_params = [p for pn,p in param_dict.items() if p.dim()<2]
        optim_groups = [
            {'params':decay_params, 'weight_decay':weight_decay},
            {'params':nodecay_params, 'weight_decay':0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        # idx (b, t)
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[ : , -self.config.block_size:] # (b, t)
            logits,_ = self(idx_cond)
            logits = logits[:, -1, :] / temperature # (b, vocab_size)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1) # (b, 1)
            idx = torch.cat((idx,idx_next), dim=1) # (b, t+1)
        return idx


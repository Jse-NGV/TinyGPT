# author: Jserw
# start: 2024/8/23
import math
import os
import pickle
import time

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP # nb
from torch.distributed import init_process_group, destroy_process_group
from contextlib import nullcontext
from model import GPTConfig, GPT # xs


# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# IO
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, 程序将在第一次测试后退出
always_save_checkpoint = True # if True save a checkpoint after each eval
init_from = 'scratch' # scratch | resume | gpt2

# wandb logging
wandb_log = False # default
wandb_project = 'owt'
wandb_run_name = 'gpt2'

# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes ??? unknown
batch_size = 12
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0, for finetune 0.1+ is better
bias = False
# adamw optimizer
learning_rate = 6e-4
max_iters = 600000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.endswith('_') and isinstance(v, (int, float, bool, str))]
config = {k:globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------

ddp = int(os.environ.get('RANK',-1))!=-1 # 判断当前是否处于分布式训练状态
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK']) # 一共多少个显卡，比如两台8卡，则rank 0~15
    ddp_local_rank = int(os.environ['LOCAL_RANK']) # 当前节点多少卡，还是上述例子，节点1 0~7，节点2也是0~7
    ddp_world_size = int(os.environ['WORLD_SIZE']) # 总显卡数
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device=device)
    master_process = ddp_rank == 0 # 主进程,this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # 这样做的目的是确保在分布式训练环境中，每个进程得到不同的随机种子，从而使其数据处理和模型初始化各不相同，增加训练的多样性和稳定性。

    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single GPU, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

tokens_per_iter = gradient_accumulation_steps * block_size * batch_size * ddp_world_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

data_dir = os.path.join('data', dataset)
def get_batch(split): # 还需要知道.bin文件里面是怎么存储的
    if split == 'train':
        data = np.memmap(os.path.join(data_dir,'train.bin'),dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data)-block_size, (batch_size,)) # (b) 随机抽样起始点，构造训练数据集
    x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix]) # (b,block_size)
    y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ix]) # (b,block_size) # 妙啊
    if device_type == 'cuda':
        # pin_memory() 是 PyTorch 中的一个方法，用于优化数据加载过程。它将数据加载到锁页内存（pinned memory）中，加速数据从主机内存到GPU的传输。
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f'found vocab_size = {meta_vocab_size} (inside {meta_path})')

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size, bias=bias, vocab_size=None, dropout=dropout)

if init_from == 'scratch':
    print('Initializing a new model from scratch')
    if meta_vocab_size is None:
        print('default to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)')
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f'Resuming training from {out_dir}')
    # resume training from a checkpoint
    ckpt_path = os.path.join(out_dir,'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]

    gptconfig = GPTConfig(model_args)
    model = GPT(gptconfig)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
model.to(device)

scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None

# compile the model
if compile:
    print('compile the model... (take a ~minute)')
    unoptimized_model = model
    model = torch.compile(model)

# wrap the model into DDP conditioner
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

def get_lr(it):
    if it < warmup_iters:
        return it / warmup_iters * learning_rate
    if it > lr_decay_iters:
        return min_lr
    ratio = (it-warmup_iters) / (lr_decay_iters-warmup_iters)
    coeff = 0.5 * (1+math.cos(math.pi * ratio))
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train','val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean() # 计算均值
    model.train()
    return out

# training loop
X, Y = get_batch('train') # first batch
t0 = time.time()
local_iter_num = 0
raw_model = model.module if ddp else model
while True:
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                'iter': iter_num,
                'train loss': losses['train'],
                'val loss': losses['val'],
                'lr': lr
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config
                }
                print(f'saving checkpoint to {out_dir}')
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient ???
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer=optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        print(f'iter {iter_num}: loss {lossf:.4f}, time: {dt * 1000:.2f}ms.')
    iter_num += 1
    local_iter_num += 1

    # termination
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
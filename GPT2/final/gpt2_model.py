import os
os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"
from dataclasses import dataclass
import torch
import torch.nn as nn
import inspect
from torch.nn import functional as F
import math
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# ---------------------------------

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)

        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

        self.n_head = config.n_head
        self.n_embd = config.n_embd

        self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size)).view(1,1,config.block_size, config.block_size))

    def forward(self, x):

        B, T, C = x.size()


        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=-1)

        k = k.view(B, T, self.n_head, C//self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C//self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C//self.n_head).transpose(1, 2)


        # att = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:,:,:T,:T]==0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # y = att @ v
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(config.n_embd * 4, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight
        # init all weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= math.sqrt(2 * self.config.n_layer)
            torch.nn.init.normal_(module.weight, mean=0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot find sequence of length {T}"

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = pos_emb + tok_emb

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None

        if targets is not None:
           loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) 
        return logits, loss


    @classmethod
    def from_pretrained(cls, model_type):
        """ Loads pretrained GPT-2 model weight from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print(f"loading weights from pretrained gpt: {model_type}")

        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1200),
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024

        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        assert len(sd_keys_hf) == len(sd_keys) , f'mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}'
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, device):

        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_patams = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_patams, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_patams)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_patams)}, with {num_nodecay_params:,} parameters")

        fused_avialable = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_avialable and 'cuda' in device
        print(f'using fused AdamW: {use_fused}')
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer


import tiktoken
import time
import numpy as np
# data loder

def load_tokens(filename):
    npt = np.load(filename)
    ppt = torch.tensor(npt, dtype=torch.long)
    return ppt

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in ['train', 'val']
        
        data_root = '/mnt/ssd1/zfz/edu_fineweb100'
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for {split}"
        if master_process:
            print(f"found {len(shards)} shards for {split}")
        
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_postition = self.B * self.T * self.process_rank
    
        
    
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_postition: self.current_postition + B*T + 1]
        x = buf[:-1].reshape(B, T)
        y = buf[1:].reshape(B, T)

        self.current_postition += B*T * self.num_processes

        if self.current_postition + B*T + 1 > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_postition = B * T * self.process_rank
        return x, y


def get_lr(step):
    if step < warmup_steps:
        return max_lr * (step+1) / warmup_steps
    
    if step > max_steps:
        return min_lr

    decay_ratio = (step - warmup_steps)/(max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

if __name__=='__main__':
    max_steps = 100
    max_lr = 6e-4
    min_lr = max_lr * 0.1
    warmup_steps = 715 # same with GPT 3
    max_steps = 19073


    # ------------------------------------------
    # simple launch with single GPU
    # python gpt2.py
    # DDP launch for multiple GPUs
    # torchrun --standalone --nproc_per_node=4 gpt2.py
    # ----------
    
    
    from torch.distributed import init_process_group, destroy_process_group 
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,4,5"

    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        assert torch.cuda.is_available()
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process= ddp_rank == 0
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        device = 'cuda' if torch.cuda.is_available else 'cpu'
        master_process= True
        print(f'using device {device}')

    

    # if device == 'cuda':
    #     print('set GPU:',torch.cuda.current_device())
    #     print('GPU name:',torch.cuda.get_device_name(torch.cuda.current_device()))

    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)

    # ------------------------

    total_batch_size = 524288
    B = 32
    T = 1024
    assert total_batch_size % (B*T*ddp_world_size) == 0, "batch size must be divisible by B*T*ddp_world_size"
    grad_accum_steps = total_batch_size // (B*T*ddp_world_size)
    if master_process:
        print(f"total desired batch size:{total_batch_size}")
        print(f"=> gradient accumulation steps:{grad_accum_steps}")

    dataloader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train") # batch size reference to the gpu memory

    # model = GPT.from_pretrained('gpt2')
    model = GPT(GPTConfig(vocab_size=50304))
    model = model.to(device)
    model = torch.compile(model) # compile the model, make training faster, it's very useful， but it needs more compile time
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model
    
    torch.set_float32_matmul_precision('high') # start TF32  // 8 times better than highest(fp32) (in reality, it is 3 times better)

    optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)

    
    
    for step in range(max_steps):
        t0 = time.time()
        optimizer.zero_grad()

        loss_accum = 0.0
        for micro_step in range(grad_accum_steps):
            x, y = dataloader.next_batch()
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16): # mixed precision, some layers selectively be running by bfloat16
                logits, loss = model(x, y)
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            if ddp: 
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1) ## 表示在最后一步，所有进程都要同步
            loss.backward()
        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # determine and set the learning rate for this iteration
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        optimizer.step()
        torch.cuda.synchronize()
        t1 = time.time()
        dt = (t1 - t0) * 1000
        tokens_pre_sec = dataloader.B * dataloader.T * grad_accum_steps * ddp_world_size / (t1 - t0)
        if master_process:
            print(f'stop: {step} | loss: {loss_accum.item()} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt:.2f} ms | tok/sec: {tokens_pre_sec:.2f}')
    if master_process:
        print('saving model to gpt2.pt')
        torch.save(model.state_dict(), 'gpt2.pth')
    
    if ddp:
        destroy_process_group()
    
    import sys; sys.exit(0)
    
    # #prefix tokens
    # import tiktoken
    # enc = tiktoken.get_encoding('gpt2') 
    # tokens = enc.encode("Hello, I'm a language model,")
    # tokens = torch.tensor(tokens, dtype=torch.long)
    # tokens = tokens.unsqueeze(0).repeat(num_return_sequence, 1)
    # x = tokens.to(device)

    # torch.manual_seed(42)
    # torch.cuda.manual_seed(42)
    # while x.size(1) < max_length:

    #     with torch.no_grad():

    #         logits = model(x)
    #         logits = logits[:,-1,:]
    #         probs = F.softmax(logits, dim=-1)

    #         topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)

    #         ix = torch.multinomial(topk_probs, 1)
            
    #         xcol = torch.gather(topk_indices, -1, ix)

    #         x = torch.cat((x, xcol), dim=1)
    
    # for i in range(num_return_sequence):
    #     tokens = x[i, :max_length].tolist()
    #     decoded = enc.decode(tokens)
    #     print(">", decoded)
import time
import numpy as np
from gpt2_model import *
from data_load import *
# data loder




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
    # python train_gpt2.py
    # DDP launch for multiple GPUs
    # torchrun --standalone --nproc_per_node=4 train_gpt2.py
    # ----------
    
    
    from torch.distributed import init_process_group, destroy_process_group 
    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

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

    dataloader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, master_process=master_process, split="train") # batch size reference to the gpu memory

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
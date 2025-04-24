# GPT-2

## speed up

1. `tensor float 32`

   ```python
   torch.set_float32_matmul_precision('high') # start TF32  // 8 times better than highest(fp32) (in reality, it is 3 times better)
   ```

   

2. `bfloat16` with mixed precious:

   ```python
   with torch.autocast(device_type='cuda', dtype=torch.bfloat16): # mixed precision, some layers selectively be running by bfloat16
           logits, loss = model(x, y)
   ```

3. `torch.compile()` 在GPU缓存层面进行调优加速，重新编译model，**减少GPU与GPU缓存（HBM）之间的读写次数**

   ```python
   model = GPT(GPTConfig())
   model = model.to(device)
   model = torch.compile(model) # compile the model, make training faster, it's very useful， but it needs more compile time
   
   ```

4. `Flash attention`

   分块思想

   涉及到softmax的动态更新（在线更新）

   ```python
   # att = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1)))
   # att = att.masked_fill(self.bias[:,:,:T,:T]==0, float('-inf'))
   # att = F.softmax(att, dim=-1)
   # y = att @ v
   y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
   ```

5. 参数的选择最好使用和 2 的次幂相关的数（这会大大提高效率）

   ```python
   #vocab_size: int = 50257
   vocab_size: int = 50304 # bring 4 percent improvement
   ```



## train optimize

1. optimizer的超参设置（参考GPT-3）

   ```python
   optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
   ```

2. 限制梯度的L2范数，防止过大幅度的参数更新

   ```python
   norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   ```

3. cosin decay for learning rate
   ```python
   max_steps = 100
   max_lr = 6e-4
   min_lr = max_lr * 0.1
   warmup_steps = 10
   max_steps = 50
   def get_lr(step):
       if step < warmup_steps:
           return max_lr * (step+1) / warmup_steps
       
       if step > max_steps:
           return min_lr
   
       decay_ratio = (step - warmup_steps)/(max_steps - warmup_steps)
       assert 0 <= decay_ratio <= 1
       coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
       return min_lr + coeff * (max_lr - min_lr)
   ```

4. weight decay（权重衰减）

   同时这里的fused也能加速模型训练，在optimize的过程中，类似于将for循环遍历所有参数的操作进行压缩，一次性取出多个参数进行更新，减少读写开销

   ```python
   
   optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)
   
   # model.configure_optimizers
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
   ```



## batch optimize

1. 通过累积的方式，在显存受限的情况下，依然能够自定义batch的大小：

   ```python
   # initial
   total_batch_size = 524288
   B = 16
   T = 1024
   assert total_batch_size % (B*T) == 0, "batch size must be divisible by B*T"
   grad_accum_steps = total_batch_size // (B*T)
   print(f"total desired batch size:{total_batch_size}")
   print(f"=> gradient accumulation steps:{grad_accum_steps}")
   
   dataloader = DataLoaderLite(B=B, T=T) # batch size reference to the gpu memory
   
   
   # train:
   for ...
   	...
       for micro_step in range(grad_accum_steps):
           x, y = dataloader.next_batch()
           x, y = x.to(device), y.to(device)
           with torch.autocast(device_type='cuda', dtype=torch.bfloat16): # mixed precision, some layers selectively be running by bfloat16
               logits, loss = model(x, y)
           loss = loss / grad_accum_steps
           loss.backward()
   	...
   ```

   

## DistributedDataParallel

1. 多卡并行，用的是实验室的四张A800，远比教程中8张A100的平均性能差

   通过对比发现A800的并行性能远远比A100的性能差，**主要原因是A800的多卡并行通信被阉割过**

   1 张 A800 vs 4 张 A800

   1 张 A800：

   ![image-20250419014612888](C:\Users\26091\AppData\Roaming\Typora\typora-user-images\image-20250419014612888.png)

   4 张 A800：（性能提升400%左右）

   ![image-20250420230521430](C:\Users\26091\AppData\Roaming\Typora\typora-user-images\image-20250420230521430.png)

   

   8 张 A100：（性能提升900%， 差距显著， 遥遥领先）
   
   ![image-20250419015001341](C:\Users\26091\AppData\Roaming\Typora\typora-user-images\image-20250419015001341.png)





## 数据集

1. **red pajama data set**
2. **FineWeb data set**
   * FineWeb-edu：sample-10BT
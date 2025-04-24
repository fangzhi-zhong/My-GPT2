import torch
import numpy as np
import os

def load_tokens(filename):
    npt = np.load(filename)
    ppt = torch.tensor(npt, dtype=torch.long)
    return ppt

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, master_process, split):
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
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def setup_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])

        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        is_dist=True
    
    else:
        #single gpu setup
        rank,world_size,local_rank = 0,1,0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_dist = False
    

    return is_dist, rank, world_size, local_rank, device



def is_main():
    return int(os.environ["LOCAL_RANK"]) == 0


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()

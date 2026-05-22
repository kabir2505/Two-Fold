from src.pretrain.trainer import run_training

import yaml
from src.common_utils.distributed_setup import setup_ddp, is_main, cleanup_ddp
from src.common_utils.utils import make_optimizer, make_scheduler_cosine
from src.pretrain.utils.data_utils import get_loader
from src.common_utils.checkpoint import load_2fold_swin
from dotdict import dotdict
import argparse
import os
import torch
from monai.transforms import Activations, AsDiscrete, Compose
from monai.utils.enums import MetricReduction
from monai.networks.blocks import PatchEmbed, UnetOutBlock, UnetrBasicBlock, UnetrUpBlock
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from functools import partial
from src.pretrain.model.twofold_ssl import TwoFoldSSL
import random
import numpy as np
from src.pretrain.trainer import run_training
from src.common_utils.utils import seed_everything
from src.pretrain.utils.memo import activation_sizes, attach_activation_hooks


def main(args):
    print(f'lr={args["lr"]} (type={type(args["lr"])})')
    #setup ddp
    args.is_dist,args.rank,args.world_size,args.local_rank, args.device = setup_ddp()
    args.distributed = args.is_dist
    
    args.total_train_time=0.0

    #set seed
    eff_seed = seed_everything(base_seed=args["seed"], rank=args.rank, deterministic=True)
    print(f"[rank {args.rank}] effective_seed={eff_seed}")

    try:
        from monai.utils import set_determinism
        set_determinism(seed = eff_seed)
    except Exception:
        pass

    #get loader
    train_loader, val_loader = get_loader(args=args)
    
    n = args["patch_grid"]
    P = n**3
    
    model = TwoFoldSSL(in_ch = args.in_channels, 
                       out_ch = args.num_classes,
                       feature_size = args.feature_size,
                       num_patches = P,
                       K = args["rotations"]).to(args.device)
    
    if args.distributed:
        from torch.nn.parallel import DistributedDataParallel as DDP
        print("DDP training")
        model = DDP(
            model,
            device_ids=[args.rank],
            output_device=args.rank,
            find_unused_parameters=True,
        )    
        
        attach_activation_hooks(model.module)
    else:
        attach_activation_hooks(model)
    optimizer = make_optimizer(model, lr=args["lr"], wd=args["weight_decay"])
    scheduler = make_scheduler_cosine(optimizer, max_epochs = args["epochs"])
    
    
    if args.resume_checkpoint is not None:
        checkpoint = torch.load(args.resume_checkpoint, map_location="cpu")

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
        args.total_train_time = checkpoint.get("total_train_time",0.0)
    
        model.to(args.device)

    
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(args.device)

          
    best_loss = run_training(model=model,train_loader=train_loader,val_loader=val_loader,optimizer=optimizer,args=args,scheduler=scheduler,start_epoch=args.start_epoch)
    
    if args.rank == 0:
        print("Training Finished")
        print(f"best loss {best_loss}")
    return best_loss

    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process pretraining")
    parser.add_argument("--config",help="Path to config", required=True)
    args = parser.parse_args()
    config = args.config
    with open(config, "r") as f:
        config = yaml.safe_load(f)
        
    os.makedirs(config["outdir"],exist_ok=True)
    config_path = os.path.join(config["outdir"], "config_used.yaml")
    with open(config_path,"w") as f:
        yaml.dump(config,f)
        
    main(args=dotdict(config))

    

    
    

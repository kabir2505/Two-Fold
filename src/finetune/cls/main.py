import os
import yaml
import argparse
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from src.common_utils.distributed_setup import setup_ddp, is_main, cleanup_ddp
from src.common_utils.utils import make_optimizer, make_scheduler_cosine
from src.finetune.utils.data_utils_cls import get_loader
from src.finetune.cls.model import Swin
from src.finetune.cls.trainer import run_training
from src.common_utils.checkpoint import load_2fold_swin
from dotdict import dotdict

def main(args):
    args.is_dist, args.rank, args.world_size, args.local_rank, args.device = setup_ddp()
    args.distributed = args.is_dist    

    if args.rank == 0:
        os.makedirs(args.outdir, exist_ok=True)
        config_path = os.path.join(args.outdir, "config_used.yaml")
        with open(config_path, "w") as f:
            yaml.dump(dict(args), f)
            
        print(f"Learning Rate: {args.lr} (type={type(args.lr)})")

    train_loader, val_loader = get_loader(args=args)

    model = Swin(args=args).to(args.device)

    optimizer = make_optimizer(model, lr=args.lr, wd=args.weight_decay)

    args.start_epoch = 0

    if getattr(args, "resume_checkpoint", None):
        if args.rank == 0: print(f"Resuming from checkpoint: {args.resume_checkpoint}")
        checkpoint = torch.load(args.resume_checkpoint, map_location=args.device)
        
        model_state = {k.replace("module.", "", 1): v for k, v in checkpoint["model"].items()}
        model.load_state_dict(model_state)
        
        optimizer.load_state_dict(checkpoint["optimizer"])  
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(args.device)
                    
        args.start_epoch = checkpoint["epoch"] + 1

    elif getattr(args, "pretrain_checkpoint", None):
        if args.rank == 0: print(f"Loading pretrained weights: {args.pretrain_checkpoint}")
        model = load_2fold_swin(swinunetr_model=model, path=args.pretrain_checkpoint, rank=args.rank)
        model = model.to(args.device)

    if args.distributed:
        if args.rank == 0: print("Wrapping model in DDP")
        model = DDP(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=False,
        )

    scheduler = make_scheduler_cosine(optimizer, max_epochs=args.epochs)
    if getattr(args, "resume_checkpoint", None) and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer, 
        scheduler=scheduler,
        start_epoch=args.start_epoch, 
        args=args
    )
    
    if args.distributed:
        cleanup_ddp()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process finetuning for classification")
    parser.add_argument("--config", help="Path to config", required=True)
    _args = parser.parse_args()
    
    config_file = _args.config.strip()
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    config_dict = dotdict(config)

    main(args=config_dict)
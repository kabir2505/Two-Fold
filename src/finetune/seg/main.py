import os
import yaml
import argparse
import torch
from functools import partial

from monai.transforms import AsDiscrete
from monai.utils.enums import MetricReduction
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR
from torch.nn.parallel import DistributedDataParallel as DDP

from src.common_utils.distributed_setup import setup_ddp
from src.common_utils.utils import make_optimizer, make_scheduler_cosine, LinearWarmupCosineAnnealingLR
from src.finetune.utils.data_utils_seg import get_loader
from src.finetune.seg.trainer import run_training
from src.common_utils.checkpoint import load_2fold_swin
from src.finetune.utils.transforms_seg import (
    get_transforms_btcv, get_transforms_brats_18, get_transforms_amos, 
    get_transforms_word, get_transforms_mmwhs, get_transforms_spleen, 
    get_transforms_lits, get_transforms_brats
)
from dotdict import dotdict

def main(args):
    # Setup DDP
    args.is_dist, args.rank, args.world_size, args.local_rank, args.device = setup_ddp()
    args.distributed = args.is_dist

    if args.rank == 0:
        os.makedirs(args.outdir, exist_ok=True)
        config_path = os.path.join(args.outdir, "config_used.yaml")
        with open(config_path, "w") as f:
            yaml.dump(dict(args), f)  

    transform_map = {
        "btcv": get_transforms_btcv,
        "mmwhs": get_transforms_mmwhs,
        "spleen": get_transforms_spleen,
        "lits": get_transforms_lits,
        "brats": get_transforms_brats,
        "amos": get_transforms_amos,
        "word": get_transforms_word,
        "brats_18": get_transforms_brats_18
    }
    
    if args.dataset not in transform_map:
        raise ValueError(f"Unknown dataset: {args.dataset}")
        
    # Get loader
    train_transform, val_transform = transform_map[args.dataset](args=args)
    train_loader, val_loader = get_loader(args=args, train_transform=train_transform, val_transform=val_transform)

    # Setup segmentation model
    model = SwinUNETR(
        in_channels=args.in_channels,
        out_channels=args.num_classes,
        feature_size=args.feature_size,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=args.dropout_path_rate,
        use_checkpoint=args.use_checkpoint,
        use_v2=True,
    ).to(args.device)

    optimizer = make_optimizer(model, lr=args.lr, wd=args.weight_decay)
    
    args.start_epoch = 0

    # Load pretrained weights or resume checkpoint
    if getattr(args, "resume_checkpoint", None):
        checkpoint = torch.load(args.resume_checkpoint, map_location=args.device)    
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])  
        
        # Move optimizer states to proper device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(args.device)
                    
        args.start_epoch = checkpoint["epoch"] + 1  
        
        # Load scheduler state if resuming
        if "scheduler" in checkpoint:
            # Rebuild scheduler structure first
            if args.dataset in ["lits", "spleen"]:
                scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=args.warmup_epochs, max_epochs=args.epochs)
            else:
                scheduler = make_scheduler_cosine(optimizer, max_epochs=args.epochs)
            # Then load the state
            scheduler.load_state_dict(checkpoint["scheduler"])
            
    else:
        # Rebuild fresh scheduler if NOT resuming
        if args.dataset in ["lits", "spleen"]:
            scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=args.warmup_epochs, max_epochs=args.epochs)
        else:
            scheduler = make_scheduler_cosine(optimizer, max_epochs=args.epochs)
            
        if getattr(args, "pretrain_checkpoint", None):
            if args.rank == 0: 
                print(f"Loading pretrained weights: {args.pretrain_checkpoint}")
            model = load_2fold_swin(swinunetr_model=model, path=args.pretrain_checkpoint, rank=args.rank)
            model = model.to(args.device)

    # Wrap in DDP AFTER weight loading
    if args.distributed:
        model = DDP(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=False,
        )

    # Evaluation configuration
    inf_size = [args.roi_x, args.roi_y, args.roi_z]
    dice_loss = DiceCELoss(include_background=False, to_onehot_y=True, softmax=True)

    post_label = AsDiscrete(to_onehot=args.num_classes)
    post_pred = AsDiscrete(argmax=True, to_onehot=args.num_classes)
    
    dice_acc = DiceMetric(include_background=False, reduction=MetricReduction.NONE, get_not_nans=True)
    dice_acc_mean = DiceMetric(include_background=False, reduction=MetricReduction.MEAN, get_not_nans=True)
    
    model_inferer = partial(
        sliding_window_inference,
        roi_size=inf_size,
        sw_batch_size=args.sw_batch_size,
        predictor=model,
        overlap=args.infer_overlap
    )
    
    # Execute Training
    accuracy = run_training(
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        optimizer=optimizer, 
        loss_func=dice_loss,
        acc_func=dice_acc, 
        acc_func_mean=dice_acc_mean, 
        args=args,
        model_inferer=model_inferer, 
        scheduler=scheduler,
        start_epoch=args.start_epoch, 
        post_pred=post_pred, 
        post_label=post_label
    )

    return accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process finetuning for segmentation")
    parser.add_argument("--config", help="Path to config", required=True)
    _args = parser.parse_args()
    
    with open(_args.config, "r") as f:
        config_data = yaml.safe_load(f)
        
    config_dict = dotdict(config_data)
    main(args=config_dict)
import os
import time
import json
import csv
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.parallel
import torch.utils.data.distributed
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F

from monai.data import decollate_batch
from src.finetune.utils.utils import AverageMeter, distributed_all_gather


def train_epoch(model, loader, optimizer, scheduler, scaler, epoch, loss_func, args):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()

    for idx, batch_data in tqdm(enumerate(loader), desc=f"Processing epoch@{epoch}", total=len(loader)):
        data, target = batch_data["image"], batch_data["label"]
        data, target = data.to(args.device, non_blocking=True), target.to(args.device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        
        with autocast(enabled=args.amp):
            logits = model(data)
            loss = loss_func(logits, target)

        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
            
        if args.is_dist:
        
            loss_list = distributed_all_gather([loss], out_numpy=True, is_valid=idx < loader.sampler.valid_length)
            global_loss = np.mean(loss_list)
            run_loss.update(global_loss, n=args.batch_size * args.world_size)
        else:
            run_loss.update(loss.item(), n=args.batch_size)

        lr = optimizer.param_groups[0]["lr"]
        
        length = max(1, len(loader) // 4)
        if args.rank == 0 and (idx + 1) % length == 0:
            print(f"Epoch: {epoch}/{args.epochs} [{idx+1}/{len(loader)}] | "
                  f"Loss: {run_loss.avg:.4f} | LR: {lr:.8f} | Time: {time.time() - start_time:.2f}s")
            start_time = time.time()
            
    return run_loss.avg
    


def validate_epoch(model, loader, epoch, acc_func, args, model_inferer=None, post_label=lambda x: x, post_pred=lambda x: x):
    """
    THE ONE PERFECT VALIDATION FUNCTION.
    Replaces val_epoch_0, val_epoch, and val_epoch_mean.
    Computes global mean and per-class distinct metrics in a single, DDP-safe pass.
    """
    model.eval()
    device = args.device

    sum_num = None   # [C]
    sum_den = None   # [C]
    
    start_time = time.time()
    
    # inference_mode is strictly faster and uses less memory than no_grad
    with torch.inference_mode():
        for idx, batch_data in tqdm(enumerate(loader), desc="Validating", total=len(loader)):
            data = batch_data["image"].to(device, non_blocking=True)
            target = batch_data["label"].to(device, non_blocking=True)

            with autocast(enabled=args.amp):
                logits = model_inferer(data) if model_inferer is not None else model(data)

            val_labels_list = decollate_batch(target)
            val_labels_convert = [post_label(x) for x in val_labels_list]
            
            val_output_list = decollate_batch(logits)
            val_output_convert = [post_pred(x) for x in val_output_list]

            acc_func.reset()
            acc_func(y_pred=val_output_convert, y=val_labels_convert)
            dice, not_nans = acc_func.aggregate()

            # Guard against precision drift (NaNs)
            dice = torch.nan_to_num(dice.to(device=device, dtype=torch.float64), nan=0.0)
            not_nans = torch.nan_to_num(not_nans.to(device=device, dtype=torch.float64), nan=0.0)

            if dice.ndim == 2:
                num = (dice * not_nans).sum(dim=0)   # [C]
                den = not_nans.sum(dim=0)           # [C]
            else:
                num = dice * not_nans               # [C]
                den = not_nans                      # [C]

            # DDP reduction: sum across ranks
            if args.distributed and torch.distributed.is_initialized():
                torch.distributed.all_reduce(num, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(den, op=torch.distributed.ReduceOp.SUM)

            if sum_num is None:
                sum_num, sum_den = num, den
            else:
                sum_num += num
                sum_den += den

            if args.rank == 0 and (idx + 1) % max(1, len(loader) // 4) == 0:
                tmp = (sum_num / torch.clamp_min(sum_den, 1)).double().cpu().numpy()
                print(f"Val {epoch}/{args.epochs} [{idx+1}/{len(loader)}] | "
                      f"Mean Dice so far: {float(tmp.mean()):.4f} | Time: {time.time() - start_time:.2f}s")
                start_time = time.time()
                
    # Finalize calculations
    per_class = (sum_num / torch.clamp_min(sum_den, 1)).double()
    weights = sum_den.double()
    
    # Mean matching MONAI's MetricReduction.MEAN
    monai_mean = float(sum_num.double().sum().cpu().item() / max(weights.sum().cpu().item(), 1.0))
    
    per_class_np = per_class.cpu().numpy()
    weights_np = weights.cpu().numpy()
    
    class_names = getattr(args, "class_names", [f"class_{i}" for i in range(len(per_class_np))])
    per_class_dict = {class_names[i]: float(per_class_np[i]) for i in range(len(per_class_np))}
    weights_dict = {class_names[i]: int(weights_np[i]) for i in range(len(weights_np))}
    
    val_distinct = json.dumps(
        {"dice": per_class_dict, "n_valid": weights_dict, "monai_mean": monai_mean},
        ensure_ascii=False
    )
    
    # Free memory
    del data, target, logits
    torch.cuda.empty_cache()
    
    return monai_mean, val_distinct
          
def run_training(model, train_loader, val_loader, optimizer, loss_func, acc_func, acc_func_mean, args, model_inferer=None, scheduler=None, start_epoch=0, post_label=None, post_pred=None):
    if args.rank == 0:
        print("\n--- TRAINING INITIATED ---")
        
    outdir = getattr(args, "outdir", "./output")
    
    if args.rank == 0:
        os.makedirs(outdir, exist_ok=True)
        log_path = os.path.join(outdir, "finetune_log.csv")
        log_exists = os.path.exists(log_path) and start_epoch > 0
        log_f = open(log_path, "a" if log_exists else "w", newline="")
        log_writer = csv.writer(log_f)
        if not log_exists:
            log_writer.writerow(["epoch", "train_loss", "val_acc", "val_distinct", "lr", "epoch_time"])
        print(f"Log initialized at {outdir}")
    
    scaler = GradScaler() if args.amp else None
    val_acc_max = 0.0
    best_path = os.path.join(outdir, "best.pt")

    if os.path.exists(best_path):
        prev = torch.load(best_path, map_location="cpu")
        if prev.get("val_acc") is not None:
            val_acc_max = float(prev["val_acc"])
            if args.rank == 0:
                print(f"[Init] Resuming with existing best.pt val_acc = {val_acc_max:.4f}")

    # DRY Helper function for checkpointing
    def save_ckpt(filename, current_epoch, acc, t_loss):
        if args.rank == 0:
            torch.save({
                "epoch": current_epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler else None,
                "scaler": scaler.state_dict() if scaler else None,
                "val_acc": acc,
                "train_loss": t_loss,
                "args": dict(args),
                "rng": {
                    "python": random.getstate(),
                    "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                },
            }, os.path.join(outdir, filename))

    for epoch in range(start_epoch, args.epochs):
        if args.is_dist:
            train_loader.sampler.set_epoch(epoch)
            torch.distributed.barrier() 
        
        epoch_start = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, scaler=scaler, epoch=epoch, loss_func=loss_func, args=args)
        
        if args.rank == 0:
            print(f"Epoch {epoch}/{args.epochs - 1} | Train Loss: {train_loss:.4f} | Time: {time.time() - epoch_start:.2f}s")

        val_acc = None
        val_distinct = ""

        # Validation Phase
        if (epoch + 1) % getattr(args, "val_every", 1) == 0:
            if args.is_dist:
                torch.distributed.barrier()
                
            # CRITICAL FIX: We now ONLY call the single validate_epoch function
            val_acc, val_distinct = validate_epoch(
                model, val_loader, epoch, acc_func, args, 
                model_inferer=model_inferer, post_label=post_label, post_pred=post_pred
            )

            if args.rank == 0:
                print(f"\n=> Final Validation {epoch}/{args.epochs - 1} | Mean ACC: {val_acc:.4f} | Time: {time.time() - epoch_start:.2f}s")
                print(f"Details: {val_distinct}\n")
                
                if val_acc > val_acc_max:
                    print(f"*** New Best Validation Score! ({val_acc_max:.4f} -> {val_acc:.4f}) ***")
                    val_acc_max = val_acc
                    save_ckpt("best.pt", epoch, val_acc, train_loss)

        if args.rank == 0:
            epoch_time = time.time() - epoch_start
            lr = optimizer.param_groups[0]["lr"]
            log_writer.writerow([epoch, train_loss, val_acc if val_acc is not None else "", val_distinct, lr, epoch_time])
            log_f.flush()

            save_ckpt("latest.pt", epoch, val_acc, train_loss)
            
            if (epoch + 1) % getattr(args, "save_every", 1) == 0:
                save_ckpt(f"epoch{epoch+1}.pt", epoch, val_acc, train_loss)

        if scheduler is not None:
            scheduler.step()
        
    if args.rank == 0:
        log_f.close()
        save_ckpt("final.pt", epoch, val_acc, train_loss)
        print(f"\nTraining finished! Best Validation Accuracy: {val_acc_max:.4f}")
    
    return val_acc_max

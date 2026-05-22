import os
import time
import csv
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.distributed as dist  
from torch.cuda.amp import GradScaler, autocast

from src.finetune.utils.utils import AverageMeter, distributed_all_gather

def resize(img):
    #input: (b,_,c,h,w)
    size = 256
    b, _, c, h, w = img.size()
    new_img = []
    for i in range(b):
        im = img[i, :, :, :, :]
        im = F.interpolate(im, size=[size, size], mode='bilinear', align_corners=True) #[C,H,W] -> [C, 256, 256]
        new_img.append(im.unsqueeze(0)) # [1,N,C,256,256]
    new_img = torch.cat(new_img, dim=0) #[B,N,C,256,256]
    return new_img


def train_epoch(model, loader, optimizer, scheduler, scaler, epoch, args):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    loss_func = torch.nn.CrossEntropyLoss()

    for idx, batch_data in tqdm(enumerate(loader), desc=f"Processing epoch@{epoch}", total=len(loader)):
        data, target = batch_data["image"], batch_data["label"]
        
        # CRITICAL FIX: Move data to GPU *before* resizing for faster processing
        data, target = data.to(args.device, non_blocking=True), target.to(args.device, non_blocking=True)
        data = resize(data) # [B, N, C, 256, 256]

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
            # CRITICAL FIX: Actually use the gathered losses for the true average
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
    


def val_epoch(model, loader, epoch, args):
    model.eval()
    run_acc = AverageMeter()
    start_time = time.time()

    with torch.inference_mode():
        for idx, batch_data in tqdm(enumerate(loader), desc="Validating..", total=len(loader)):
            data, target = batch_data["image"], batch_data["label"]

            data, target = data.to(args.device, non_blocking=True), target.to(args.device, non_blocking=True)
            data = resize(data)

            with autocast(enabled=args.amp):
                logits = model(data)
            
            preds = logits.argmax(dim=1)

            # Local counts
            local_correct = (preds == target).sum().to(dtype=torch.float32)
            local_count = torch.tensor(float(target.numel()), device=local_correct.device)

            is_valid = None
            if args.distributed and hasattr(loader, "sampler") and hasattr(loader.sampler, "valid_length"):
                is_valid = torch.tensor(idx < loader.sampler.valid_length, dtype=torch.bool, device=local_correct.device)
            
            if args.distributed:
                # Gather from all ranks
                gathered = distributed_all_gather([local_correct, local_count], out_numpy=False, is_valid=is_valid)

                # Reduce to global scalars on *every* rank
                all_correct = torch.stack([t for t in gathered[0]], dim=0).sum()
                all_count = torch.stack([t for t in gathered[1]], dim=0).sum()
            else:
                all_correct = local_correct
                all_count = local_count
            
            # For this batch across all GPUs
            step_acc = (all_correct / torch.clamp_min(all_count, 1.0)).item() 
            run_acc.update(step_acc, n=int(all_count.item()))

    epoch_acc = float(run_acc.avg)
    
    if args.rank == 0:
        print(f"[VAL] Epoch={epoch} | Acc={epoch_acc:.4f} | Time={time.time()-start_time:.2f}s")
    
    del data, target, logits
    torch.cuda.empty_cache()
    
    return epoch_acc



def run_training(model, train_loader, val_loader, optimizer, args, scheduler=None, start_epoch=0):
    if args.rank == 0:
        print("\n--- CLASSIFICATION TRAINING INITIATED ---")

    outdir = args.outdir
    
    if args.rank == 0:
        os.makedirs(outdir, exist_ok=True)
        log_path = os.path.join(outdir, "finetune_log.csv")
        log_exists = os.path.exists(log_path) and start_epoch > 0
        log_f = open(log_path, "a" if log_exists else "w", newline="")
        log_writer = csv.writer(log_f)
        if not log_exists:
            log_writer.writerow(["epoch", "train_loss", "val_acc", "lr", "epoch_time"])
        print(f"Log initialized at {outdir}")
        
    scaler = GradScaler() if args.amp else None
    val_acc_max = 0.0

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
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            dist.barrier() 
        
        epoch_start = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, scaler=scaler, epoch=epoch, args=args)

        val_acc = None

        if (epoch + 1) % getattr(args, "val_every", 1) == 0:
            if args.distributed:
                dist.barrier()
            
            val_acc = val_epoch(model, val_loader, epoch=epoch, args=args)

            if args.rank == 0:
                print(f"=> Validation {epoch}/{args.epochs-1} | Acc: {val_acc:.4f} | Time: {time.time() - epoch_start:.2f}s")
                
                if val_acc > val_acc_max:
                    print(f"*** New Best Validation Score! ({val_acc_max:.4f} -> {val_acc:.4f}) ***")
                    val_acc_max = val_acc
                    save_ckpt("best.pt", epoch, val_acc, train_loss)

        if args.rank == 0:
            epoch_time = time.time() - epoch_start
            lr = optimizer.param_groups[0]["lr"]
            log_writer.writerow([epoch, train_loss, val_acc if val_acc is not None else "", lr, epoch_time])
            log_f.flush()

            save_ckpt("latest.pt", epoch, val_acc, train_loss)
            
            if (epoch + 1) % getattr(args, "save_every", 1) == 0:
                save_ckpt(f"epoch{epoch+1}.pt", epoch, val_acc, train_loss)

        if scheduler is not None:
            scheduler.step()
        
    if args.rank == 0:
        log_f.close()
        save_ckpt("final.pt", epoch, val_acc, train_loss)

    # Safely broadcast the final best accuracy to all workers
    if args.distributed:
        t = torch.tensor([val_acc_max], device=args.device, dtype=torch.float32)
        dist.broadcast(t, src=0)
        val_acc_max = float(t.item())

    if args.rank == 0:
        print(f"\nTraining finished! Best Accuracy: {val_acc_max:.4f}")

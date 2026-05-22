from __future__ import annotations
import os, yaml, time
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.pretrain.utils.patch_utils import apply_twofold
from src.pretrain.model.twofold_ssl import TwoFoldSSL
from src.pretrain.utils.losses import TwoFoldLoss
from src.common_utils.utils import make_optimizer, make_scheduler_cosine, count_params
from  src.finetune.utils.utils  import AverageMeter, distributed_all_gather
from src.pretrain.utils.memo import activation_sizes
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import csv

import random

from datetime import timedelta
from fvcore.nn import FlopCountAnalysis


def record_memory(tag,args):
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
 
def train_epoch(model, loader, optimizer, scheduler, scaler, epoch, args):
    model.train()
    start_time = time.time()
    
    run_loss = AverageMeter()
    rec_meter = AverageMeter()
    mask_meter = AverageMeter()
    rot_meter = AverageMeter()
   
    fwd_only_peak_gib = None
    crit = TwoFoldLoss(
            recon_variant="masked_only_charbonnier",
            lambda_rec=args.lambda_rec, lambda_mask=args.lambda_mask, lambda_rot=args.lambda_rot,
            use_uncertainty=False
        )
    for idx, batch_data in tqdm(enumerate(loader), desc=f"Processing epoch@{epoch}", total = len(loader)):

    

        data = batch_data["image"]
        
        
        data = data.cuda(args.rank)
        n = args["patch_grid"]
        p = args["mask_rate"]
        K = args["rotations"]
        roi = args.get("roi", 64)
        
        optimizer.zero_grad(set_to_none=True)
                
        x_tilde, M, R = apply_twofold(x=data, n=n, p_mask=p, K=K)
        
        if epoch == 0 and idx == 0:
            model_eval_state = model.training
            model.eval()
            with torch.no_grad():
                flops_batch = FlopCountAnalysis(model, x_tilde).total()
            model.train(model_eval_state)

            bs = x_tilde.size(0)
            flops_per_sample = flops_batch / bs

            print(f"[FLOPs] Forward FLOPs per batch: {flops_batch/1e9:.3f} GFLOPs")
            print(f"[FLOPs] Forward FLOPs per sample: {flops_per_sample/1e9:.3f} GFLOPs")

        if epoch == 0:
            torch.cuda.reset_peak_memory_stats(args.device)

            with torch.no_grad():
                _ = model(x_tilde)

            fwd_only_peak_bytes = torch.cuda.max_memory_allocated(args.device)
            fwd_only_peak_gib = fwd_only_peak_bytes / (1024 **3)

            print( f"[Memory profiling] Forward only peak memory:", f"{fwd_only_peak_gib:.2f} GiB")

        torch.cuda.reset_peak_memory_stats(args.device)

        with autocast(enabled=args.amp):
                    x_hat, m_logits, r_logits, z = model(x_tilde)
                    out = crit(x_hat, data, m_logits, M, r_logits, R, grid_shape=(n, n, n))

                    loss = out["loss_total"]
                    L_mask = out["loss_mask"]
                    L_rot = out["loss_rot"]
                    L_rec = out["loss_rec"]


        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
                

        full_peak_bytes = torch.cuda.max_memory_allocated(args.device)
        full_peak_gib = full_peak_bytes / (1024**3)

        if epoch == 0 and fwd_only_peak_gib is not None:
            activation_mem_gib = full_peak_gib - fwd_only_peak_gib
            print(
                f"[Memory profiling] Full (fwd+bwd) peak: {full_peak_gib:.2f} GiB\n"
                f"[Memory profiling] Activation+grad memory (approx): "
                f"{activation_mem_gib:.2f} GiB"
            )

        if args.is_dist:
            loss_list = distributed_all_gather([loss], out_numpy=True )
            L_rec_list = distributed_all_gather([L_rec], out_numpy=True)
            L_mask_list = distributed_all_gather([L_mask], out_numpy=True)
            L_rot_list = distributed_all_gather([L_rot], out_numpy=True)
            run_loss.update(
                np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0), n=args.batch_size * args.world_size
            )

            rec_meter.update(
                np.mean(np.mean(np.stack(L_rec_list, axis=0), axis=0), axis=0), n=args.batch_size * args.world_size
            )

            rot_meter.update(
                np.mean(np.mean(np.stack(L_rot_list, axis=0), axis=0), axis=0), n=args.batch_size * args.world_size
            )          

            mask_meter.update(
                np.mean(np.mean(np.stack(L_mask_list, axis=0), axis=0), axis=0), n=args.batch_size * args.world_size
            )          


        else:
            run_loss.update(loss.item(), n=args.batch_size)
        
        lr = optimizer.param_groups[0]['lr']
        
        if scheduler is not None:
            scheduler.step()
        
    if args.rank == 0:
        print(
            f"Epoch [{epoch}/{args.epochs}]  "
            f"Loss: {run_loss.avg:.4f} | "
            f"L_rec: {rec_meter.avg:.4f} | "
            f"L_mask: {mask_meter.avg:.4f} | "
            f"L_rot: {rot_meter.avg:.4f} | "
            f"LR: {lr:.6f} | "
            f"Time: {time.time() - start_time:.2f}s"
        )
    start_time = time.time()   
        
    for param in model.parameters():
        param.grad = None
        
    return run_loss.avg, rec_meter.avg, mask_meter.avg, rot_meter.avg
    


def run_training(model, train_loader, optimizer, args, scheduler = None, start_epoch=0):
    
    outdir = args["outdir"]
    os.makedirs(outdir,exist_ok=True)
    log_path = os.path.join(outdir,"pretrain_log.csv")
    log_exists = os.path.exists(log_path) 
    
    if args.rank == 0:
        log_f = open(log_path,"a" if log_exists else "w", newline="")
        log_writer = csv.writer(log_f)    
        if not log_exists:
            log_writer.writerow(["epoch","train_loss",f"recon loss_{args["lambda_rec"]}", f"mask loss_{args["lambda_mask"]}", f"rotation loss_{args["lambda_rot"]}","lr","epoch_time"])
        print("log created at",args.outdir)
    
    scaler = GradScaler() if args.amp else None    
    train_loss_min = float('inf')
    
    for epoch in range(start_epoch,args.epochs):
        if args.is_dist:
            train_loader.sampler.set_epoch(epoch)
            torch.distributed.barrier()
        if args.rank == 0  and epoch == start_epoch:
            total_params = sum(p.numel() for p in model.parameters())
            print("params:", total_params)
            
        epoch_start = time.time()
        train_loss, rec_loss, mask_loss, rot_loss = train_epoch(
            model, train_loader, optimizer, scheduler=scheduler, scaler=scaler, epoch=epoch, args=args
        )
        
        epoch_time = time.time() - epoch_start
        if args.rank ==0:
            args.total_train_time += epoch_time

        if args.rank == 0:
            total_readable = str(timedelta(seconds=int(args.total_train_time)))
            print(
                "Epoch {}/{} ".format(epoch, args.epochs - 1),
                "loss: {:.4f}".format(train_loss),
                "time {:.2f}s".format(time.time() - epoch_start),
                "total_time {}".format(total_readable),
            )
            log_writer.writerow([epoch, train_loss, rec_loss, mask_loss, rot_loss, optimizer.param_groups[0]['lr'], time.time() - epoch_start])
            log_f.flush()
            
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler else None,
                    "scaler": scaler.state_dict() if scaler else None,
                    "train_loss": train_loss,
                    "args": dict(args),
                    "total_train_time":args.total_train_time,
                    "rng": {
                            "python": random.getstate(),
                            "numpy": np.random.get_state(),
                            "torch": torch.get_rng_state(),
                            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                        },
                },
                os.path.join(outdir,"latest.pt")
            )
            
            if train_loss < train_loss_min:
                torch.save(
                    {
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict() if scheduler else None,
                        "scaler": scaler.state_dict() if scaler else None,
                        "train_loss": train_loss,
                        "args": dict(args),
                        "total_train_time":args.total_train_time,
                        "rng": {
                            "python": random.getstate(),
                            "numpy": np.random.get_state(),
                            "torch": torch.get_rng_state(),
                            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                        },
                    },
                    os.path.join(outdir,"best.pt")
                )
                train_loss_min = train_loss
            
            if epoch > 0 and (epoch+1)%args.save_every == 0:
                torch.save(
                    {
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict() if scheduler else None,
                        "scaler": scaler.state_dict() if scaler else None,
                        "train_loss": train_loss,
                        "args": dict(args),
                        "total_train_time":args.total_train_time,
                        "rng": {
                            "python": random.getstate(),
                            "numpy": np.random.get_state(),
                            "torch": torch.get_rng_state(),
                            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                        },
                    },
                    os.path.join(outdir,f"epoch{epoch+1}.pt")
                )
    

        if scheduler is not None:
            scheduler.step()
            
    if args.rank == 0:
        log_f.close()
        print("Training finished.")
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler else None,
                "scaler": scaler.state_dict() if scaler else None,
                "train_loss": train_loss,
                "args": dict(args),
                "total_train_time":args.total_train_time,
                "rng": {
                            "python": random.getstate(),
                            "numpy": np.random.get_state(),
                            "torch": torch.get_rng_state(),
                            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                        },
            },
            os.path.join(outdir,"final.pt")
        )
        
    
    return train_loss_min


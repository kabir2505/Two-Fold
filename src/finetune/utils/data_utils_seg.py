
import math
import os
import pickle
import numpy as np
import torch
import itertools as it
from monai import data, transforms
from monai.data import *
from monai import data, transforms
from monai.data import *
import random
import json

def datafold_read_0(datalist, basedir, fold=0, key="training"):
    with open(datalist) as f:
        json_data = json.load(f)

    json_data = json_data[key]

    for d in json_data:
        for k, v in d.items():
            if isinstance(d[k], list):
                d[k] = [os.path.join(basedir, iv) for iv in d[k]]
            elif isinstance(d[k], str):
                d[k] = os.path.join(basedir, d[k]) if len(d[k]) > 0 else d[k]

    tr = []
    val = []
    for d in json_data:
        if "fold" in d and d["fold"] == fold:
            val.append(d)
        else:
            tr.append(d)

    return tr, val



import json, os
from copy import deepcopy

def datafold_read(datalist, basedir):
    with open(datalist) as f:
        json_data = json.load(f)

    train_key = "training" if "training" in json_data else "train"
    val_key = "validation" if "validation" in json_data else "val"

    tr = deepcopy(json_data.get(train_key, []))
    val = deepcopy(json_data.get(val_key, []))

    def _absify(items):
        for d in items:
            for k, v in d.items():
                if isinstance(v, list):
                    d[k] = [os.path.join(basedir, iv) if iv else iv for iv in v]
                elif isinstance(v, str):
                    d[k] = os.path.join(basedir, v) if v else v
        return items

    tr = _absify(tr)
    val = _absify(val)

    return tr, val


class Sampler(torch.utils.data.Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, make_even=True):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.shuffle = shuffle
        self.make_even = make_even
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        indices = list(range(len(self.dataset)))
        self.valid_length = len(indices[self.rank : self.total_size : self.num_replicas])

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        if self.make_even:
            if len(indices) < self.total_size:
                if self.total_size - len(indices) < len(indices):
                    indices += indices[: (self.total_size - len(indices))]
                else:
                    extra_ids = np.random.randint(low=0, high=len(indices), size=self.total_size - len(indices))
                    indices += [indices[ids] for ids in extra_ids]
            assert len(indices) == self.total_size
        indices = indices[self.rank : self.total_size : self.num_replicas]
        self.num_samples = len(indices)
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch



import numpy as np
import torch
from collections import Counter


import numpy as np
import torch


def sanity_check_lits_labels(dataset, tag="", num_samples=1):
    """
    Handles MONAI datasets where __getitem__ returns:
    - dict (normal case)
    - list[dict] (RandCropByPosNegLabeld case)
    """

    print(f"\n[LiTS Label Sanity Check] {tag}")

    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]

        # Case 1: RandCropByPosNegLabeld → list of dicts
        if isinstance(sample, list):
            samples = sample
        else:
            samples = [sample]

        for j, s in enumerate(samples):
            label = s["label"]

            if torch.is_tensor(label):
                label = label.detach().cpu().numpy()

            unique, counts = np.unique(label, return_counts=True)
            stats = dict(zip(unique.tolist(), counts.tolist()))

            print(f" Sample {i}, crop {j}:")
            print(f"   Unique labels: {unique}")
            print(f"   Counts: {stats}")

def get_loader(args,train_transform,val_transform):
    data_dir = args.data_dir
    datalist_json = os.path.join(data_dir, args.json_list)
  
    num_workers = args.workers
    PICKLE_PROTOCOL_SAFE = 2
    def seed_worker(worker_id: int):
        # torch.initial_seed() is unique per worker; condense to 32-bit for numpy/random
        worker_seed = (torch.initial_seed() % 2**32)
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        # Optional: also seed torch within the worker context explicitly
        torch.manual_seed(worker_seed)

    # A generator fed to DataLoader for reproducible PyTorch RNG in workers
    g = torch.Generator()
    g.manual_seed(int(args["seed"]) + int(args.rank))  # rank-shifted

    if args.test_mode:
        test_files = load_decathlon_datalist(datalist_json, True, "validation", base_dir=data_dir)
        test_ds = PersistentDataset(data=test_files,
                                     transform=val_transform,
                                     pickle_protocol=PICKLE_PROTOCOL_SAFE,
                                     cache_dir=args.cache_dir)
        test_sampler = Sampler(test_ds, shuffle=False) if args.distributed else None
        test_loader = data.DataLoader(
            test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=args.workers,
            sampler=test_sampler,
            pin_memory=True,
            persistent_workers=True,
        )
        loader = test_loader
        print("test mode on")
    else:

        if args.datafold_read:
            datalist,_ = datafold_read(datalist=datalist_json, basedir=data_dir)
        else:
            datalist = load_decathlon_datalist(datalist_json, True, "training", base_dir=data_dir)
        print(f"[Sanity Check] Number of training samples loaded: {len(datalist)}")
        if args.use_normal_dataset:
            print('use persistent')
            train_ds = PersistentDataset(data=datalist,
                                     transform=train_transform,
                                     pickle_protocol=PICKLE_PROTOCOL_SAFE,
                                     cache_dir=args.cache_dir)
            # train_ds = data.Dataset(data=datalist, transform=train_transform)
        else:
            train_ds = data.CacheDataset(
                data=datalist, transform=train_transform, cache_num=24, cache_rate=1.0, num_workers=args.workers
            )
        
        sanity_check_lits_labels(train_ds, tag="TRAIN (after transforms)", num_samples=2)

        print(f"[Sanity Check] Number of training dataset items: {len(train_ds)}")
        train_sampler = Sampler(train_ds) if args.distributed else None
        train_loader = data.DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            num_workers=args.workers,
            sampler=train_sampler,
            pin_memory=True,
            worker_init_fn=seed_worker, generator = g,
            persistent_workers=True if num_workers > 0 else False
        )

        print(f"[Sanity Check] Number of batches in training loader: {len(train_loader)}")
        print(f"[Sanity Check] Batch size: {args.batch_size}")
        

        print("Train", datalist[:5])
        print("len(train)", len(datalist))
        
        import sys
    
        if args.datafold_read:
            _,val_files = datafold_read(datalist=datalist_json, basedir=data_dir)
            print("val_files", val_files[:5])
            print("len of val is " , len(val_files))
        else:
            val_files = load_decathlon_datalist(datalist_json, True, "validation", base_dir=data_dir)
           # print("val_files", val_files)
        #subset_size = 2
        # val_files = val_files[:8]

        
        # val_ds = data.Dataset(data=val_files, transform=val_transform)
        val_ds = PersistentDataset(data=val_files,
                                     transform=val_transform,
                                     pickle_protocol=PICKLE_PROTOCOL_SAFE,
                                     cache_dir=args.cache_dir)
        val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
        print(f"[Sanity Check] Number of validation samples loaded: {len(val_files)}")
        print(f"[Sanity Check] Number of validation dataset items: {len(val_ds)}")

        val_loader = data.DataLoader(
            val_ds, batch_size=1, shuffle=False, num_workers=args.workers, sampler=val_sampler, pin_memory=True,
            worker_init_fn=seed_worker, generator = g, persistent_workers=True if num_workers > 0 else False
        )

        print(f"[Sanity Check] Number of batches in validation loader: {len(val_loader)}")
        print("[Sanity Check] Data Setup complete.\n")
        loader = [train_loader, val_loader]
    
        print("len(val_ds)", len(val_ds))
        print("len(val_loader)", len(val_loader))
        print("len(train_ds)", len(train_ds))
        print("len(train_loader)", len(train_loader))
    print(f"returning a list")
    return loader

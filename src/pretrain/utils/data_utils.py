from torch.utils.data import Dataset as _TorchDataset
from torch.utils.data import Subset
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from monai.data import *
import pickle
from monai.transforms import *
from math import *
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
    ScaleIntensityRanged, CropForegroundd, SpatialPadd,
    RandSpatialCropd, CenterSpatialCropd,
    RandFlipd, RandRotate90d, RandShiftIntensityd, ToTensord
)
import os
import torch
def get_loader(args):
    is_main_process = (not args.distributed) or (args.rank == 0)
    splits1 = "dataset.json"
    splits2 = "dataset.json"
    splits3 = "dataset_LUNA16_0.json"

    splits4 = "dataset_clean.json"

    datadir1 = args.data_btcv
    datadir2 = args.data_tcia
    datadir3 = args.data_luna

    jsonlist1 =  os.path.join(datadir1, "dataset.json")
    jsonlist2 =  os.path.join(datadir2, splits2)
    jsonlist3 =  os.path.join(datadir3, splits3)
    

    
    num_workers = args.workers
    
    datalist1 = load_decathlon_datalist(jsonlist1, False, "training", base_dir=datadir1)
    print("Dataset 1 BTCV: number of data: {}".format(len(datalist1)))
    new_datalist1 = []
    for item in datalist1:
        item_dict = {"image": item["image"]}
        new_datalist1.append(item_dict)
    
    print('new_datalist1',new_datalist1)

    datalist2 = load_decathlon_datalist(jsonlist2, False, "training", base_dir=datadir2)
    print("Dataset 2 Covid 19: number of data: {}".format(len(datalist2)))

    print('new_datalist1',new_datalist1)
    
    datalist3 = load_decathlon_datalist(jsonlist3, False, "training", base_dir=datadir3)
    print("Dataset 3 Luna: number of data: {}".format(len(datalist3)))
    new_datalist3 = []
    for item in datalist3:
        item_dict = {"image": item["image"]}
        new_datalist3.append(item_dict)

    print('new_datalist3',new_datalist3)
    
    vallist1 = load_decathlon_datalist(jsonlist1, False, "validation", base_dir=datadir1)
    vallist2 = load_decathlon_datalist(jsonlist2, False, "validation", base_dir=datadir2)
    vallist3 = load_decathlon_datalist(jsonlist3, False, "validation", base_dir=datadir3)

    print(f"\nDataset 1 BTCV: number of data: {len(datalist1)}")
    print(f"Dataset 2 COVID19: number of data: {len(datalist2)}")
    #print(f"Dataset 3 LUNA16: number of data: {len(datalist3)}")



    if args.split == "btcv_only":
        datalist = new_datalist1
        val_files = vallist1
    elif args.split == "8k":
        datalist = new_datalist1
        datalist += datalist2
        val_files = vallist1
        val_files += vallist2 
    
    if args.split == "16k":
        datalist = new_datalist1
        datalist += datalist2
        datalist = datalist + new_datalist3

        val_files = vallist1
        val_files += vallist2
        val_files = val_files + vallist3

 
    print(f"Total training samples: {len(datalist)}")
    print(f"Total validation samples: {len(val_files)}")
         
    
    print("Dataset all training: number of data: {}".format(len(datalist)))
    print("Dataset all validation: number of data: {}".format(len(val_files)))

  
    roi = (args.roi_x, args.roi_y, args.roi_z)  # expected (96, 96, 96)

    # ---------- TRANSFORM SPLIT ----------
    # 1) deterministic (cacheable): IO, geometry, intensity, foreground, pad
    cache_tf_train = Compose([
        LoadImaged(keys=["image"], image_only=True, dtype=np.int16),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=args.a_min, a_max=args.a_max,
            b_min=args.b_min, b_max=args.b_max, clip=True
        ),
        CropForegroundd(keys=["image"], source_key="image"),
        SpatialPadd(keys=["image"], spatial_size=roi),
    ])
    

    # 2) random (runtime): fresh every __getitem__
    runtime_tf_train = Compose([
        # exact-size ROI (random location) => guarantees fresh crop every call
        RandSpatialCropd(keys=["image"], roi_size=roi, random_size=False),

        # light augmentations
        RandFlipd(keys=["image"], prob=args.RandFlipd_prob, spatial_axis=0),
        RandFlipd(keys=["image"], prob=args.RandFlipd_prob, spatial_axis=1),
        RandFlipd(keys=["image"], prob=args.RandFlipd_prob, spatial_axis=2),
        RandRotate90d(keys=["image"], prob=args.RandRotate90d_prob, max_k=3),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=args.RandShiftIntensityd_prob),

        ToTensord(keys=["image"]),
    ])
    
    val_transforms = Compose([
        LoadImaged(keys=["image"], image_only=True, dtype=np.int16),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=args.a_min, a_max=args.a_max,
            b_min=args.b_min, b_max=args.b_max, clip=True
        ),
        CropForegroundd(keys=["image"], source_key="image"),
        SpatialPadd(keys=["image"], spatial_size=roi),   # ensure >= roi
        CenterSpatialCropd(keys=["image"], roi_size=roi),# EXACT roi size, centered
        ToTensord(keys=["image"]),
    ])
    
    
    
    
    if args.cache_dataset:
        print("Using MONAI CacheDataset (cache deterministic only)")
        cache_rate = getattr(args, "cache_rate", 0.5)
        base_train = CacheDataset(data=datalist, transform=cache_tf_train,
                                  cache_rate=cache_rate, num_workers=num_workers)
    elif args.smartcache_dataset:
        print("Using MONAI SmartCacheDataset (cache deterministic only)")
        base_train = SmartCacheDataset(
            data=datalist,
            transform=cache_tf_train,
            replace_rate=getattr(args, "replace_rate", 0.25),   # smoother rolling refresh
            cache_num=getattr(args, "cache_num", 2048),         # tune to your RAM
        )
    else:
        print("Using MONAI PersistentDataset (cache deterministic only, on disk)")
        # train_ds = Dataset(data=datalist, transform=train_transforms)
        base_train = PersistentDataset(
            data=datalist,
            transform=cache_tf_train,
            pickle_protocol=pickle.HIGHEST_PROTOCOL,
            cache_dir=args.cache_dir,
        )

    # Wrap with runtime randoms so crops/augs are NEW every access
    train_ds = Dataset(data=base_train, transform=runtime_tf_train)

    g = torch.Generator()
    seed = int(getattr(args, "seed", 42))
    rank = int(getattr(args, "rank", 0))
    g.manual_seed(seed + rank)
    
    if args.distributed:
        if is_main_process: print("Using DDP train sampler")
        train_sampler = DistributedSampler(
            dataset=train_ds, 
            even_divisible=True, 
            shuffle=True, 
            seed=seed
        )
    else:
        train_sampler = None


    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, num_workers=num_workers, sampler=train_sampler,
        drop_last=True, pin_memory=True, persistent_workers=True if num_workers > 0 else False,generator=g
    )

    val_ds = PersistentDataset(data=val_files,
                               transform=val_transforms,
                               pickle_protocol=pickle.HIGHEST_PROTOCOL,
                               cache_dir=args.cache_dir)
    
    if args.distributed and args.rank == 0:
        print('val_ds',len(val_ds))
        print('train_ds',len(train_ds))
    
    if is_main_process:
            print(f"[RANK {rank}] Validating with {len(val_ds)} samples")
            
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=num_workers, shuffle=False, drop_last=True, persistent_workers=True if num_workers > 0 else False)

    return train_loader, val_loader



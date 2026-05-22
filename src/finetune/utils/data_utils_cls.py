import monai
import math
import os
import pickle
import numpy as np
import torch

from monai.data import DistributedSampler
from monai import data, transforms
from monai.data import *
import pandas as pd
import random



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




def get_loader(args):
    '''Get the dataloader for the CCII dataset.'''
    # Transforms
    def __transforms__(augmentation=True, npy=None, args=None):
        RANDOM_BRIGHTNESS = 7
        RANDOM_CONTRAST = 5
        pre_size = 420
        final_size = 384
        spatial_limit = int((pre_size-final_size)/2.0)
        # pre_top_left = int((512-pre_size)/2.0)
        final_top_left = int((512-final_size)/2.0)

        npy_normalized = npy.astype(np.float32) / 255.0 # cast to float
        if augmentation:
            # random flip
            if random.uniform(0, 1) < 0.5: #horizontal flip
                npy_normalized = np.flipud(npy_normalized)
            # color jitter
            br = random.randint(-RANDOM_BRIGHTNESS, RANDOM_BRIGHTNESS) / 100.
            npy_normalized = npy_normalized + br
            # Random contrast
            cr = 1.0 + random.randint(-RANDOM_CONTRAST, RANDOM_CONTRAST) / 100.
            npy_normalized = npy_normalized * cr
            # clip values to 0-1 range
            npy_normalized = np.clip(npy_normalized, 0, 1.0)
            # random crop
            offset_x = random.randint(-spatial_limit, spatial_limit)
            offset_y = random.randint(-spatial_limit, spatial_limit)
            npy_normalized = npy_normalized[
                :,
                final_top_left+offset_x : final_top_left+final_size+offset_x,
                final_top_left+offset_y : final_top_left+final_size+offset_y
                ]
        else:
            npy_normalized = npy_normalized[
                :,
                final_top_left : final_top_left+final_size,
                final_top_left : final_top_left+final_size
                ]
        return npy_normalized
    
    import os
    train_files_name = os.path.join(args.csv_list, f'CC_CCII_fold{args.fold}_train.csv')
    val_files_name = os.path.join(args.csv_list, f'CC_CCII_fold{args.fold}_valid.csv')
    print(train_files_name)
    print(val_files_name)

    import os
    import pandas as pd

    print("DEBUG: csv_list =", repr(args.csv_list))
    print("DEBUG: fold =", repr(args.fold))

    train_files_name = os.path.join(args.csv_list, f'CC_CCII_fold{args.fold}_train.csv')
    val_files_name   = os.path.join(args.csv_list, f'CC_CCII_fold{args.fold}_valid.csv')

    print("DEBUG: train_files_name =", os.path.abspath(train_files_name))
    print("DEBUG: val_files_name   =", os.path.abspath(val_files_name))
    print("DEBUG: exists(train) =", os.path.exists(train_files_name))
    print("DEBUG: exists(val)   =", os.path.exists(val_files_name))

    train_files = pd.read_csv(train_files_name)
    val_files = pd.read_csv(val_files_name)

    train_ds = CC_CCII(data=train_files, transforms=__transforms__, augmentation=True, args=args)
    print(f'=>Train len {len(train_ds)}')
    #train_sampler = monai.data.DistributedSampler(train_ds, num_replicas = args.world_size, rank = args.rank, shuffle=True)
    train_sampler = Sampler(train_ds) if args.distributed else None
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size,
        num_workers=8, pin_memory=True, persistent_workers=True, sampler = train_sampler, shuffle = (train_sampler is None)
    )
    
    val_ds = CC_CCII(data=val_files, transforms=__transforms__, augmentation=False,args=args)
    print(f'=>Val len {len(val_ds)}')
    #val_sampler = monai.data.DistributedSampler(val_ds, num_replicas = args.world_size, rank = args.rank, shuffle=False)
    val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, persistent_workers=True,sampler = val_sampler)
    return train_loader, val_loader


class CC_CCII(torch.utils.data.Dataset):
    '''CC_CCII Covid-19 classification dataset.
    This dataset is used for Covid-19 classification.
    It loads the data from the given directory and csv file.
    The data is preprocessed and augmented using various techniques.
    http://ncov-ai.big.ac.cn/download?lang=en
    '''
    def __init__(self, data=None, transforms=None, augmentation=True, args=None):
        super().__init__()
        self.augmentation = augmentation
        self.df_meta = pd.read_csv(os.path.join(args.csv_list, 'CC_CCII_metadata.csv'))

        df = data
        self.patients = df['patient_id']
        self.scans = df['scan_id']
        self.targets = df['target']
        self.transforms = transforms
        self.args = args

    def __getitem__(self, index):
        target = int(self.targets[index])
        npy = np.load(
            os.path.join(
                self.args.data_dir,
                'p'+str(self.patients[index])+'-s'+str(self.scans[index])+'.npy'
                )
            )

        meta = self.df_meta[(self.df_meta['patient_id'] == self.patients[index])]
        covariates = [
            'Age',
            'Sex(Male1/Female2)',
            'Critical_illness',
            'Liver_function',
            'Lung_function',
            'Progression (Days)'
        ]
        if meta.size == 0:
            meta = np.array([47, 1.5, 0, 1, 2, 6.89],dtype='f8')
        else:
            meta = meta.sample(frac=1.0, replace=True, weights=None, random_state=0, axis=0)
            meta = np.squeeze(meta[covariates].to_numpy(), axis=0)
        meta[0] = np.clip(meta[0] / 100, 0.25, 0.95)
        meta[1] = meta[1] - 1
        meta[3] = meta[3] / 5
        meta[4] = meta[4] / 5
        meta[-1] = meta[-1] / 14

        npy_normalized = self.transforms(self.augmentation, npy, self.args)
        npy_normalized = npy_normalized[np.newaxis,]
        return {
            'image': npy_normalized,
            'label': target
        }

    def __len__(self):
        return len(self.targets)

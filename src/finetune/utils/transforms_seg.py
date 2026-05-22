from monai import data, transforms
from monai.data import *
from monai import data, transforms
from monai.data import *
from monai.transforms import *
import numpy as np
import torch


class Convert_WHS_label(MapTransform):

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            out = d[key].clone()
            out[d[key] == 205] = 1
            out[d[key] == 420] = 2
            out[d[key] == 500] = 3
            out[d[key] == 550] = 4
            out[d[key] == 600] = 5
            out[d[key] == 820] = 6
            out[d[key] == 850] = 7

            d[key] = out.float()
        return d

def get_transforms_btcv(args):
    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.EnsureChannelFirstd(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            transforms.Spacingd(
                keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest")
            ),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            
            transforms.RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                pos=3,
                neg=1,
                num_samples=args.sw_batch_size,
                image_key="image",
                image_threshold=0,
            ),
            transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=2),
            transforms.RandRotate90d(keys=["image", "label"], prob=args.RandRotate90d_prob, max_k=3),
            transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=args.RandScaleIntensityd_prob),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=args.RandShiftIntensityd_prob),
            
            # transforms.ToTensord(keys=["image", "label"]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.EnsureChannelFirstd(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            transforms.Spacingd(
                keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest")
            ),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            # transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    return train_transform,val_transform
    
    
def get_transforms_mmwhs(args):
    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.EnsureChannelFirstd(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            transforms.Spacingd(
                keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest")
            ),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            Convert_WHS_label(keys="label"),
            transforms.RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                pos=9,
                neg=1,
                num_samples=args.sw_batch_size,
                image_key="image",
                image_threshold=0,
            ),
            transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=2),
            transforms.RandRotate90d(keys=["image", "label"], prob=args.RandRotate90d_prob, max_k=3),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.EnsureChannelFirstd(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            transforms.Spacingd(
                keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest")
            ),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            Convert_WHS_label(keys="label"),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )
    return train_transform, val_transform


def get_transforms_spleen(args):
    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.EnsureChannelFirstd(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            transforms.Spacingd(
                keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest")
            ),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            transforms.SpatialPadd(keys=["image", "label"], spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                                   mode="constant"),

            transforms.RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                pos=3,
                neg=1,
                num_samples=args.sw_batch_size,
                image_key="image",
                image_threshold=0,
            ),
            transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=2),
            transforms.RandRotate90d(keys=["image", "label"], prob=args.RandRotate90d_prob, max_k=3),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=args.RandShiftIntensityd_prob),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.EnsureChannelFirstd(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            transforms.Spacingd(
                keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest")
            ),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            transforms.SpatialPadd(keys=["image", "label"], spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                                   mode="constant"),
        ]
    )

    return train_transform, val_transform



def merge_lits_liver_tumor(x):
    if not torch.is_tensor(x):
        x = torch.as_tensor(x)
    return (x > 0).to(torch.uint8)



def get_transforms_lits(args):
    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.EnsureChannelFirstd(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            transforms.Spacingd(
                keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest")
            ),

            #merge {1:liver, 2:tumor} -> 1
            transforms.Lambdad(keys="label", func=merge_lits_liver_tumor),

            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            transforms.SpatialPadd(keys=["image", "label"], spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                                   mode="constant"),

            transforms.RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                pos=1,
                neg=1,
                num_samples=args.sw_batch_size,
                image_key="image",
                image_threshold=0,
            ),
            transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=2),
            transforms.RandRotate90d(keys=["image", "label"], prob=args.RandRotate90d_prob, max_k=3),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=args.RandShiftIntensityd_prob),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.EnsureChannelFirstd(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            transforms.Spacingd(
                keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest")
            ),


            transforms.Lambdad(keys="label", func=merge_lits_liver_tumor),

            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            transforms.SpatialPadd(keys=["image", "label"], spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                                   mode="constant"),
        ]
    )

    return train_transform, val_transform




def get_transforms_brats_0(args):
    train_transform = Compose(
    [
        # load 4 Nifti images and stack them together
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        CropForegroundd(
            keys=["image", "label"], source_key="image", k_divisible=[args.roi_x, args.roi_y, args.roi_z]
        ),

        SpatialPadd(keys=["image", "label"], spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                               mode="constant"),

        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(args.roi_x, args.roi_y, args.roi_z),
            pos=1,
            neg=1,
            num_samples=args.sw_batch_size,
            image_key="image",
            image_threshold=0,
        ),
        RandFlipd(keys=["image", "label"], prob=0.1, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.1, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.1, spatial_axis=2),
        ]
    )
    val_transform = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            CropForegroundd(
                keys=["image", "label"], source_key="image", k_divisible=[args.roi_x, args.roi_y, args.roi_z]
            ),

            SpatialPadd(keys=["image", "label"], spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                        mode="constant"),
        ]
    )
    

    return train_transform, val_transform










def get_transforms_brats(args):
    train_transform = Compose(
        [
            # load 4 Nifti images and stack them together
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            CropForegroundd(
                keys=["image", "label"], source_key="image", k_divisible=[args.roi_x, args.roi_y, args.roi_z]
            ),

            SpatialPadd(keys=["image", "label"], spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                        mode="constant"),

            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                pos=1,
                neg=1,
                num_samples=args.sw_batch_size,
                image_key="image",
                image_threshold=0,
            ),
            RandFlipd(keys=["image", "label"], prob=0.1, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.1, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.1, spatial_axis=2),

            Convert_brats_Classesd(keys=["label"])
        ]
    )
    val_transform = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            CropForegroundd(
                keys=["image", "label"], source_key="image", k_divisible=[args.roi_x, args.roi_y, args.roi_z]
            ),

            SpatialPadd(keys=["image", "label"], spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                        mode="constant"),

            RandShiftIntensityd(keys="image", offsets=0.1, prob=0),
            Convert_brats_Classesd(keys=["label"])
        ]
    )

    return train_transform, val_transform


class Convert_brats_Classesd(MapTransform):
    """
        Convert labels to multi channels based on brats18 classes:
        label 1 is the necrotic and non-enhancing tumor core
        label 2 is the peritumoral edema
        label 4 is the GD-enhancing tumor
        The possible classes are TC (Tumor core), WT (Whole tumor)
        and ET (Enhancing tumor).
        """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:

            new_img = d[key].clone()
            new_img[d[key] > 0] = 0

            new_img[d[key] == 1] = 1
            new_img[d[key] == 2] = 2
            new_img[d[key] == 4] = 3

            # print(torch.unique(d[key]), torch.unique(new_img.float()))
            d[key] = new_img.float()


        return d


def get_transforms_amos(args):

    train_transform = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image","label"], pixdim=(args.space_x, args.space_y, args.space_z),
                mode=("bilinear", "nearest")
            ),
            ScaleIntensityRanged(
                keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            RandCropByPosNegLabeld(
                keys=["image","label"],
                label_key="label",
                spatial_size=(args.roi_x,args.roi_y, args.roi_z),
                pos=9,
                neg=1,
                num_samples=args.sw_batch_size,
                image_key="image",
                image_threshold=0,
            ),

            RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=2),
            RandRotate90d(keys=["image", "label"], prob=args.RandRotate90d_prob, max_k=3),

        ]
    )


    val_transform = Compose(
        [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z),
            mode=("bilinear", "nearest")
        ),
        ScaleIntensityRanged(
            keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min,
            b_max=args.b_max, clip=True
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        ]
    )

    return train_transform, val_transform


  


def get_transforms_word(args):

    train_transform = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z),
            mode=("bilinear","nearest")
        ),
        ScaleIntensityRanged(
            keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min,
            b_max=args.b_max, clip=True
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        RandCropByPosNegLabeld(keys=["image", "label"], label_key="label", spatial_size=(args.roi_x, args.roi_y, args.roi_z),
            pos=9,
            neg=1,
            num_samples=args.sw_batch_size,
            image_key="image",
            image_threshold=0,
        ),
        RandFlipd(keys=["image","label"], prob=args.RandFlipd_prob, spatial_axis=0),
        RandFlipd(keys=["image","label"], prob=args.RandFlipd_prob, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=2),
        RandRotate90d(keys=["image", "label"], prob=args.RandRotate90d_prob, max_k=3),
        RandScaleIntensityd(keys="image", factors=0.1, prob=args.RandScaleIntensityd_prob),
        RandShiftIntensityd(keys="image", offsets=0.1,prob=args.RandShiftIntensityd_prob),       
    ]


    val_transform = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image","label"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
    ]

    return train_transform, val_transform


def get_transforms_brats_18(args):
    train_transform = Compose(
    [
        # load 4 Nifti images and stack them together
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        CropForegroundd(
            keys=["image", "label"], source_key="image", k_divisible=[args.roi_x, args.roi_y, args.roi_z]
        ),

        SpatialPadd(keys=["image", "label"], spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                               mode="constant"),

        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(args.roi_x, args.roi_y, args.roi_z),
            pos=1,
            neg=1,
            num_samples=args.sw_batch_size,
            image_key="image",
            image_threshold=0,
        ),
        RandFlipd(keys=["image", "label"], prob=0.1, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.1, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.1, spatial_axis=2),
        ]
    )

    val_transform = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            CropForegroundd(
                keys=["image", "label"], source_key="image", k_divisible=[args.roi_x, args.roi_y, args.roi_z]
            ),

            SpatialPadd(keys=["image", "label"], spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                        mode="constant"),
        ]
    )

    return train_transform, val_transform

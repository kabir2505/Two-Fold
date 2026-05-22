# viz_transforms.py
import os, random, math, pathlib
import numpy as np
import matplotlib.pyplot as plt
from dotdict import dotdict
import torch
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, ScaleIntensityRanged,
    CropForegroundd, SpatialPadd, RandSpatialCropd, RandFlipd, RandRotate90d,
    RandShiftIntensityd, CopyItemsd, ToTensord
)
from monai.data import load_decathlon_datalist
import argparse
import yaml

import nibabel as nib

def _pick_indices(n_total, n_pick, seed=0):
    rng = random.Random(seed)
    n_pick = min(n_pick, n_total)
    idxs = list(range(n_total))
    rng.shuffle(idxs)
    return idxs[:n_pick]

def _slice_from_tensor(x, axis="axial", slice_idx="center"):
    """
    x: torch.Tensor or np.ndarray [1, D, H, W] or [D, H, W]
    returns: 2D numpy slice
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if x.ndim == 3:  # [D,H,W] -> add C=1
        x = x[None, ...]
    assert x.ndim == 4 and x.shape[0] == 1, f"Expected [1,D,H,W], got {x.shape}"
    _, D, H, W = x.shape

    if slice_idx == "center":
        idx_d, idx_h, idx_w = D // 2, H // 2, W // 2
    else:
        idx_d = int(np.clip(slice_idx, 0, D - 1))
        idx_h = int(np.clip(slice_idx, 0, H - 1))
        idx_w = int(np.clip(slice_idx, 0, W - 1))

    axis = axis.lower()
    if axis == "axial":      # z
        return x[0, idx_d, :, :]
    elif axis == "coronal":  # y
        return x[0, :, idx_h, :]
    elif axis == "sagittal": # x
        return x[0, :, :, idx_w]
    else:
        raise ValueError("axis must be one of {'axial','coronal','sagittal'}")

def _nice_vmin_vmax(img, p1=1.0, p99=99.0):
    lo, hi = np.percentile(img, [p1, p99])
    if math.isclose(hi, lo):
        hi = lo + 1e-6
    return float(lo), float(hi)

def _subplot_rows(fig, axs_top, axs_bot, title_left, sl_raw_native, sl_raw_aligned, sl_base, sl_geom, sl_aug):
    # Top row: images (5 columns)
    vmin, vmax = _nice_vmin_vmax(sl_base)  # common window from baseline
    axs_top[0].imshow(sl_raw_native, cmap="gray")
    axs_top[0].set_title(f"{title_left}\nRaw (native)", fontsize=10)
    axs_top[1].imshow(sl_raw_aligned, cmap="gray", vmin=vmin, vmax=vmax)
    axs_top[1].set_title("Raw (aligned ROI)", fontsize=10)
    axs_top[2].imshow(sl_base, cmap="gray", vmin=vmin, vmax=vmax)
    axs_top[2].set_title("Baseline", fontsize=10)
    axs_top[3].imshow(sl_geom, cmap="gray", vmin=vmin, vmax=vmax)
    axs_top[3].set_title("Geom-only", fontsize=10)
    axs_top[4].imshow(sl_aug, cmap="gray", vmin=vmin, vmax=vmax)
    axs_top[4].set_title("Augmented", fontsize=10)

    # Bottom row: diffs (3 columns centered)
    diff1 = sl_base - sl_raw_aligned
    diff2 = sl_geom - sl_base
    diff3 = sl_aug - sl_geom
    lim = np.percentile(np.abs(np.concatenate([diff1.ravel(), diff2.ravel(), diff3.ravel()])), 99.0)
    lim = float(lim if lim > 0 else 1e-6)

    axs_bot[1].imshow(diff1, cmap="bwr", vmin=-lim, vmax=+lim); axs_bot[1].set_title("Baseline − Raw(aligned)", fontsize=10)
    axs_bot[2].imshow(diff2, cmap="bwr", vmin=-lim, vmax=+lim); axs_bot[2].set_title("Geom − Baseline", fontsize=10)
    axs_bot[3].imshow(diff3, cmap="bwr", vmin=-lim, vmax=+lim); axs_bot[3].set_title("Aug − Geom", fontsize=10)

    for row in (axs_top, axs_bot):
        for ax in row:
            ax.axis("off")


def build_viz_transform(args):
    """
    Outputs aligned keys:
      - image_raw_ras: copy captured after RAS orientation (no scaling)
      - image_baseline: crop to ROI (no geom/intensity aug)
      - image_geom: baseline + geom augs
      - image: full aug (geom + intensity shift)
    All crops are identical across views to enable pixel-wise differences.
    """
    roi = (args.roi_x, args.roi_y, args.roi_z)

    pre = [
        LoadImaged(keys=["image"], image_only=True, dtype=np.int16),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        # capture the RAS-oriented raw BEFORE scaling:
        CopyItemsd(keys=["image"], times=1, names=["image_raw_ras"]),
        # scale ONLY the main 'image' view:
        ScaleIntensityRanged(
            keys=["image"],
            a_min=args.a_min, a_max=args.a_max,
            b_min=args.b_min, b_max=args.b_max, clip=True
        ),
        CropForegroundd(keys=["image", "image_raw_ras"], source_key="image"),
        SpatialPadd(keys=["image", "image_raw_ras"], spatial_size=roi),
        # duplicate 'image' so we can diverge later:
        CopyItemsd(keys=["image"], times=2, names=["image_baseline", "image_geom"]),
        # identical random crop for all views:
        RandSpatialCropd(keys=["image", "image_baseline", "image_geom", "image_raw_ras"],
                         roi_size=roi, random_size=False),
    ]

    geom = [
        RandFlipd(keys=["image", "image_geom"], prob=args.RandFlipd_prob, spatial_axis=0),
        RandFlipd(keys=["image", "image_geom"], prob=args.RandFlipd_prob, spatial_axis=1),
        RandFlipd(keys=["image", "image_geom"], prob=args.RandFlipd_prob, spatial_axis=2),
        RandRotate90d(keys=["image", "image_geom"], prob=args.RandRotate90d_prob, max_k=3),
    ]

    intensity = [
        RandShiftIntensityd(keys="image", offsets=0.1, prob=args.RandShiftIntensityd_prob),
    ]

    tail = [ToTensord(keys=["image", "image_baseline", "image_geom", "image_raw_ras"])]

    return Compose(pre + geom + intensity + tail)

def visualize_raw_vs_transforms(
    args,
    n_per_dataset=2,
    axis="axial",
    slice_idx="center",
    seed=2025,
    save_path=None
):
    """
    Shows: Raw(native), Raw(aligned ROI), Baseline, Geom, Aug + three difference maps.
    """
    random.seed(seed)
    np.random.seed(seed)

    json_btcv = os.path.join(args.data_btcv, "btcv.json")
    json_tcia = os.path.join(args.data_tcia, "dataset_TCIAcovid19_0.json")
    json_luna = os.path.join(args.data_luna, "dataset_LUNA16_0.json")

    dl_btcv = load_decathlon_datalist(json_btcv, False, "training", base_dir=args.data_btcv)
    dl_tcia = load_decathlon_datalist(json_tcia, False, "training", base_dir=args.data_tcia)
    dl_luna = load_decathlon_datalist(json_luna, False, "training", base_dir=args.data_luna)

    # Only keep image paths
    dl_btcv = [{"image": it["image"]} for it in dl_btcv]
    dl_tcia = [{"image": it["image"]} for it in dl_tcia]
    dl_luna = [{"image": it["image"]} for it in dl_luna]

    picks = {
        "BTCV":    [dl_btcv[i] for i in _pick_indices(len(dl_btcv), n_per_dataset, seed=seed+1)],
        "COVID19": [dl_tcia[i] for i in _pick_indices(len(dl_tcia), n_per_dataset, seed=seed+2)],
        "LUNA16":  [dl_luna[i] for i in _pick_indices(len(dl_luna), n_per_dataset, seed=seed+3)],
    }

    xform = build_viz_transform(args)
    xform.set_random_state(seed=seed)  # reproducible crops/augs for viz

    n_rows = sum(len(v) for v in picks.values())
    # 2 rows (images + diffs), 5 columns images and 5 columns for bottom (we'll fill 3 middle with diffs)
    fig, axs = plt.subplots(n_rows * 2, 5, figsize=(5 * 3.2, n_rows * 2 * 2.8), constrained_layout=True)

    if axs.ndim == 2:
        pass
    else:
        axs = np.array(axs).reshape(n_rows * 2, 5)

    row_block = 0
    for ds_name, items in picks.items():
        for it in items:
            # ----- raw native (direct nibabel load) -----
            raw_path = it["image"]
            nii = nib.load(raw_path)                 # native orientation
            vol_native = nii.get_fdata(dtype=np.float32)  # [X,Y,Z] in file's native axes
            sl_raw_native = _slice_from_tensor(vol_native, axis=axis, slice_idx="center")

            # ----- transformed views (aligned ROI) -----
            data_dict = {"image": raw_path}
            out = xform(data_dict)

            raw_aligned = out["image_raw_ras"]       # [1,D,H,W], RAS + same crop as others
            base = out["image_baseline"]             # scaled + crop; no geom/intensity
            geom = out["image_geom"]                 # + geom
            aug  = out["image"]                      # + geom + intensity

            sl_raw_aligned = _slice_from_tensor(raw_aligned, axis=axis, slice_idx=slice_idx)
            sl_base = _slice_from_tensor(base, axis=axis, slice_idx=slice_idx)
            sl_geom = _slice_from_tensor(geom, axis=axis, slice_idx=slice_idx)
            sl_aug  = _slice_from_tensor(aug,  axis=axis, slice_idx=slice_idx)

            title_left = f"{ds_name}\n{pathlib.Path(raw_path).name}"
            axs_top = axs[row_block * 2 + 0]
            axs_bot = axs[row_block * 2 + 1]
            _subplot_rows(fig, axs_top, axs_bot, title_left,
                          sl_raw_native, sl_raw_aligned, sl_base, sl_geom, sl_aug)
            row_block += 1

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"[saved] {save_path}")
    else:
        plt.show()


if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="Process pretraining")
    parser.add_argument("--config",help="Path to config", required=True)
    args = parser.parse_args()
    config = args.config
    with open(config, "r") as f:
        config = yaml.safe_load(f)
 

    # Visualize 2 cases from each dataset, axial center slice
    visualize_raw_vs_transforms(
        dotdict(config),
        n_per_dataset=2,
        axis="axial",            # 'axial'|'coronal'|'sagittal'
        slice_idx="center",      # or an int
        seed=2025,
        save_path="/mnt/data/backup/tj/2fold_voco_vis/viz_transforms_panel.png"
    )

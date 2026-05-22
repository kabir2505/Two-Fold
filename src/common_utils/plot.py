import pandas as pd
import matplotlib.pyplot as plt
import re
import os

def plot_training_log(csv_path):
    """
    Reads a training log CSV and plots:
      - total train loss
      - scaled individual losses (based on their suffix, e.g. mask_loss_0.5 -> x0.5)
      - learning rate over epochs
    """

    df = pd.read_csv(csv_path)

    # Find loss columns that have a suffix like _0.5 or _1.0
    loss_cols = [c for c in df.columns if re.search(r'_loss_\d+(\.\d+)?', c)]
    print(loss_cols)
    scaled_losses = {}
    for col in loss_cols:
        # Extract numeric suffix (e.g. mask_loss_0.5 -> 0.5)
        match = re.search(r'_(\d+(\.\d+)?)$', col)
        scale = float(match.group(1)) if match else 1.0
        print(scale)
        #scaled_losses[col] = df[col] * scale
        scaled_losses[col] = df[col] 

    # --- Plot setup ---
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Left y-axis: losses
    ax1.set_title("Training Loss Breakdown")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="tab:blue")
    ax1.plot(df["epoch"], df["train_loss"], label="Train Loss", color="tab:blue", linewidth=2)
    for name, values in scaled_losses.items():
        ax1.plot(df["epoch"], values, label=f"{name} (scaled)", linestyle="--", alpha=0.8)

    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.legend(loc="upper left")

    # Right y-axis: learning rate
    ax2 = ax1.twinx()
    ax2.set_ylabel("Learning Rate", color="tab:red")
    ax2.plot(df["epoch"], df["lr"], label="LR", color="tab:red", linestyle=":")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    plt.tight_layout()
    plt.show()
    filename = "btcv_pretrain_all"
    plt.savefig(f"{filename}.png")

plot_training_log("/mnt/data/backup/tj/2fold_voco_data/outputs/pretrain_btcv_tall/pretrain_log.csv")
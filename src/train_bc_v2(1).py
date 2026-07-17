"""
BC training with a SEPARATE gripper head.

Why: gripper open/close is bimodal (two discrete states), not a smooth
continuum. Regressing it jointly with the 5 continuous arm joints under one
MSE loss tends to produce washy, indecisive predictions near the boundary
(this is the same multi-modality issue the AWE paper found with plain MSE
regression on multi-modal targets — see Figure 6 in that paper).

Fix: two heads —
  - arm_head: regression, 5 continuous outputs, MSE loss
  - gripper_head: binary classification (open=0 / closed=1), BCE loss

You need to pick a threshold in tick-space that separates "open" from
"closed" in your data — check gripper_events.png from earlier, or just use
the midpoint between your calibrated open/close values.

Usage:
    python3 train_bc_v2.py
"""

import h5py
import numpy as np
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

CLEAN_DIR = "./demos_clean"
HISTORY_LEN = 5
BATCH_SIZE = 32
EPOCHS = 200
LR = 1e-3
VAL_SPLIT = 0.2
HIDDEN_SIZE = 64
NUM_ARM_JOINTS = 5   # joints 1-5

# --- IMPORTANT: set this based on your actual gripper calibration ---
GRIPPER_OPEN_TICK = 130    # your calibrated "open" value
GRIPPER_CLOSED_TICK = 300  # your calibrated "closed" value
GRIPPER_THRESHOLD = (GRIPPER_OPEN_TICK + GRIPPER_CLOSED_TICK) / 2  # midpoint

torch.manual_seed(0)
np.random.seed(0)


def load_episodes():
    files = sorted(glob.glob(f"{CLEAN_DIR}/*.h5"))
    episodes = []
    for f in files:
        with h5py.File(f, "r") as h:
            jp = h["joint_positions"][:].astype(np.float32)
        episodes.append(jp)
    return episodes


def compute_normalization(episodes):
    """Normalize only the 5 arm joints (gripper is handled separately as a class label)."""
    all_data = np.concatenate(episodes, axis=0)[:, :NUM_ARM_JOINTS]
    mean = all_data.mean(axis=0)
    std = all_data.std(axis=0) + 1e-6
    return mean, std


class BCDataset(Dataset):
    def __init__(self, episodes, mean, std, history_len):
        self.samples = []
        for ep in episodes:
            arm = ep[:, :NUM_ARM_JOINTS]
            gripper = ep[:, NUM_ARM_JOINTS]  # raw ticks
            arm_norm = (arm - mean) / std
            gripper_binary = (gripper > GRIPPER_THRESHOLD).astype(np.float32)  # 0=open, 1=closed

            # full 6-dim history input: normalized arm + raw binary gripper
            full_input = np.concatenate([arm_norm, gripper_binary[:, None]], axis=1)

            for t in range(history_len, len(ep) - 1):
                hist = full_input[t - history_len:t]
                arm_target = arm_norm[t + 1]
                gripper_target = gripper_binary[t + 1]
                self.samples.append((hist, arm_target, gripper_target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        hist, arm_target, gripper_target = self.samples[idx]
        return (torch.tensor(hist), torch.tensor(arm_target),
                torch.tensor(gripper_target, dtype=torch.float32))


class BCPolicyV2(nn.Module):
    def __init__(self, input_dim=NUM_ARM_JOINTS + 1, hidden_size=HIDDEN_SIZE):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, batch_first=True)
        self.arm_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, NUM_ARM_JOINTS),
        )
        self.gripper_head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),   # raw logit — sigmoid applied in loss/inference
        )

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        last_hidden = h_n[-1]
        arm_pred = self.arm_head(last_hidden)
        gripper_logit = self.gripper_head(last_hidden).squeeze(-1)
        return arm_pred, gripper_logit


def split_episodes(episodes, val_split):
    idx = np.random.permutation(len(episodes))
    n_val = max(1, int(len(episodes) * val_split))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return [episodes[i] for i in train_idx], [episodes[i] for i in val_idx]


def train():
    episodes = load_episodes()
    print(f"Loaded {len(episodes)} episodes")

    train_eps, val_eps = split_episodes(episodes, VAL_SPLIT)
    mean, std = compute_normalization(train_eps)
    print(f"Gripper threshold (ticks): {GRIPPER_THRESHOLD}")

    train_ds = BCDataset(train_eps, mean, std, HISTORY_LEN)
    val_ds = BCDataset(val_eps, mean, std, HISTORY_LEN)
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = BCPolicyV2()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    arm_loss_fn = nn.MSELoss()
    gripper_loss_fn = nn.BCEWithLogitsLoss()  # numerically stable sigmoid + BCE combined

    train_losses, val_losses, val_gripper_acc = [], [], []

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        for hist, arm_target, gripper_target in train_loader:
            optimizer.zero_grad()
            arm_pred, gripper_logit = model(hist)
            loss = arm_loss_fn(arm_pred, arm_target) + gripper_loss_fn(gripper_logit, gripper_target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * hist.size(0)
        epoch_loss /= len(train_ds)

        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for hist, arm_target, gripper_target in val_loader:
                arm_pred, gripper_logit = model(hist)
                loss = arm_loss_fn(arm_pred, arm_target) + gripper_loss_fn(gripper_logit, gripper_target)
                val_loss += loss.item() * hist.size(0)
                pred_class = (torch.sigmoid(gripper_logit) > 0.5).float()
                correct += (pred_class == gripper_target).sum().item()
                total += gripper_target.size(0)
        val_loss /= max(1, len(val_ds))
        acc = correct / max(1, total)

        train_losses.append(epoch_loss)
        val_losses.append(val_loss)
        val_gripper_acc.append(acc)

        if epoch % 20 == 0 or epoch == EPOCHS - 1:
            print(f"Epoch {epoch:4d} | train {epoch_loss:.4f} | val {val_loss:.4f} | gripper_acc {acc:.3f}")

    torch.save({
        "model_state": model.state_dict(),
        "mean": mean, "std": std,
        "history_len": HISTORY_LEN, "hidden_size": HIDDEN_SIZE,
        "gripper_threshold": GRIPPER_THRESHOLD,
        "gripper_open_tick": GRIPPER_OPEN_TICK,
        "gripper_closed_tick": GRIPPER_CLOSED_TICK,
    }, "bc_policy_v2.pt")
    print("\nSaved bc_policy_v2.pt")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(train_losses, label="train")
    axes[0].plot(val_losses, label="val")
    axes[0].set_yscale("log")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(val_gripper_acc)
    axes[1].set_title("Val gripper classification accuracy")
    axes[1].set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig("training_curve_v2.png", dpi=120)
    print("Saved training_curve_v2.png")


if __name__ == "__main__":
    train()

"""
Behavioral Cloning training on kinesthetic demo data.

Task framing (as discussed): since these are kinesthetic-taught demos with
no separate "commanded action," the label for each timestep is simply the
NEXT observed joint position. The policy learns:

    joint_positions[t] (+ recent history)  -->  joint_positions[t+1]

Model: small LSTM over a short history window, predicting the next
6-dim joint position vector (5 arm joints + gripper).

Usage:
    python3 train_bc.py
"""

import h5py
import numpy as np
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

CLEAN_DIR = "./demos_clean"
HISTORY_LEN = 5          # how many past timesteps the model sees
BATCH_SIZE = 32
EPOCHS = 200
LR = 1e-3
VAL_SPLIT = 0.2           # fraction of EPISODES held out for validation
HIDDEN_SIZE = 64
NUM_JOINTS = 6

torch.manual_seed(0)
np.random.seed(0)


# ---------------- DATA LOADING & NORMALIZATION ----------------

def load_episodes():
    files = sorted(glob.glob(f"{CLEAN_DIR}/*.h5"))
    episodes = []
    for f in files:
        with h5py.File(f, "r") as h:
            jp = h["joint_positions"][:].astype(np.float32)
        episodes.append(jp)
    return episodes


def compute_normalization(episodes):
    """Normalize using mean/std over ALL timesteps in the training data.
    Ticks are on a 0-1023 scale but joints have different ranges, so
    per-joint normalization matters."""
    all_data = np.concatenate(episodes, axis=0)
    mean = all_data.mean(axis=0)
    std = all_data.std(axis=0) + 1e-6
    return mean, std


class BCDataset(Dataset):
    """Builds (history_window -> next_state) training pairs from episodes."""

    def __init__(self, episodes, mean, std, history_len):
        self.samples = []
        for ep in episodes:
            ep_norm = (ep - mean) / std
            for t in range(history_len, len(ep_norm) - 1):
                hist = ep_norm[t - history_len:t]      # (history_len, 6)
                target = ep_norm[t + 1]                 # (6,) next position
                self.samples.append((hist, target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        hist, target = self.samples[idx]
        return torch.tensor(hist), torch.tensor(target)


# ---------------- MODEL ----------------

class BCPolicy(nn.Module):
    def __init__(self, input_dim=NUM_JOINTS, hidden_size=HIDDEN_SIZE):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_dim),
        )

    def forward(self, x):
        # x: (batch, history_len, 6)
        out, (h_n, c_n) = self.lstm(x)
        last_hidden = h_n[-1]          # (batch, hidden_size)
        return self.head(last_hidden)  # (batch, 6) predicted next state


# ---------------- TRAIN / VAL SPLIT (by episode, not by timestep!) ----------------

def split_episodes(episodes, val_split):
    """IMPORTANT: split by whole episodes, not by individual timesteps.
    Splitting by timestep would leak information (adjacent timesteps within
    the same episode are highly correlated), giving a falsely optimistic
    validation score."""
    idx = np.random.permutation(len(episodes))
    n_val = max(1, int(len(episodes) * val_split))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    train_eps = [episodes[i] for i in train_idx]
    val_eps = [episodes[i] for i in val_idx]
    return train_eps, val_eps


# ---------------- TRAINING LOOP ----------------

def train():
    episodes = load_episodes()
    print(f"Loaded {len(episodes)} episodes")

    train_eps, val_eps = split_episodes(episodes, VAL_SPLIT)
    print(f"Train episodes: {len(train_eps)}, Val episodes: {len(val_eps)}")

    mean, std = compute_normalization(train_eps)  # fit normalization on TRAIN only
    print("Per-joint mean:", mean.round(1))
    print("Per-joint std :", std.round(1))

    train_ds = BCDataset(train_eps, mean, std, HISTORY_LEN)
    val_ds = BCDataset(val_eps, mean, std, HISTORY_LEN)
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = BCPolicy()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    train_losses, val_losses = [], []

    for epoch in range(EPOCHS):
        model.train()
        epoch_train_loss = 0.0
        for hist, target in train_loader:
            optimizer.zero_grad()
            pred = model(hist)
            loss = loss_fn(pred, target)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * hist.size(0)
        epoch_train_loss /= len(train_ds)

        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for hist, target in val_loader:
                pred = model(hist)
                loss = loss_fn(pred, target)
                epoch_val_loss += loss.item() * hist.size(0)
        epoch_val_loss /= max(1, len(val_ds))

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)

        if epoch % 20 == 0 or epoch == EPOCHS - 1:
            print(f"Epoch {epoch:4d} | train_loss {epoch_train_loss:.5f} | val_loss {epoch_val_loss:.5f}")

    # save model + normalization stats together (need both for inference later)
    torch.save({
        "model_state": model.state_dict(),
        "mean": mean,
        "std": std,
        "history_len": HISTORY_LEN,
        "hidden_size": HIDDEN_SIZE,
    }, "bc_policy.pt")
    print("\nSaved trained model to bc_policy.pt")

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="train loss")
    plt.plot(val_losses, label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("MSE loss (normalized space)")
    plt.yscale("log")
    plt.legend()
    plt.title("BC Training Curve")
    plt.tight_layout()
    plt.savefig("training_curve.png", dpi=120)
    print("Saved training_curve.png")


if __name__ == "__main__":
    train()

"""
Visualize cleaned kinesthetic demos:
1. Per-joint trajectories overlaid across all episodes (spot consistency/outliers)
2. Gripper trajectory with detected pick/place events marked
3. Episode length distribution

Usage:
    python3 visualize_demos.py
"""

import h5py
import numpy as np
import glob
import matplotlib.pyplot as plt

CLEAN_DIR = "./demos_clean"
JOINT_NAMES = ["Joint1", "Joint2", "Joint3", "Joint4", "Joint5", "Gripper"]


def load_all():
    files = sorted(glob.glob(f"{CLEAN_DIR}/*.h5"))
    episodes = []
    for f in files:
        with h5py.File(f, "r") as h:
            jp = h["joint_positions"][:]
            ts = h["timestamps"][:]
        episodes.append({"name": f.split("/")[-1], "jp": jp, "ts": ts})
    return episodes


def plot_all_joints(episodes):
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    axes = axes.flatten()

    for j in range(6):
        ax = axes[j]
        for ep in episodes:
            ax.plot(ep["ts"], ep["jp"][:, j], alpha=0.5, linewidth=1)
        ax.set_title(JOINT_NAMES[j])
        ax.set_xlabel("time (s)")
        ax.set_ylabel("position (ticks)")

    plt.tight_layout()
    plt.savefig("all_joints_overlay.png", dpi=120)
    print("Saved all_joints_overlay.png")


def plot_gripper_events(episodes):
    fig, ax = plt.subplots(figsize=(10, 6))
    for ep in episodes:
        gripper = ep["jp"][:, 5]
        ts = ep["ts"]
        ax.plot(ts, gripper, alpha=0.4)

        diffs = np.diff(gripper)
        close_idx = np.argmax(diffs)
        open_idx = np.argmin(diffs)
        ax.scatter(ts[close_idx], gripper[close_idx], color="red", s=20, zorder=5)
        ax.scatter(ts[open_idx], gripper[open_idx], color="green", s=20, zorder=5)

    ax.set_title("Gripper trajectories (red=detected close/pick, green=detected open/place)")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("gripper position (ticks)")
    plt.tight_layout()
    plt.savefig("gripper_events.png", dpi=120)
    print("Saved gripper_events.png")


def plot_episode_lengths(episodes):
    lengths = [len(ep["jp"]) for ep in episodes]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(lengths)), lengths)
    ax.set_xlabel("episode index")
    ax.set_ylabel("length (timesteps)")
    ax.set_title(f"Episode lengths (n={len(lengths)}, mean={np.mean(lengths):.0f})")
    plt.tight_layout()
    plt.savefig("episode_lengths.png", dpi=120)
    print("Saved episode_lengths.png")


if __name__ == "__main__":
    episodes = load_all()
    print(f"Loaded {len(episodes)} clean episodes")
    plot_all_joints(episodes)
    plot_gripper_events(episodes)
    plot_episode_lengths(episodes)

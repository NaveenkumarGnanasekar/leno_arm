
import h5py
import numpy as np
import glob
import os
import shutil

RAW_DIR = "./demos3"
CLEAN_DIR = "./demos_clean"

MAX_JUMP_TICKS = 100       # single-step change larger than this = likely a glitch
MIN_EPISODE_LEN = 20       # timesteps; shorter = probably an accidental short recording
LENGTH_OUTLIER_MAD_MULT = 4.0   # flag episodes whose length is this many MADs from median
SHAPE_OUTLIER_MAD_MULT = 4.0    # flag episodes whose shape-deviation is this many MADs from median
APPLY_SMOOTHING = True     # light median filter to remove single-sample noise
SMOOTH_WINDOW = 3          # must be odd


def resample_trajectory(jp, n_points=50):
    """Resample a variable-length episode to a fixed number of time-normalized
    points, so episodes of different lengths/durations can be compared directly."""
    t_orig = np.linspace(0, 1, len(jp))
    t_new = np.linspace(0, 1, n_points)
    resampled = np.zeros((n_points, jp.shape[1]))
    for j in range(jp.shape[1]):
        resampled[:, j] = np.interp(t_new, t_orig, jp[:, j])
    return resampled


def detect_shape_outliers(episodes_data, mad_mult):
    """Compare each episode's resampled shape against the per-timestep MEDIAN
    across all episodes. Uses a MAD-based relative threshold (like the length
    check) rather than a fixed absolute tick value -- this self-calibrates to
    whatever the natural pacing/shape variance of YOUR dataset actually is,
    instead of guessing a number that may be too strict or too loose.
    Catches outliers that glitch-detection misses: episodes with no single
    big jump, but that are smoothly, consistently different overall."""
    resampled_all = np.stack([resample_trajectory(jp) for jp in episodes_data])  # (N, 50, 6)
    median_traj = np.median(resampled_all, axis=0)  # (50, 6)

    max_devs = np.array([np.max(np.abs(traj - median_traj)) for traj in resampled_all])
    median_dev = np.median(max_devs)
    mad_dev = np.median(np.abs(max_devs - median_dev)) + 1e-6

    outlier_flags = (max_devs - median_dev) > mad_mult * mad_dev
    return outlier_flags.tolist(), max_devs


def median_filter_1d(arr, window):
    """Simple median filter along time axis, per-joint."""
    pad = window // 2
    padded = np.pad(arr, ((pad, pad), (0, 0)), mode="edge")
    out = np.zeros_like(arr)
    for i in range(arr.shape[0]):
        out[i] = np.median(padded[i:i + window], axis=0)
    return out


def main():
    os.makedirs(CLEAN_DIR, exist_ok=True)
    files = sorted(glob.glob(f"{RAW_DIR}/*.h5"))
    print(f"Found {len(files)} raw episodes\n")

    
    survivors = []  
    dropped = 0

    for f in files:
        name = os.path.basename(f)
        with h5py.File(f, "r") as h:
            jp = h["joint_positions"][:]
            ts = h["timestamps"][:]
            attrs = dict(h.attrs)

        if len(jp) < MIN_EPISODE_LEN:
            print(f"DROP {name}: too short ({len(jp)} steps)")
            dropped += 1
            continue

        diffs = np.abs(np.diff(jp, axis=0))
        max_jump = diffs.max()
        if max_jump > MAX_JUMP_TICKS:
            jump_loc = np.unravel_index(np.argmax(diffs), diffs.shape)
            print(f"DROP {name}: jump of {max_jump:.0f} ticks at step {jump_loc[0]}, joint {jump_loc[1]+1}")
            dropped += 1
            continue

        survivors.append((name, jp, ts, attrs))

    print(f"\nPass 1 done: {len(survivors)} survived length/glitch checks\n")

   
    lengths = np.array([len(jp) for _, jp, _, _ in survivors])
    median_len = np.median(lengths)
    mad_len = np.median(np.abs(lengths - median_len)) + 1e-6
    length_outlier_flags = np.abs(lengths - median_len) > LENGTH_OUTLIER_MAD_MULT * mad_len

    for i, is_outlier in enumerate(length_outlier_flags):
        if is_outlier:
          name = survivors[i][0]
          print(f"FLAG {name}: length {lengths[i]} is a length outlier (median={median_len:.0f})")

    
    episodes_data = [jp for _, jp, _, _ in survivors]
    shape_outlier_flags, max_devs = detect_shape_outliers(episodes_data, SHAPE_OUTLIER_MAD_MULT)

    for i, is_outlier in enumerate(shape_outlier_flags):
        name = survivors[i][0]
        marker = "FLAG" if is_outlier else "    "
        print(f"{marker} {name}: shape deviation = {max_devs[i]:.0f} ticks")

    
    kept = 0
    for i, (name, jp, ts, attrs) in enumerate(survivors):
        if length_outlier_flags[i] or shape_outlier_flags[i]:
            print(f"DROP {name}: outlier (length_outlier={length_outlier_flags[i]}, shape_outlier={shape_outlier_flags[i]})")
            dropped += 1
            continue

        if APPLY_SMOOTHING:
            jp = median_filter_1d(jp, SMOOTH_WINDOW)

        out_path = f"{CLEAN_DIR}/{name}"
        with h5py.File(out_path, "w") as h:
            h.create_dataset("joint_positions", data=jp)
            h.create_dataset("timestamps", data=ts)
            for k, v in attrs.items():
                h.attrs[k] = v
        kept += 1

    print(f"\nDone. Kept {kept}, dropped {dropped}. Clean data in '{CLEAN_DIR}/'")
    print(f"\nIf too many/few episodes get flagged, tune LENGTH_OUTLIER_MAD_MULT "
          f"and SHAPE_OUTLIER_MAD_MULT at the top of this script.")


if __name__ == "__main__":
    main()

"""
Closed-loop BC policy execution on the real AX-12A arm.

Loop:
  1. Read current joint positions (real feedback)
  2. Feed last HISTORY_LEN readings into the trained LSTM
  3. Get predicted next joint position
  4. Clamp to safe joint limits, command servos to move there
  5. Wait briefly, repeat

Safety:
  - MOVE_SPEED kept low by default
  - Every predicted position is clamped to JOINT_LIMITS before sending
  - A per-step maximum delta is enforced so the model can never command a
    huge, sudden jump even if it predicts something wild
  - Keep a hand near the power switch for your first several runs

Usage:
    python3 run_policy.py
"""

from dynamixel_sdk import PortHandler, PacketHandler, GroupSyncWrite
import torch
import torch.nn as nn
import numpy as np
import time
import home
# ---------------- CONFIG ----------------
DEVICENAME = '/dev/ttyUSB0'
BAUDRATE = 1000000
PROTOCOL_VERSION = 1.0

ADDR_TORQUE_ENABLE  = 24
ADDR_MOVING_SPEED   = 32
ADDR_GOAL_POSITION  = 30
ADDR_PRESENT_POSITION = 36

JOINT_LIMITS = {
    1: {"min": 280, "max": 690},
    2: {"min": 189, "max": 709},
    3: {"min": 240, "max": 1000},
    4: {"min": 17,  "max": 1000},
    5: {"min": 256, "max": 800},
    6: {"min": 119, "max": 484},
}
ALL_IDS = list(JOINT_LIMITS.keys())

HOME_POSE = {1: 513, 2: 190, 3: 241, 4: 80, 5: 595, 6: 200}

MODEL_PATH = "bc_policy.pt"
CONTROL_HZ = 20  # MUST match RECORD_HZ from data collection (20Hz).
                          # Mismatched frequency changes the effective "speed"
                          # implied by the history window the model was trained on.
MOVE_SPEED = 60    # keep conservative for early runs
MAX_STEP_DELTA = 80        # ticks — hard cap on how far any joint can move
                            # in a single control step, regardless of prediction
MAX_EPISODE_STEPS = 65  #ncreased since we're now running at higher Hz
HISTORY_LEN = 5             # must match training config
SMOOTHING_ALPHA = 0.7# exponential smoothing on commanded target;
                             # lower = smoother/slower to react, higher = snappier/noisier.
                             # Start at 0.3, tune from there.
# -----------------------------------------


# ---------------- MODEL DEFINITION (must match training script) ----------------

class BCPolicy(nn.Module):
    def __init__(self, input_dim=6, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_dim),
        )

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        return self.head(h_n[-1])


# ---------------- HARDWARE HELPERS ----------------

port = PortHandler(DEVICENAME)
packet = PacketHandler(PROTOCOL_VERSION)
port.openPort()
port.setBaudRate(BAUDRATE)


def torque_on(ids):
    for jid in ids:
        packet.write1ByteTxRx(port, jid, ADDR_TORQUE_ENABLE, 1)

def torque_off(ids):
    for jid in ids:
        packet.write1ByteTxRx(port, jid, ADDR_TORQUE_ENABLE, 0)

def set_speed(ids, speed):
    for jid in ids:
        packet.write2ByteTxRx(port, jid, ADDR_MOVING_SPEED, speed)

def get_current_positions(ids):
    positions = {}
    for jid in ids:
        pos, _, _ = packet.read2ByteTxRx(port, jid, ADDR_PRESENT_POSITION)
        positions[jid] = pos
    return positions

def clamp_to_limits(pose: dict):
    safe_pose = {}
    for jid, goal in pose.items():
        limits = JOINT_LIMITS[jid]
        clamped = max(limits["min"], min(limits["max"], goal))
        safe_pose[jid] = clamped
    return safe_pose

def clamp_step_delta(current: dict, target: dict, max_delta):
    """Prevent any single-step command from moving a joint further than
    max_delta ticks from its CURRENT actual position, regardless of what
    the model predicted. This is the key safety net against a bad/wild
    prediction causing a violent motion."""
    safe_target = {}
    for jid, goal in target.items():
        cur = current[jid]
        delta = goal - cur
        delta = max(-max_delta, min(max_delta, delta))
        safe_target[jid] = cur + delta
    return safe_target

def move_to_pose(pose: dict):
    pose = clamp_to_limits(pose)
    sync_write = GroupSyncWrite(port, packet, ADDR_GOAL_POSITION, 2)
    for jid, pos in pose.items():
        pos = int(pos)
        param = [pos & 0xFF, (pos >> 8) & 0xFF]
        sync_write.addParam(jid, param)
    sync_write.txPacket()
    sync_write.clearParam()

def go_home():
    print("Moving to home pose...")
    torque_on(ALL_IDS)
    set_speed(ALL_IDS, MOVE_SPEED)
    move_to_pose(HOME_POSE)
    time.sleep(2.0)
    print("At home.")


# ---------------- INFERENCE LOOP ----------------

def load_model():
    # weights_only=False needed because the checkpoint contains numpy arrays
    # (mean/std normalization stats), not just model weights. Safe here since
    # this is a file we generated ourselves in train_bc.py.
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    model = BCPolicy(hidden_size=checkpoint["hidden_size"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    mean = checkpoint["mean"]
    std = checkpoint["std"]
    history_len = checkpoint["history_len"]
    return model, mean, std, history_len


def run_policy():
    model, mean, std, history_len = load_model()
    print(f"Loaded model. Normalization mean={mean.round(1)}, std={std.round(1)}")

    go_home()
    torque_on(ALL_IDS)
    set_speed(ALL_IDS, MOVE_SPEED)

    # seed history buffer with HISTORY_LEN copies of the current (home) position
    current = get_current_positions(ALL_IDS)
    current_vec = np.array([current[jid] for jid in ALL_IDS], dtype=np.float32)
    history = [current_vec.copy() for _ in range(history_len)]

    # smoothed_target starts at the current position — this is what actually
    # gets sent to the servos, blended gradually toward each new prediction
    # rather than jumping straight to it (reduces frame-to-frame jitter)
    smoothed_target = current_vec.copy()

    period = 1.0 / CONTROL_HZ

    print(f"\nRunning policy for up to {MAX_EPISODE_STEPS} steps. Ctrl+C to stop early.\n")
    try:
        for step in range(MAX_EPISODE_STEPS):
            loop_start = time.time()

            # normalize history, run model
            hist_arr = np.stack(history, axis=0)               # (history_len, 6)
            hist_norm = (hist_arr - mean) / std
            hist_tensor = torch.tensor(hist_norm, dtype=torch.float32).unsqueeze(0)  # (1, hist, 6)

            with torch.no_grad():
                pred_norm = model(hist_tensor).squeeze(0).numpy()  # (6,)

            pred = pred_norm * std + mean   # de-normalize back to tick space

            # exponential smoothing: blend new prediction with the previously
            # commanded target, rather than jumping straight to the raw
            # prediction. This is what fixes frame-to-frame jitter/oscillation —
            # same idea as ACT's temporal ensembling, simplified to a single-step EMA.
            smoothed_target = SMOOTHING_ALPHA * pred + (1 - SMOOTHING_ALPHA) * smoothed_target
            pred_pose = {jid: smoothed_target[i] for i, jid in enumerate(ALL_IDS)}

            # read real current position, apply safety clamps
            current = get_current_positions(ALL_IDS)
            safe_pose = clamp_step_delta(current, pred_pose, MAX_STEP_DELTA)
            safe_pose = clamp_to_limits(safe_pose)

            move_to_pose(safe_pose)

            print(f"Step {step:3d} | pred: {[int(pred_pose[j]) for j in ALL_IDS]} "
                  f"| sent: {[int(safe_pose[j]) for j in ALL_IDS]}")

            # update history with the REAL observed position (read again after
            # the move command — grounding predictions in real feedback, not
            # in the model's own past predictions)
            time.sleep(0.15)  # brief settle time before reading back
            new_current = get_current_positions(ALL_IDS)
            new_vec = np.array([new_current[jid] for jid in ALL_IDS], dtype=np.float32)
            history.pop(0)
            history.append(new_vec)

            elapsed = time.time() - loop_start
            if elapsed < period:
                time.sleep(period - elapsed)

    except KeyboardInterrupt:
        print("\nStopped by user.")

    torque_off(ALL_IDS)
    print("Torque released. Done.")


if __name__ == "__main__":
    run_policy()
    home.main()
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
ARM_IDS = [1, 2, 3, 4, 5]
GRIPPER_ID = 6

HOME_POSE = {1: 513, 2: 190, 3: 241, 4: 80, 5: 595, 6: 200}

MODEL_PATH = "bc_policy_v2.pt"
CONTROL_HZ = 20           # MUST match RECORD_HZ from data collection (20Hz).
                          # Mismatched frequency changes the effective "speed"
                          # implied by the history window the model was trained on.
MOVE_SPEED = 60           # keep conservative for early runs
MAX_STEP_DELTA = 80        # ticks — hard cap on how far any joint can move
                            # in a single control step, regardless of prediction
MAX_EPISODE_STEPS = 300    # increased since we're now running at higher Hz
HISTORY_LEN = 5             # must match training config
SMOOTHING_ALPHA = 0.3       # exponential smoothing on commanded target;
                             # lower = smoother/slower to react, higher = snappier/noisier.
                             # Start at 0.3, tune from there.
# -----------------------------------------


# ---------------- MODEL DEFINITION (must match training script) ----------------

class BCPolicyV2(nn.Module):
    def __init__(self, input_dim=6, hidden_size=64, num_arm_joints=5):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, batch_first=True)
        self.arm_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_arm_joints),
        )
        self.gripper_head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        last_hidden = h_n[-1]
        arm_pred = self.arm_head(last_hidden)
        gripper_logit = self.gripper_head(last_hidden).squeeze(-1)
        return arm_pred, gripper_logit


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
    # this is a file we generated ourselves in train_bc_v2.py.
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    model = BCPolicyV2(hidden_size=checkpoint["hidden_size"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    mean = checkpoint["mean"]
    std = checkpoint["std"]
    history_len = checkpoint["history_len"]
    gripper_open = checkpoint["gripper_open_tick"]
    gripper_closed = checkpoint["gripper_closed_tick"]
    return model, mean, std, history_len, gripper_open, gripper_closed


def run_policy():
    model, mean, std, history_len, gripper_open, gripper_closed = load_model()
    print(f"Loaded model. Arm mean={mean.round(1)}, std={std.round(1)}")
    print(f"Gripper open={gripper_open}, closed={gripper_closed}")

    go_home()
    torque_on(ALL_IDS)
    set_speed(ALL_IDS, MOVE_SPEED)
    gripper_threshold = (gripper_open + gripper_closed) / 2

    def encode_gripper_binary(raw_tick):
        return 1.0 if raw_tick > gripper_threshold else 0.0

    # seed history buffer: 5 normalized arm values + 1 binary gripper value
    current = get_current_positions(ALL_IDS)
    arm_vec = np.array([current[jid] for jid in ARM_IDS], dtype=np.float32)
    arm_norm = (arm_vec - mean) / std
    gripper_bin = encode_gripper_binary(current[GRIPPER_ID])
    full_vec = np.concatenate([arm_norm, [gripper_bin]])
    history = [full_vec.copy() for _ in range(history_len)]

    # smoothed_target: only for the 5 continuous arm joints (ticks, not normalized)
    smoothed_arm_target = arm_vec.copy()

    period = 1.0 / CONTROL_HZ

    print(f"\nRunning policy for up to {MAX_EPISODE_STEPS} steps. Ctrl+C to stop early.\n")
    try:
        for step in range(MAX_EPISODE_STEPS):
            loop_start = time.time()

            hist_tensor = torch.tensor(np.stack(history, axis=0), dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                arm_pred_norm, gripper_logit = model(hist_tensor)
                arm_pred_norm = arm_pred_norm.squeeze(0).numpy()
                gripper_prob = torch.sigmoid(gripper_logit).item()

            arm_pred = arm_pred_norm * std + mean   # de-normalize arm prediction to ticks

            # smoothing only applies to the continuous arm joints
            smoothed_arm_target = SMOOTHING_ALPHA * arm_pred + (1 - SMOOTHING_ALPHA) * smoothed_arm_target

            # gripper: discrete decision, NOT smoothed the same way (smoothing a
            # binary decision recreates the washy-average problem). Instead use
            # the probability directly — send fully open or fully closed.
            gripper_target = gripper_closed if gripper_prob > 0.5 else gripper_open

            pred_pose = {ARM_IDS[i]: smoothed_arm_target[i] for i in range(len(ARM_IDS))}
            pred_pose[GRIPPER_ID] = gripper_target

            # read real current position, apply safety clamps (gripper exempted
            # from the step-delta clamp since it should snap to its target state
            # promptly, not creep toward it)
            current = get_current_positions(ALL_IDS)
            safe_pose = clamp_step_delta(current, pred_pose, MAX_STEP_DELTA)
            safe_pose[GRIPPER_ID] = gripper_target
            safe_pose = clamp_to_limits(safe_pose)

            move_to_pose(safe_pose)

            print(f"Step {step:3d} | arm: {[int(smoothed_arm_target[i]) for i in range(len(ARM_IDS))]} "
                  f"| gripper_prob: {gripper_prob:.2f} -> {'CLOSED' if gripper_prob > 0.5 else 'OPEN'}")

            # update history with REAL observed feedback, not our own predictions
            time.sleep(0.15)
            new_current = get_current_positions(ALL_IDS)
            new_arm_vec = np.array([new_current[jid] for jid in ARM_IDS], dtype=np.float32)
            new_arm_norm = (new_arm_vec - mean) / std
            new_gripper_bin = encode_gripper_binary(new_current[GRIPPER_ID])
            new_full_vec = np.concatenate([new_arm_norm, [new_gripper_bin]])
            history.pop(0)
            history.append(new_full_vec)

            elapsed = time.time() - loop_start
            if elapsed < period:
                time.sleep(period - elapsed)

    except KeyboardInterrupt:
        print("\nStopped by user.")

    torque_off(ALL_IDS)
    print("Torque released. Done.")


if __name__ == "__main__":
    run_policy()

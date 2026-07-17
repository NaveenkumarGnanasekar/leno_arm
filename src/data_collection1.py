

from dynamixel_sdk import PortHandler, PacketHandler, GroupSyncWrite
import time
import threading
import h5py
import numpy as np
import os

# ---------------- CONFIG ----------------
DEVICENAME = '/dev/ttyUSB0'
BAUDRATE = 1000000
PROTOCOL_VERSION = 1.0

ADDR_TORQUE_ENABLE     = 24
ADDR_MOVING_SPEED      = 32
ADDR_GOAL_POSITION     = 30
ADDR_PRESENT_POSITION  = 36

JOINT_LIMITS = {
    1: {"min": 280, "max": 690},
    2: {"min": 189, "max": 709},
    3: {"min": 240, "max": 1000},
    4: {"min": 17,  "max": 1000},
    5: {"min": 256, "max": 800},
    6: {"min": 119, "max": 484},   # gripper
}

ALL_IDS    = list(JOINT_LIMITS.keys())  # [1, 2, 3, 4, 5, 6]
ARM_IDS    = [1, 2, 3, 4, 5]
GRIPPER_ID = 6

GRIPPER_HOME = 120   # <-- CONFIRM: tick value for gripper's resting state

HOME_POSE = {
    1: 513,
    2: 190,
    3: 241,
    4: 80,
    5: 595,
    6: GRIPPER_HOME,
}

HOME_MOVE_SPEED = 80
RECORD_HZ = 20
SAVE_DIR = "./demos3"
# -----------------------------------------

port = PortHandler(DEVICENAME)
packet = PacketHandler(PROTOCOL_VERSION)

if not port.openPort():
    raise RuntimeError("Failed to open port. Check DEVICENAME / permissions / cable.")
if not port.setBaudRate(BAUDRATE):
    raise RuntimeError("Failed to set baud rate.")


# ---------------- LOW-LEVEL HELPERS ----------------

def torque_off(ids):
    for jid in ids:
        comm_result, error = packet.write1ByteTxRx(port, jid, ADDR_TORQUE_ENABLE, 0)
        if comm_result != 0:
            print(f"[Joint {jid}] Comm error: {packet.getTxRxResult(comm_result)}")
        elif error != 0:
            print(f"[Joint {jid}] Servo error: {packet.getRxPacketError(error)}")

def torque_on(ids):
    for jid in ids:
        comm_result, error = packet.write1ByteTxRx(port, jid, ADDR_TORQUE_ENABLE, 1)
        if comm_result != 0:
            print(f"[Joint {jid}] Comm error: {packet.getTxRxResult(comm_result)}")
        elif error != 0:
            print(f"[Joint {jid}] Servo error: {packet.getRxPacketError(error)}")

def set_speed(ids, speed):
    for jid in ids:
        packet.write2ByteTxRx(port, jid, ADDR_MOVING_SPEED, speed)

def clamp_to_limits(pose: dict):
    safe_pose = {}
    for jid, goal in pose.items():
        limits = JOINT_LIMITS[jid]
        clamped = max(limits["min"], min(limits["max"], goal))
        if clamped != goal:
            print(f"WARNING: joint {jid} goal {goal} out of range "
                  f"[{limits['min']}, {limits['max']}] -> clamped to {clamped}")
        safe_pose[jid] = clamped
    return safe_pose

def move_to_pose(pose: dict):
    pose = clamp_to_limits(pose)
    sync_write = GroupSyncWrite(port, packet, ADDR_GOAL_POSITION, 2)
    for jid, pos in pose.items():
        param = [pos & 0xFF, (pos >> 8) & 0xFF]
        sync_write.addParam(jid, param)
    sync_write.txPacket()
    sync_write.clearParam()

def get_current_positions(ids):
    positions = {}
    for jid in ids:
        pos, _, _ = packet.read2ByteTxRx(port, jid, ADDR_PRESENT_POSITION)
        positions[jid] = pos
    return positions

def wait_until_reached(target_pose, tolerance=10, timeout=6.0):
    start = time.time()
    while time.time() - start < timeout:
        current = get_current_positions(list(target_pose.keys()))
        if all(abs(current[jid] - target_pose[jid]) <= tolerance for jid in target_pose):
            print("Reached target pose.")
            return True
        time.sleep(0.05)
    print("Timeout — check for obstruction or large travel distance.")
    return False


# ---------------- HIGH-LEVEL ACTIONS ----------------

def go_home():
    """Actively drive ALL joints (arm + gripper) to HOME_POSE, then release torque."""
    print("\nMoving to home pose (including gripper)...")
    torque_on(ALL_IDS)
    set_speed(ALL_IDS, HOME_MOVE_SPEED)
    move_to_pose(HOME_POSE)
    wait_until_reached(HOME_POSE)
    torque_off(ALL_IDS)
    print("At home. All joints torque-off and ready for kinesthetic teaching.\n")


def record_episode(episode_num):
    """Kinesthetic recording: all joints torque-off, log positions at fixed rate
    until user presses Enter to stop."""
    torque_off(ALL_IDS)
    print(f"\n--- Recording episode {episode_num} ---")
    print("Move the arm through the task now (control gripper by hand too).")
    print("Press Enter when done with this episode...")

    log = []
    stop_flag = threading.Event()

    def recorder_loop():
        start_time = time.time()
        period = 1.0 / RECORD_HZ
        while not stop_flag.is_set():
            loop_start = time.time()
            positions = get_current_positions(ALL_IDS)
            t = time.time() - start_time
            row = [positions[jid] for jid in ALL_IDS] + [t]
            log.append(row)
            elapsed = time.time() - loop_start
            if elapsed < period:
                time.sleep(period - elapsed)

    thread = threading.Thread(target=recorder_loop)
    thread.start()
    input()  # blocks until Enter is pressed
    stop_flag.set()
    thread.join()

    log = np.array(log)
    os.makedirs(SAVE_DIR, exist_ok=True)
    filename = f"{SAVE_DIR}/demo_{episode_num:03d}.h5"
    with h5py.File(filename, "w") as f:
        f.create_dataset("joint_positions", data=log[:, :len(ALL_IDS)])
        f.create_dataset("timestamps", data=log[:, -1])
        f.attrs["joint_ids"] = ALL_IDS
        f.attrs["gripper_id"] = GRIPPER_ID

    print(f"Saved {filename} — {len(log)} timesteps, {log[-1, -1]:.1f}s")


# ---------------- MENU LOOP ----------------

def main_menu():
    episode_counter = 1
    while True:
        print("\n==== MENU ====")
        print("r - record a new episode")
        print("h - move all joints (incl. gripper) to home")
        print("x - exit")
        choice = input("Choice: ").strip().lower()

        if choice == 'r':
            record_episode(episode_counter)
            episode_counter += 1
        elif choice == 'h':
            go_home()
        elif choice == 'x':
            break
        else:
            print("Invalid choice.")

    port.closePort()
    print("Port closed. Done.")


if __name__ == "__main__":
    main_menu()
from dynamixel_sdk import PortHandler, PacketHandler, GroupSyncWrite
import time

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

# This is the pose we want to move TO (your reported "home" position)
HOME_POSE = {
    1: 513,
    2: 190,
    3: 241,
    4: 80,
    5: 595,
    6: 118,
}

MOVE_SPEED =  100 # keep slow until you trust the motion

port = PortHandler(DEVICENAME)
packet = PacketHandler(PROTOCOL_VERSION)
port.openPort()
port.setBaudRate(BAUDRATE)

def torque_on(ids):
    for jid in ids:
        packet.write1ByteTxRx(port, jid, ADDR_TORQUE_ENABLE, 1)

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

def wait_until_reached(target_pose, tolerance=10, timeout=5.0):
    start = time.time()
    while time.time() - start < timeout:
        current = get_current_positions(list(target_pose.keys()))
        if all(abs(current[jid] - target_pose[jid]) <= tolerance for jid in target_pose):
            print("Reached target pose.")
            return True
        time.sleep(0.05)
    print("Timeout — check for obstruction or large travel distance.")
    return False
def main():
    ids = list(JOINT_LIMITS.keys())
    torque_on(ids)
    set_speed(ids, MOVE_SPEED)

    print("Moving to home pose:", HOME_POSE)
    move_to_pose(HOME_POSE)
    wait_until_reached(HOME_POSE)

    print("Final positions:", get_current_positions(ids))
    port.closePort()
if __name__ == "__main__":
    main()
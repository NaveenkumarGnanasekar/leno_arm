from dynamixel_sdk import PortHandler, PacketHandler

DEVICENAME = '/dev/ttyUSB0'
BAUDRATE = 1000000
PROTOCOL_VERSION = 1.0

ADDR_CW_ANGLE_LIMIT  = 6   # "min" — Clockwise Angle Limit
ADDR_CCW_ANGLE_LIMIT = 8   # "max" — Counter-Clockwise Angle Limit
ADDR_PRESENT_POSITION = 36

JOINT_IDS = [1, 2, 3, 4, 5, 6]

port = PortHandler(DEVICENAME)
packet = PacketHandler(PROTOCOL_VERSION)
port.openPort()
port.setBaudRate(BAUDRATE)

print(f"{'ID':<4}{'Min':<8}{'Max':<8}{'Current':<8}")
joint_info = {}
for jid in JOINT_IDS:
    min_limit, _, _ = packet.read2ByteTxRx(port, jid, ADDR_CW_ANGLE_LIMIT)
    max_limit, _, _ = packet.read2ByteTxRx(port, jid, ADDR_CCW_ANGLE_LIMIT)
    current, _, _   = packet.read2ByteTxRx(port, jid, ADDR_PRESENT_POSITION)
    joint_info[jid] = {"min": min_limit, "max": max_limit, "current": current}
    print(f"{jid:<4}{min_limit:<8}{max_limit:<8}{current:<8}")

port.closePort()
print(joint_info)
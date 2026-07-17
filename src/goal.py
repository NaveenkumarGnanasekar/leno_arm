

#!/usr/bin/env python3

import os

if os.name == 'nt':
    import msvcrt
    def getch():
        return msvcrt.getch().decode()
else:
    import sys, tty, termios
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    def getch():
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

from dynamixel_sdk import *

# Control table address for AX-12A
ADDR_AX_TORQUE_ENABLE       = 24  # Torque enable
ADDR_AX_CW_ANGLE_LIMIT      = 6   # CW angle limit
ADDR_AX_CCW_ANGLE_LIMIT     = 8   # CCW angle limit
ADDR_AX_PRESENT_POSITION    = 36  # Present position
ADDR_AX_GOAL_POSITION       = 30  # Goal position
ADDR_AX_MOVING_SPEED        = 32  # Moving speed

# Protocol version
PROTOCOL_VERSION             = 1.0  # AX-12A uses Protocol 1.0

# Default setting
DXL_ID                       = int(input("Enter motor id : "))    # Dynamixel ID
BAUDRATE                     = 1000000  # Default baudrate for AX-12A
DEVICENAME                   = '/dev/ttyUSB0'  # Adjust for your system

TORQUE_ENABLE                = 1  # Enable torque
TORQUE_DISABLE               = 0


POSITION_MIN                 = 0      # 0 degrees
POSITION_MAX                 = 1023   # 300 degrees
ANGLE_RANGE                  = 300.0  # Disable torque

# Initialize PortHandler instance
portHandler = PortHandler(DEVICENAME)

# Initialize PacketHandler instance
packetHandler = PacketHandler(PROTOCOL_VERSION)

# Open port
if portHandler.openPort():
    print("Succeeded to open the port")
else:
    print("Failed to open the port")
    quit()

# Set port baudrate
if portHandler.setBaudRate(BAUDRATE):
    print("Succeeded to change the baudrate")
else:
    print("Failed to change the baudrate")
    quit()

# Enable Torque
dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_AX_TORQUE_ENABLE, TORQUE_ENABLE)
if dxl_comm_result != COMM_SUCCESS:
    print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
elif dxl_error != 0:
    print("%s" % packetHandler.getRxPacketError(dxl_error))
else:
    print(f"Torque enabled for Dynamixel ID {DXL_ID}")

# Function to determine direction and set speed
def determine_and_move_goal(present_pos, goal_pos, cw_limit, ccw_limit):
    if goal_pos < cw_limit or goal_pos > ccw_limit:
        print("Goal position out of limits!")

    # Determine direction
    if goal_pos > present_pos:
        direction = "CCW"
        speed = 200  # Counterclockwise speed
    else:
        direction = "CW"
        speed = 0x0400 | 200  # Clockwise speed

    # Write moving speed
    packetHandler.write2ByteTxRx(portHandler, DXL_ID, ADDR_AX_MOVING_SPEED, speed)

    # Write goal position
    dxl_comm_result, dxl_error = packetHandler.write2ByteTxRx(portHandler, DXL_ID, ADDR_AX_GOAL_POSITION, goal_pos)
    if dxl_comm_result != COMM_SUCCESS:
        print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
    elif dxl_error != 0:
        print("%s" % packetHandler.getRxPacketError(dxl_error))
    else:
        print(f"Moving {direction} to goal position: {goal_pos}")

# Read current parameters
cw_limit,comm_result, dxl_error = packetHandler.read2ByteTxRx(portHandler, DXL_ID, ADDR_AX_CW_ANGLE_LIMIT)
ccw_limit,comm_reult, dxl_error = packetHandler.read2ByteTxRx(portHandler, DXL_ID, ADDR_AX_CCW_ANGLE_LIMIT)
present_pos,comm_reulst, dxl_error = packetHandler.read2ByteTxRx(portHandler, DXL_ID, ADDR_AX_PRESENT_POSITION)

print(f"Present Position: {present_pos}")
print(f"CW Limit: {cw_limit}, CCW Limit: {ccw_limit}")

cw_deg = int((cw_limit / POSITION_MAX) * ANGLE_RANGE)
ccw_deg = int((ccw_limit / POSITION_MAX) * ANGLE_RANGE)

# Input goal position from user
goal_pos_deg = int(input(f"Enter goal position  degree ({cw_deg} - {ccw_deg}): "))
goal_pos = int(((goal_pos_deg / ANGLE_RANGE) * POSITION_MAX))
determine_and_move_goal(present_pos, goal_pos, cw_limit, ccw_limit)

# Wait for user input before stopping
getch()

# Stop and disable torque
packetHandler.write2ByteTxRx(portHandler, DXL_ID, ADDR_AX_MOVING_SPEED, 0)
packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_AX_TORQUE_ENABLE, TORQUE_DISABLE)

# Close port
portHandler.closePort()


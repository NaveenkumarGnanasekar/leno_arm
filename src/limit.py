

#!/usr/bin/env python3

from dynamixel_sdk import *

# Control table addresses
ADDR_AX_TORQUE_ENABLE       = 24
ADDR_AX_CW_ANGLE_LIMIT      = 6
ADDR_AX_CCW_ANGLE_LIMIT     = 8
ADDR_AX_PRESENT_POSITION    = 36
ADDR_AX_GOAL_POSITION       = 30
ADDR_AX_MOVING_SPEED        = 32

# Protocol version
PROTOCOL_VERSION             = 1.0

DXL_ID                       = int(input("Enter motor id : "))
BAUDRATE                     = 1000000
DEVICENAME                   = '/dev/ttyUSB0'

TORQUE_ENABLE                = 1
TORQUE_DISABLE               = 0

POSITION_MIN                 = 0      # 0 degrees
POSITION_MAX                 = 1023   # 300 degrees
ANGLE_RANGE                  = 300.0


# Initialize PortHandler and PacketHandler
portHandler = PortHandler(DEVICENAME)
packetHandler = PacketHandler(PROTOCOL_VERSION)

# Open port
if not portHandler.openPort():
    print("Failed to open the port")
    quit()

# Set baudrate
if not portHandler.setBaudRate(BAUDRATE):
    print("Failed to set the baudrate")
    quit()

## Enable torque
#dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_AX_TORQUE_ENABLE, TORQUE_ENABLE)
#if dxl_comm_result != COMM_SUCCESS:
#    print(f"Error enabling torque: {packetHandler.getTxRxResult(dxl_comm_result)}")
#elif dxl_error != 0:
#    print(f"Error in torque enable packet: {packetHandler.getRxPacketError(dxl_error)}")
#else:
#    print(f"Torque enabled for Dynamixel ID {DXL_ID}")

present_pos,comm_reulst, dxl_error = packetHandler.read2ByteTxRx(portHandler, DXL_ID, ADDR_AX_PRESENT_POSITION)

print(f"Present Position: {present_pos}")
input()


present_pos,comm_reulst, dxl_error = packetHandler.read2ByteTxRx(portHandler, DXL_ID, ADDR_AX_PRESENT_POSITION)

print(f"Present Position: {present_pos}")




# Function to set CW and CCW angle limits
def set_angle_limits(cw_limit, ccw_limit):
    # Write CW Limit
    dxl_comm_result, dxl_error = packetHandler.write2ByteTxRx(portHandler, DXL_ID, ADDR_AX_CW_ANGLE_LIMIT, cw_limit)
    if dxl_comm_result != COMM_SUCCESS:
        print(f"Error setting CW Limit: {packetHandler.getTxRxResult(dxl_comm_result)}")
    elif dxl_error != 0:
        print(f"Error in CW Limit packet: {packetHandler.getRxPacketError(dxl_error)}")
    else:
        print(f"CW Limit set to: {cw_limit}")

    # Write CCW Limit
    dxl_comm_result, dxl_error = packetHandler.write2ByteTxRx(portHandler, DXL_ID, ADDR_AX_CCW_ANGLE_LIMIT, ccw_limit)
    if dxl_comm_result != COMM_SUCCESS:
        print(f"Error setting CCW Limit: {packetHandler.getTxRxResult(dxl_comm_result)}")
    elif dxl_error != 0:
        print(f"Error in CCW Limit packet: {packetHandler.getRxPacketError(dxl_error)}")
    else:
        print(f"CCW Limit set to: {ccw_limit}")


# Set angle limits manually
cw_limit = int(input("Enter cw limit ( 0 - 1023 :)"))
ccw_limit = int(input("Enter cw limit ( 0 - 1023 :)"))
#cw_limit = int(((cw_limit_deg / ANGLE_RANGE) * POSITION_MAX))
#ccw_limit = int(((ccw_limit_deg / ANGLE_RANGE) * POSITION_MAX))
set_angle_limits(cw_limit, ccw_limit)

# Read back limits to verify
cw_limit, dxl_comm_result, dxl_error = packetHandler.read2ByteTxRx(portHandler, DXL_ID, ADDR_AX_CW_ANGLE_LIMIT)
ccw_limit, dxl_comm_result, dxl_error = packetHandler.read2ByteTxRx(portHandler, DXL_ID, ADDR_AX_CCW_ANGLE_LIMIT)
print(f"Verified CW Limit: {cw_limit}, CCW Limit: {ccw_limit}")

# Disable torque before exiting
packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_AX_TORQUE_ENABLE, TORQUE_DISABLE)

# Close port
portHandler.closePort()


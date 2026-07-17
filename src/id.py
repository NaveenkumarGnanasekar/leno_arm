

#!/usr/bin/env python3

from dynamixel_sdk import *

# Control table addresses
ADDR_AX_TORQUE_ENABLE       = 24
ADDR_AX_ID                  = 3

# Protocol version
PROTOCOL_VERSION             = 1.0

# Default settings
CURRENT_ID                   = int(input("Enter current motor ID: "))
NEW_ID                       = int(input("Enter new motor ID: "))
BAUDRATE                     = 1000000
DEVICENAME                   = '/dev/ttyUSB0'

TORQUE_ENABLE                = 1
TORQUE_DISABLE               = 0

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

# Enable torque for the current ID
dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, CURRENT_ID, ADDR_AX_TORQUE_ENABLE, TORQUE_ENABLE)
if dxl_comm_result != COMM_SUCCESS:
    print(f"Error enabling torque: {packetHandler.getTxRxResult(dxl_comm_result)}")
elif dxl_error != 0:
    print(f"Error in torque enable packet: {packetHandler.getRxPacketError(dxl_error)}")
else:
    print(f"Torque enabled for Dynamixel ID {CURRENT_ID}")

# Change the motor ID
dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, CURRENT_ID, ADDR_AX_ID, NEW_ID)
if dxl_comm_result != COMM_SUCCESS:
    print(f"Error changing ID: {packetHandler.getTxRxResult(dxl_comm_result)}")
elif dxl_error != 0:
    print(f"Error in ID change packet: {packetHandler.getRxPacketError(dxl_error)}")
else:
    print(f"Successfully changed Dynamixel ID from {CURRENT_ID} to {NEW_ID}")

# Disable torque before exiting
packetHandler.write1ByteTxRx(portHandler, NEW_ID, ADDR_AX_TORQUE_ENABLE, TORQUE_DISABLE)

# Close port
portHandler.closePort()




import os
from dynamixel_sdk import *  # Import Dynamixel SDK library

# Define constants
DEVICENAME = "/dev/ttyUSB0"  # Change to your USB port
BAUDRATE = 1000000          # Default baud rate for AX motors
PROTOCOL_VERSION = 1.0 
ADDR_PRESENT_POSITION = 36 
ADDR_AX_CW_ANGLE_LIMIT      = 6   # CW angle limit
ADDR_AX_CCW_ANGLE_LIMIT     = 8    # AX uses Protocol 1.0

def find_dynamixel_ids(port_handler, packet_handler, max_id=50):
    """Find all connected Dynamixel motors and return their IDs."""
    found_ids = {}
    for motor_id in range(1, max_id):
        # Ping each ID
        dxl_model_number, dxl_comm_result, dxl_error = packet_handler.ping(port_handler, motor_id)
        if dxl_comm_result == COMM_SUCCESS:
            print(f"[INFO] Found Dynamixel ID: {motor_id}, Model Number: {dxl_model_number}")
            dxl_present_position, dxl_comm_result, dxl_error = packet_handler.read2ByteTxRx(port_handler, motor_id, ADDR_PRESENT_POSITION)
            cw_limit,comm_result, dxl_error = packet_handler.read2ByteTxRx(port_handler, motor_id, ADDR_AX_CW_ANGLE_LIMIT)
            ccw_limit,comm_reult, dxl_error = packet_handler.read2ByteTxRx(port_handler, motor_id, ADDR_AX_CCW_ANGLE_LIMIT)
            found_ids[motor_id] = {dxl_present_position, cw_limit, ccw_limit}
        elif dxl_comm_result != COMM_RX_TIMEOUT:
            print(f"[ERROR] Communication error for ID {motor_id}: {packet_handler.getTxRxResult(dxl_comm_result)}")
    return found_ids

def main():
    # Initialize PortHandler and PacketHandler
    port_handler = PortHandler(DEVICENAME)
    packet_handler = PacketHandler(PROTOCOL_VERSION)
    
    # Open port
    if port_handler.openPort():
        print("[INFO] Port opened successfully.")
    else:
        print("[ERROR] Failed to open port. Check connection.")
        return
    
    # Set baudrate
    if port_handler.setBaudRate(BAUDRATE):
        print("[INFO] Baudrate set successfully.")
    else:
        print("[ERROR] Failed to set baudrate.")
        return
    
    try:
        # Find and list all IDs
        ids = find_dynamixel_ids(port_handler, packet_handler)
        if ids:
            print(f"[INFO] Detected Dynamixel IDs: {ids}")
        else:
            print("[INFO] No Dynamixel motors found.")
    finally:
        # Close port
        port_handler.closePort()

if __name__ == "__main__":
    main()




from dynamixel_sdk import *  # Dynamixel SDK library

# Control table address
ADDR_PRESENT_POSITION = 36
LEN_PRESENT_POSITION = 2

# Communication settings
DEVICENAME = "/dev/ttyUSB0"  # Update with your port
BAUDRATE = 1000000          # Default baud rate
PROTOCOL_VERSION = 1.0      # Protocol version for AX series

DXL_ID = 4           # Change to your motor's I

def read_position(port_handler, packet_handler):
    # Read present position
    dxl_present_position, dxl_comm_result, dxl_error = packet_handler.read2ByteTxRx(
        port_handler, DXL_ID, ADDR_PRESENT_POSITION
    )
    
    if dxl_comm_result != COMM_SUCCESS:
        print(f"[ERROR] Communication Error: {packet_handler.getTxRxResult(dxl_comm_result)}")
    elif dxl_error != 0:
        print(f"[ERROR] Error from Dynamixel: {packet_handler.getRxPacketError(dxl_error)}")
    else:
        print(f"[INFO] Current Position: {dxl_present_position}")
        return dxl_present_position

    return None

def main():
    # Initialize PortHandler and PacketHandler
    port_handler = PortHandler(DEVICENAME)
    packet_handler = PacketHandler(PROTOCOL_VERSION)
    
    # Open port
    if not port_handler.openPort():
        print("[ERROR] Failed to open port.")
        return
    
    # Set baudrate
    if not port_handler.setBaudRate(BAUDRATE):
        print("[ERROR] Failed to set baudrate.")
        return
    
    try:
        # Read the position
        position = read_position(port_handler, packet_handler)
        if position is not None:
            print(f"[INFO] Position Read Successfully: {position}")
        else:
            print("[INFO] Could not read position.")
    finally:
        port_handler.closePort()

if __name__ == "__main__":
    main()


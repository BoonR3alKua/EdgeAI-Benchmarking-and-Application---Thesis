"""
CIFAR-100 Image Sender via UART
Sends images from CIFAR-100 dataset to MAX78000 device via UART COM5
"""
import serial
import serial.tools.list_ports
import numpy as np
import tensorflow as tf
import time
import struct
from tqdm import tqdm
import argparse

# ============================================================================
# CONFIGURATION - Change these values as needed
# ============================================================================
SERIAL_PORT = 'COM6'        # Default COM port, will auto-detect if this fails
BAUD_RATE = 115200          # Baud rate for serial communication
NUM_IMAGES = 50             # Number of images to send (benchmark default)
USE_TEST_SET = True         # True = test set, False = training set
TIMEOUT = 5                 # Serial timeout in seconds
ACT_MODE_8BIT = True        # True = [-128, 127], False = [-128/128, 127/128]
USE_CHW_FORMAT = True      # True = CHW (Channels,Height,Width), False = HWC (Height,Width,Channels)
# ============================================================================

# Image Configuration (CIFAR-100 standard)
IMAGE_SIZE = 32
NUM_CHANNELS = 3
BYTES_PER_IMAGE = IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS  # 3072 bytes

# Protocol Configuration - Using unlikely 4-byte sequences
# These markers are chosen to be extremely unlikely in normalized image data
START_MARKER = b'\xDE\xAD\xBE\xEF'  # DEADBEEF - Start of transmission marker
END_MARKER = b'\xCA\xFE\xBA\xBE'    # CAFEBABE - End of transmission marker


def load_cifar100_images(num_images=1000, use_test_set=True):
    """
    Load CIFAR-100 images from TensorFlow datasets
    
    Args:
        num_images: number of images to load
        use_test_set: if True, use test set; otherwise use training set
    
    Returns:
        images: numpy array of shape (num_images, 32, 32, 3)
        labels: numpy array of shape (num_images,)
    """
    print("Loading CIFAR-100 dataset from TensorFlow...")
    
    # Load CIFAR-100 dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
    
    # Use test set or training set
    if use_test_set:
        images = x_test[:num_images]
        labels = y_test[:num_images].flatten()
        print(f"Using test set")
    else:
        images = x_train[:num_images]
        labels = y_train[:num_images].flatten()
        print(f"Using training set")
    
    print(f"Loaded {len(images)} images with shape {images.shape}")
    return images, labels


def normalize_image(image, act_mode_8bit=None):
    """
    Normalize image according to AI8X training normalization.
    AI8X uses: (x - 128) normalization to map [0,255] to [-128,127]
    
    Args:
        image: numpy array of shape (32, 32, 3) with values 0-255
        act_mode_8bit: if None, uses global ACT_MODE_8BIT
                      if True, returns int8 values in [-128, 127]
                      if False, returns float values in [-128/128, 127/128]
    
    Returns:
        normalized image as either int8 or float array
    """
    if act_mode_8bit is None:
        act_mode_8bit = ACT_MODE_8BIT
    
    # AI8X normalization: x - 128
    normalized = image.astype(np.float32) - 128.0
    
    if act_mode_8bit:
        # Return as int8 for 8-bit mode
        return normalized.astype(np.int8)
    else:
        # For float mode, divide by 128 to get [-1, 1) range
        return (normalized / 128.0).astype(np.float32)


def preprocess_image_for_cnn(image, use_chw=None, act_mode_8bit=None):
    """
    Preprocess image to match CNN input format with AI8X normalization.
    
    Args:
        image: numpy array of shape (32, 32, 3) with values 0-255
        use_chw: if None, uses global USE_CHW_FORMAT
                if True, converts to CHW format (Channels, Height, Width)
                if False, keeps HWC format (Height, Width, Channels)
        act_mode_8bit: if None, uses global ACT_MODE_8BIT
    
    Returns:
        processed_data: byte array ready for transmission
    """
    if use_chw is None:
        use_chw = USE_CHW_FORMAT
    if act_mode_8bit is None:
        act_mode_8bit = ACT_MODE_8BIT
    
    # Apply AI8X normalization
    processed = normalize_image(image, act_mode_8bit)
    
    # Convert to CHW format if requested
    if use_chw:
        # Transpose from HWC (32, 32, 3) to CHW (3, 32, 32)
        processed = np.transpose(processed, (2, 0, 1))
    
    # Flatten and convert to bytes
    flattened = processed.flatten()
    
    return flattened.tobytes()


def send_image_via_uart(ser, image_data, image_idx, label):
    """
    Send a single image via UART with protocol framing
    Fixed packet size: START_MARKER(4) + IDX(4) + LABEL(1) + IMAGE(3072) = 3081 bytes
    
    Args:
        ser: serial port object
        image_data: preprocessed image bytes (3072 bytes)
        image_idx: index of the image
        label: ground truth label
        
    Returns:
        success: boolean indicating if transmission was successful
    """
    try:
        # Create fixed-size packet: [START_MARKER(4)][IMAGE_IDX(4)][LABEL(1)][IMAGE_DATA(3072)]
        packet = bytearray()
        packet.extend(START_MARKER)                   # 4 bytes
        packet.extend(struct.pack('<I', image_idx))  # 4 bytes for image index
        packet.extend(struct.pack('<B', label))       # 1 byte for label
        packet.extend(image_data)                     # 3072 bytes for image
        # Total: 3081 bytes (fixed size, no END_MARKER needed)
        
        # Send packet
        bytes_sent = ser.write(packet)
        
        # Wait for device to process and respond
        time.sleep(0.1)  # Give device time to process
        if ser.in_waiting > 0:
            response = ser.read(ser.in_waiting)
            try:
                print(response.decode('utf-8', errors='ignore'), end='')
            except:
                pass
        
        return bytes_sent == len(packet)
        
    except serial.SerialException as e:
        print(f"Serial error: {e}")
        return False


def send_cifar100_images(num_images=None, use_test_set=None,
                        serial_port=None, baudrate=None, timeout=None,
                        use_chw=None, act_mode_8bit=None):
    """
    Main function to send CIFAR-100 images via UART
    
    Args:
        num_images: number of images to send (default: global NUM_IMAGES)
        use_test_set: if True, use test set; otherwise use training set (default: global USE_TEST_SET)
        serial_port: COM port to use (default: global SERIAL_PORT)
        baudrate: baud rate for serial communication (default: global BAUD_RATE)
        timeout: timeout in seconds (default: global TIMEOUT)
        use_chw: CHW format if True, HWC if False (default: global USE_CHW_FORMAT)
        act_mode_8bit: 8-bit mode if True, float mode if False (default: global ACT_MODE_8BIT)
    """
    # Use global defaults if not specified (evaluated at runtime)
    if num_images is None:
        num_images = NUM_IMAGES
    if use_test_set is None:
        use_test_set = USE_TEST_SET
    if serial_port is None:
        serial_port = SERIAL_PORT
    if baudrate is None:
        baudrate = BAUD_RATE
    if timeout is None:
        timeout = TIMEOUT
    if use_chw is None:
        use_chw = USE_CHW_FORMAT
    if act_mode_8bit is None:
        act_mode_8bit = ACT_MODE_8BIT
    
    print("="*60)
    print("CIFAR-100 to MAX78000 UART Sender")
    print("="*60)
    print(f"Configuration:")
    print(f"  - Number of images: {num_images}")
    print(f"  - Dataset: {'Test' if use_test_set else 'Training'} set")
    print(f"  - Serial port: {serial_port}")
    print(f"  - Baud rate: {baudrate}")
    print(f"  - Format: {'CHW (Channels,Height,Width)' if use_chw else 'HWC (Height,Width,Channels)'}")
    print(f"  - Mode: {'8-bit int8 [-128,127]' if act_mode_8bit else 'Float [-1.0,1.0)'}")
    
    # Load images
    images, labels = load_cifar100_images(num_images, use_test_set)
    
    # Auto-detect COM port if default fails
    available_ports = []
    for port in serial.tools.list_ports.comports():
        available_ports.append(f"{port.device}: {port.description}")
    
    print("\nAvailable COM ports:")
    for port_info in available_ports:
        print(f"  - {port_info}")
    
    # Try to open serial port
    port_to_use = serial_port
    ser = None
    
    try:
        print(f"\nTrying to open serial port {port_to_use} at {baudrate} baud...")
        ser = serial.Serial(
            port=port_to_use,
            baudrate=baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=timeout,
            write_timeout=timeout
        )
    except serial.SerialException as e:
        print(f"\n❌ Error opening {port_to_use}: {e}")
        
        # Try to auto-detect MAX78000
        print("\nAttempting to auto-detect MAX78000...")
        for port in serial.tools.list_ports.comports():
            if "USB Serial" in port.description or "MAXIM" in port.description.upper() or "MAX78000" in port.description.upper():
                try:
                    print(f"  Trying {port.device}...")
                    ser = serial.Serial(
                        port=port.device,
                        baudrate=baudrate,
                        bytesize=serial.EIGHTBITS,
                        parity=serial.PARITY_NONE,
                        stopbits=serial.STOPBITS_ONE,
                        timeout=timeout,
                        write_timeout=timeout
                    )
                    port_to_use = port.device
                    print(f"  ✓ Successfully opened {port.device}")
                    break
                except:
                    continue
        
        if ser is None:
            print("\n❌ Could not open any COM port!")
            print("Please check:")
            print("1. Device is connected via USB")
            print("2. Close any serial monitors (PuTTY, TeraTerm, VS Code Serial Monitor)")
            print("3. Drivers are installed correctly")
            print("4. Try unplugging and replugging the device")
            return False
    
    if ser is None:
        return False
    
    # Successfully opened port
    print(f"Serial port opened successfully")
    print(f"Port: {ser.name}")
    print(f"Baud rate: {ser.baudrate}")
    
    # Wait for device to be ready
    time.sleep(2)
    
    # Clear any existing data in buffers
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    
    # Send images
    print(f"\nSending {num_images} images...")
    success_count = 0
    fail_count = 0
    
    start_time = time.time()
    
    for idx in tqdm(range(num_images), desc="Sending images"):
        # Preprocess image with current format settings
        image_data = preprocess_image_for_cnn(images[idx], use_chw=use_chw, act_mode_8bit=act_mode_8bit)
        
        # Send image
        if send_image_via_uart(ser, image_data, idx, labels[idx]):
            success_count += 1
        else:
            fail_count += 1
            print(f"\nFailed to send image {idx}")
        
        # Small delay between images to prevent buffer overflow
        time.sleep(0.5)  # 500ms delay to prevent packet loss
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Statistics
    print("\n" + "="*60)
    print("Transmission Complete")
    print("="*60)
    print(f"Total images sent: {success_count}/{num_images}")
    print(f"Failed transmissions: {fail_count}")
    print(f"Total time: {elapsed_time:.2f} seconds")
    print(f"Average time per image: {elapsed_time/num_images*1000:.2f} ms")
    print(f"Throughput: {num_images/elapsed_time:.2f} images/second")
    print(f"Data rate: {(BYTES_PER_IMAGE*num_images)/(elapsed_time*1024):.2f} KB/s")
    
    # Close serial port
    ser.close()
    print("\nSerial port closed")
    
    return True


def main():
    """
    Main entry point with command line arguments
    """
    parser = argparse.ArgumentParser(description='Send CIFAR-100 images to MAX78000 via UART')
    parser.add_argument('--num-images', type=int,
                        help=f'Number of images to send (default: {NUM_IMAGES})')
    parser.add_argument('--train-set', action='store_true',
                        help='Use training set instead of test set')
    parser.add_argument('--port', type=str,
                        help=f'Serial port to use (default: {SERIAL_PORT})')
    parser.add_argument('--baudrate', type=int,
                        help=f'Baud rate (default: {BAUD_RATE})')
    parser.add_argument('--timeout', type=int,
                        help=f'Serial timeout in seconds (default: {TIMEOUT})')
    parser.add_argument('--hwc', action='store_true',
                        help='Use HWC format instead of CHW (for testing)')
    parser.add_argument('--float-mode', action='store_true',
                        help='Use float mode [-1,1) instead of int8 [-128,127]')
    
    args = parser.parse_args()
    
    print("Required packages: pyserial, tensorflow, numpy, tqdm")
    print("Install with: pip install pyserial tensorflow numpy tqdm\n")
    
    # Determine format and mode from arguments
    # CIFAR-100 models use HWC format by default (verified from cnn.c: "Input data: HWC")
    # Use --hwc flag only for testing/debugging with models that explicitly need HWC
    use_chw = False  # Default to HWC (Height, Width, Channels)
    act_mode_8bit = not args.float_mode  # Default is 8-bit unless --float-mode is specified
    
    success = send_cifar100_images(
        num_images=args.num_images,
        use_test_set=not args.train_set,
        serial_port=args.port,
        baudrate=args.baudrate,
        timeout=args.timeout,
        use_chw=use_chw,
        act_mode_8bit=act_mode_8bit
    )
    
    if success:
        print("\n✓ Program completed successfully")
    else:
        print("\n✗ Program failed")


if __name__ == "__main__":
    main()
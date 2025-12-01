"""
CIFAR-100 Image Sender via UART
Sends images from CIFAR-100 dataset to MAX78000 device via UART
For use with SimpleNet/MobileNetV2/EfficientNetV2 models that expect 32x32 input
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
# CONFIGURATION SETTINGS
# ============================================================================
SERIAL_PORT = 'COM6'        # Default COM port, will auto-detect if this fails
BAUD_RATE = 115200          # Baud rate for serial communication
NUM_IMAGES = 50             # Number of images to send (benchmark default)
USE_TEST_SET = True         # True = test set, False = training set
TIMEOUT = 5                 # Serial timeout in seconds
ACT_MODE_8BIT = True        # True = [-128, 127], False = [-128/128, 127/128]
USE_CHW_FORMAT = True       # True = CHW (Channels,Height,Width), False = HWC (Height,Width,Channels)
                            # CHW: Data sent as [R channel 1024 bytes][G channel 1024 bytes][B channel 1024 bytes]
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


def normalize_image(image, act_mode_8bit=True):
    """
    Normalize image using AI8X standard normalization
    Converts from [0, 255] to [-128, 127] using simple subtraction
    
    Args:
        image: numpy array of shape (32, 32, 3) with values 0-255
        act_mode_8bit: if True, output is [-128, 127]; if False, output is [-1, 1]
        
    Returns:
        normalized: numpy array with normalized values
    """
    # Convert to float for normalization
    img = image.astype(np.float32)
    
    # AI8X-style normalization: simply subtract 128 to center at 0
    # This maps [0, 255] -> [-128, 127]
    img = img - 128.0
    img = np.clip(img, -128, 127)  # Clamp to valid range
    
    if not act_mode_8bit:
        # Divide by 128 to get [-1, 1] range
        img = img / 128.0
    
    return img


def preprocess_image_for_cnn(image, act_mode_8bit=None, use_chw=None):
    """
    Preprocess image to match the CNN input format with proper normalization
    
    Args:
        image: numpy array of shape (32, 32, 3) with values 0-255 in HWC format
        act_mode_8bit: if True, normalize to [-128, 127]; if False, to [-1, 1]
        use_chw: if True, convert to CHW format; if False, keep HWC format
        
    Returns:
        processed_data: byte array ready for transmission
    """
    # Use global defaults if not specified
    if act_mode_8bit is None:
        act_mode_8bit = ACT_MODE_8BIT
    if use_chw is None:
        use_chw = USE_CHW_FORMAT
    
    # Verify input is in HWC format
    assert image.shape == (IMAGE_SIZE, IMAGE_SIZE, 3), \
        f"Expected HWC format ({IMAGE_SIZE}, {IMAGE_SIZE}, 3), got {image.shape}"
    
    # Apply normalization (same as training)
    normalized = normalize_image(image, act_mode_8bit)
    
    if act_mode_8bit:
        # Convert to int8 for 8-bit mode
        normalized = normalized.astype(np.int8)
        # Convert to uint8 for transmission (reinterpret bytes)
        processed = normalized.view(np.uint8)
    else:
        # For float mode, we need to quantize back to int8 range for transmission
        # The device will interpret these as signed values
        quantized = (normalized * 128.0).astype(np.int8)
        processed = quantized.view(np.uint8)
    
    if use_chw:
        # CHW format: Channels, Height, Width (3x32x32)
        # All R values first, then all G values, then all B values
        processed_chw = np.transpose(processed, (2, 0, 1))
        flattened = processed_chw.flatten(order='C')
    else:
        # HWC format: Height, Width, Channels (32x32x3)
        # Interleaved R,G,B per pixel: R0,G0,B0, R1,G1,B1, ...
        flattened = processed.flatten(order='C')
    
    return flattened.tobytes()


def send_image_via_uart(ser, image_data, image_idx, label):
    """
    Send a single image via UART with protocol framing
    
    Args:
        ser: serial port object
        image_data: preprocessed image bytes (3072 bytes for 32x32)
        image_idx: index of the image
        label: ground truth label
        
    Returns:
        success: boolean indicating if transmission was successful
    """
    try:
        # Create packet: [START_MARKER][IMAGE_IDX][LABEL][IMAGE_DATA]
        # No end marker needed - fixed packet size provides framing
        packet = bytearray()
        packet.extend(START_MARKER)  # 4 bytes: DEADBEEF
        packet.extend(struct.pack('<I', image_idx))  # 4 bytes for image index
        packet.extend(struct.pack('<B', label))       # 1 byte for label
        packet.extend(image_data)                     # 3072 bytes for 32x32 image
        
        # Send packet
        bytes_sent = ser.write(packet)
        ser.flush()  # Ensure all data is transmitted
        
        # Wait for device to process
        # 3072 bytes at 115200 baud (with 10 bits per byte) = ~0.27 seconds
        # Add margin for CNN inference (~3ms) and printing
        time.sleep(0.5)  # Wait for transmission + processing
        
        # Read all available response from device
        device_response = bytearray()
        max_retries = 10
        retry_count = 0
        
        # Keep reading until we get the complete response
        while retry_count < max_retries:
            if ser.in_waiting > 0:
                chunk = ser.read(ser.in_waiting)
                device_response.extend(chunk)
                retry_count = 0  # Reset retry counter when we get data
            else:
                retry_count += 1
                time.sleep(0.1)  # Short wait between retries
        
        # Display device output
        if device_response:
            try:
                device_output = device_response.decode('utf-8', errors='ignore')
                print(device_output, end='')  # Device output already has newlines
            except Exception as e:
                print(f"Error decoding device response: {e}")
        
        return bytes_sent == len(packet)
        
    except serial.SerialException as e:
        print(f"Serial error: {e}")
        return False


def open_serial_port(port=SERIAL_PORT, baudrate=BAUD_RATE, timeout=TIMEOUT):
    """
    Open serial port with auto-detection fallback
    
    Args:
        port: COM port to use
        baudrate: baud rate for serial communication
        timeout: timeout in seconds
    
    Returns:
        ser: serial port object or None if failed
        port_name: name of the opened port
    """
    # Show available COM ports
    available_ports = []
    for port_info in serial.tools.list_ports.comports():
        available_ports.append(f"{port_info.device}: {port_info.description}")
    
    print("\nAvailable COM ports:")
    for port_info in available_ports:
        print(f"  - {port_info}")
    
    # Try to open serial port
    ser = None
    port_to_use = port
    
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
        for port_info in serial.tools.list_ports.comports():
            if ("USB Serial" in port_info.description or 
                "MAXIM" in port_info.description.upper() or 
                "MAX78000" in port_info.description.upper()):
                try:
                    print(f"  Trying {port_info.device}...")
                    ser = serial.Serial(
                        port=port_info.device,
                        baudrate=baudrate,
                        bytesize=serial.EIGHTBITS,
                        parity=serial.PARITY_NONE,
                        stopbits=serial.STOPBITS_ONE,
                        timeout=timeout,
                        write_timeout=timeout
                    )
                    port_to_use = port_info.device
                    print(f"  ✓ Successfully opened {port_info.device}")
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
            return None, None
    
    if ser is None:
        return None, None
    
    # Successfully opened port
    print(f"Serial port opened successfully")
    print(f"Port: {ser.name}")
    print(f"Baud rate: {ser.baudrate}")
    
    return ser, port_to_use


def send_cifar100_images(num_images=None, use_test_set=None,
                         serial_port=None, baudrate=None, timeout=None,
                         act_mode_8bit=None, use_chw=None):
    """
    Main function to send CIFAR-100 images via UART
    
    Args:
        num_images: number of images to send
        use_test_set: if True, use test set; otherwise use training set
        serial_port: COM port to use
        baudrate: baud rate for serial communication
        timeout: timeout in seconds
        act_mode_8bit: if True, use 8-bit mode; if False, use float mode
        use_chw: if True, convert to CHW format; if False, keep HWC format
    """
    # Use global defaults if not specified
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
    if act_mode_8bit is None:
        act_mode_8bit = ACT_MODE_8BIT
    if use_chw is None:
        use_chw = USE_CHW_FORMAT
    
    print("="*70)
    print("CIFAR-100 to MAX78000 UART Sender")
    print("="*70)
    print(f"Configuration:")
    print(f"  - Number of images: {num_images}")
    print(f"  - Dataset: {'Test' if use_test_set else 'Training'} set")
    print(f"  - Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"  - Bytes per image: {BYTES_PER_IMAGE} bytes")
    print(f"  - Serial port: {serial_port}")
    print(f"  - Baud rate: {baudrate}")
    print(f"  - Normalization: {'8-bit [-128, 127]' if act_mode_8bit else 'Float [-1, 1]'}")
    print(f"  - Data format: {'CHW (Channels,Height,Width)' if use_chw else 'HWC (Height,Width,Channels)'}")
    
    # Load images
    images, labels = load_cifar100_images(num_images, use_test_set)
    
    # Open serial port
    ser, port_name = open_serial_port(serial_port, baudrate, timeout)
    if ser is None:
        return False
    
    # Wait for device to be ready
    time.sleep(2)
    
    # Clear any existing data in buffers
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    
    # Send images
    print(f"\nSending {num_images} images...\n")
    success_count = 0
    fail_count = 0
    
    start_time = time.time()
    
    for idx in tqdm(range(num_images), desc="Sending images"):
        # Preprocess image with normalization and format conversion
        image_data = preprocess_image_for_cnn(images[idx], act_mode_8bit, use_chw)
        
        # Send image
        if send_image_via_uart(ser, image_data, idx, labels[idx]):
            success_count += 1
        else:
            fail_count += 1
            print(f"\nFailed to send image {idx}")
        
        # Small additional delay to ensure device is ready for next packet
        time.sleep(0.2)  # 200ms extra margin
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Statistics
    print("\n" + "="*70)
    print("BENCHMARK RESULTS - TRANSMISSION STATISTICS")
    print("="*70)
    print(f"Image size:               {IMAGE_SIZE}x{IMAGE_SIZE}x{NUM_CHANNELS}")
    print(f"Total images sent:        {success_count}/{num_images}")
    print(f"Failed transmissions:     {fail_count}")
    print(f"Total time:               {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print(f"Average time per image:   {elapsed_time/num_images:.2f} seconds ({elapsed_time/num_images*1000:.2f} ms)")
    print(f"Transmission throughput:  {num_images/elapsed_time:.2f} images/second")
    print(f"Data rate:                {(BYTES_PER_IMAGE*num_images)/(elapsed_time*1024):.2f} KB/s")
    print(f"Total data transmitted:   {(BYTES_PER_IMAGE*num_images)/1024:.2f} KB ({(BYTES_PER_IMAGE*num_images)/(1024*1024):.3f} MB)")
    print("="*70)
    
    # Close serial port
    ser.close()
    print("\nSerial port closed")
    
    return True


def main():
    """
    Main entry point with command line arguments
    """
    parser = argparse.ArgumentParser(description='Send CIFAR-100 images to MAX78000 via UART')
    parser.add_argument('--num-images', type=int, default=None,
                        help=f'Number of images to send (default: {NUM_IMAGES})')
    parser.add_argument('--train-set', action='store_true',
                        help='Use training set instead of test set')
    parser.add_argument('--port', type=str, default=None,
                        help=f'Serial port to use (default: {SERIAL_PORT})')
    parser.add_argument('--baudrate', type=int, default=None,
                        help=f'Baud rate (default: {BAUD_RATE})')
    parser.add_argument('--timeout', type=int, default=None,
                        help=f'Serial timeout in seconds (default: {TIMEOUT})')
    parser.add_argument('--float-mode', action='store_true',
                        help='Use float normalization [-1, 1] instead of 8-bit [-128, 127]')
    parser.add_argument('--hwc', action='store_true',
                        help='Use HWC (Height,Width,Channels) format instead of CHW (Channels,Height,Width)')
    
    args = parser.parse_args()
    
    print("Required packages: pyserial, tensorflow, numpy, tqdm")
    print("Install with: pip install pyserial tensorflow numpy tqdm\n")
    
    # Determine format settings
    # Default is CHW format (USE_CHW_FORMAT = True)
    use_chw = not args.hwc  # True (CHW) unless --hwc flag is set
    act_mode_8bit = not args.float_mode  # True (8-bit) unless --float-mode flag is set
    
    success = send_cifar100_images(
        num_images=args.num_images,
        use_test_set=not args.train_set,  # True (test set) unless --train-set flag is set
        serial_port=args.port,
        baudrate=args.baudrate,
        timeout=args.timeout,
        act_mode_8bit=act_mode_8bit,
        use_chw=use_chw
    )
    
    if success:
        print("\n✓ Program completed successfully")
    else:
        print("\n✗ Program failed")


if __name__ == "__main__":
    main()

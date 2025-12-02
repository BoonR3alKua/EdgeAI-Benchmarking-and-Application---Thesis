"""
CIFAR-100 Image Sender via UART - Queue-based Benchmark Version
Sends all images first, then device processes them continuously.
Protocol: [DEADBEEF][NUM_IMAGES(4)][Label0][Image0][Label1][Image1]...
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
TIMEOUT = 30                # Serial timeout in seconds (longer for batch processing)
ACT_MODE_8BIT = True        # True = [-128, 127], False = [-128/128, 127/128]
USE_CHW_FORMAT = True       # True = CHW (Channels,Height,Width), False = HWC (Height,Width,Channels)
MAX_QUEUE_SIZE = 10         # Must match device MAX_QUEUE_SIZE (limited by SRAM)
# ============================================================================

# Image Configuration (CIFAR-100 standard)
IMAGE_SIZE = 32
NUM_CHANNELS = 3
BYTES_PER_IMAGE = IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS  # 3072 bytes

# Protocol Configuration
START_MARKER = b'\xDE\xAD\xBE\xEF'  # DEADBEEF - Start of transmission marker


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


def send_batch_header(ser, num_images):
    """
    Send batch header with start marker and number of images.
    Protocol: [DEADBEEF][NUM_IMAGES(4)]
    """
    header = bytearray()
    header.extend(START_MARKER)  # 4 bytes: DEADBEEF
    header.extend(struct.pack('<I', num_images))  # 4 bytes: number of images (little-endian)
    
    bytes_sent = ser.write(header)
    ser.flush()
    return bytes_sent == len(header)


def wait_for_ready(ser, timeout_seconds=10):
    """
    Wait for 'READY' signal from device before sending images.
    This ensures device is ready to receive data.
    """
    start_time = time.time()
    response = ""
    
    while (time.time() - start_time) < timeout_seconds:
        if ser.in_waiting > 0:
            chunk = ser.read(ser.in_waiting)
            try:
                text = chunk.decode('utf-8', errors='ignore')
                response += text
                print(text, end='', flush=True)  # Echo device output
                
                if 'READY' in response:
                    return True
            except:
                pass
        else:
            time.sleep(0.01)
    
    return False


def send_image_data(ser, image_data, label):
    """
    Send a single image (label + image data) without start marker.
    Protocol: [LABEL(1)][IMAGE_DATA(3072)]
    """
    packet = bytearray()
    packet.extend(struct.pack('<B', label))  # 1 byte for label
    packet.extend(image_data)                 # 3072 bytes for image
    
    bytes_sent = ser.write(packet)
    ser.flush()
    return bytes_sent == len(packet)


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


def read_device_output(ser, timeout_seconds=60):
    """
    Read and display all output from device until benchmark completes.
    Looks for [DONE] marker to know when benchmark is finished.
    """
    start_time = time.time()
    device_response = bytearray()
    benchmark_done = False
    
    print("\n" + "="*70)
    print("DEVICE OUTPUT")
    print("="*70)
    
    while not benchmark_done and (time.time() - start_time) < timeout_seconds:
        if ser.in_waiting > 0:
            chunk = ser.read(ser.in_waiting)
            device_response.extend(chunk)
            
            # Decode and print incrementally
            try:
                text = chunk.decode('utf-8', errors='ignore')
                print(text, end='', flush=True)
                
                # Check if benchmark is done
                if '[DONE]' in device_response.decode('utf-8', errors='ignore'):
                    benchmark_done = True
            except:
                pass
        else:
            time.sleep(0.01)  # Small delay to avoid busy-waiting
    
    print("\n" + "="*70)
    
    return benchmark_done


def send_cifar100_benchmark(num_images=None, use_test_set=None,
                            serial_port=None, baudrate=None, timeout=None,
                            act_mode_8bit=None, use_chw=None):
    """
    Send CIFAR-100 images as a batch for benchmarking.
    Keeps port open and runs continuously until Ctrl+C.
    New protocol: [DEADBEEF][NUM_IMAGES(4)][Label0][Image0][Label1][Image1]...
    
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
    
    # Limit to max queue size
    if num_images > MAX_QUEUE_SIZE:
        print(f"Warning: Limiting images to {MAX_QUEUE_SIZE} (device queue limit)")
        num_images = MAX_QUEUE_SIZE
    
    print("="*70)
    print("CIFAR-100 Benchmark Sender (Queue-based Protocol)")
    print("="*70)
    print(f"Configuration:")
    print(f"  - Number of images: {num_images}")
    print(f"  - Dataset: {'Test' if use_test_set else 'Training'} set")
    print(f"  - Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"  - Bytes per image: {BYTES_PER_IMAGE} bytes")
    print(f"  - Total data: {(BYTES_PER_IMAGE * num_images) / 1024:.2f} KB")
    print(f"  - Serial port: {serial_port}")
    print(f"  - Baud rate: {baudrate}")
    print(f"  - Normalization: {'8-bit [-128, 127]' if act_mode_8bit else 'Float [-1, 1]'}")
    print(f"  - Data format: {'CHW' if use_chw else 'HWC'}")
    
    # Load images
    images, labels = load_cifar100_images(num_images, use_test_set)
    
    # Preprocess all images first (do this before opening port)
    print(f"\nPreprocessing {num_images} images...")
    preprocessed_images = []
    for idx in tqdm(range(num_images), desc="Preprocessing"):
        image_data = preprocess_image_for_cnn(images[idx], act_mode_8bit, use_chw)
        preprocessed_images.append((image_data, labels[idx]))
    
    # Main loop - keep trying to open port and run benchmarks
    ser = None
    run_count = 0
    
    print("\n" + "="*70)
    print("Starting continuous benchmark mode (Press Ctrl+C to stop)")
    print("="*70)
    
    try:
        while True:
            # Open serial port if not already open
            if ser is None or not ser.is_open:
                ser, port_name = open_serial_port(serial_port, baudrate, timeout)
                if ser is None:
                    print("\nRetrying in 3 seconds...")
                    time.sleep(3)
                    continue
                
                # Wait for device to be ready after opening port
                print("\nWaiting for device to be ready...")
                time.sleep(2)
            
            run_count += 1
            print(f"\n{'='*70}")
            print(f"BENCHMARK RUN #{run_count}")
            print(f"{'='*70}")
            
            # Clear any existing data in buffers
            ser.reset_input_buffer()
            ser.reset_output_buffer()
            
            # Send batch header
            print(f"\nSending batch header (N={num_images})...")
            if not send_batch_header(ser, num_images):
                print("Failed to send batch header! Reconnecting...")
                ser.close()
                ser = None
                time.sleep(1)
                continue
            
            # Wait for device to signal it's ready to receive
            print("Waiting for device READY signal...")
            if not wait_for_ready(ser, timeout_seconds=10):
                print("Device did not respond with READY! Reconnecting...")
                ser.close()
                ser = None
                time.sleep(1)
                continue
            
            print("Device ready, sending images...")
            
            # Send all images
            print(f"Sending {num_images} images...")
            tx_start_time = time.time()
            
            send_failed = False
            for idx, (image_data, label) in enumerate(tqdm(preprocessed_images, desc="Transmitting")):
                if not send_image_data(ser, image_data, label):
                    print(f"\nFailed to send image {idx}! Reconnecting...")
                    send_failed = True
                    break
            
            if send_failed:
                ser.close()
                ser = None
                time.sleep(1)
                continue
            
            tx_end_time = time.time()
            tx_time = tx_end_time - tx_start_time
            
            print(f"\n✓ All {num_images} images sent in {tx_time:.2f}s")
            print(f"  Transmission rate: {(BYTES_PER_IMAGE * num_images) / tx_time / 1024:.2f} KB/s")
            
            # Wait for device to process all images and report results
            print("\nWaiting for device to complete benchmark...")
            benchmark_complete = read_device_output(ser, timeout_seconds=120)
            
            if benchmark_complete:
                print(f"\n✓ Benchmark run #{run_count} completed successfully!")
            else:
                print(f"\n⚠ Benchmark run #{run_count} may not have completed (timeout)")
            
            # Small delay before next run
            print("\nWaiting 10 seconds before next run... (Ctrl+C to stop)")
            time.sleep(10)
            
    except KeyboardInterrupt:
        print("\n\n" + "="*70)
        print("Keyboard interrupt received. Stopping...")
        print(f"Total benchmark runs completed: {run_count}")
        print("="*70)
    
    finally:
        # Close serial port
        if ser is not None and ser.is_open:
            ser.close()
            print("Serial port closed")
    
    return True


def main():
    """
    Main entry point with command line arguments
    """
    parser = argparse.ArgumentParser(description='Send CIFAR-100 images to MAX78000 via UART (Benchmark Mode)')
    parser.add_argument('--num-images', type=int, default=None,
                        help=f'Number of images to send (default: {NUM_IMAGES}, max: {MAX_QUEUE_SIZE})')
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
                        help='Use HWC (Height,Width,Channels) format instead of CHW')
    
    args = parser.parse_args()
    
    print("="*70)
    print("CIFAR-100 Benchmark Tool for MAX78000")
    print("="*70)
    print("Required packages: pyserial, tensorflow, numpy, tqdm")
    print("Install with: pip install pyserial tensorflow numpy tqdm\n")
    
    # Determine format settings
    use_chw = not args.hwc
    act_mode_8bit = not args.float_mode
    
    success = send_cifar100_benchmark(
        num_images=args.num_images,
        use_test_set=not args.train_set,
        serial_port=args.port,
        baudrate=args.baudrate,
        timeout=args.timeout,
        act_mode_8bit=act_mode_8bit,
        use_chw=use_chw
    )
    
    if success:
        print("\n✓ Benchmark session completed successfully")
    else:
        print("\n✗ Benchmark session failed")


if __name__ == "__main__":
    main()

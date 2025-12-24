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
NUM_IMAGES = 1000           # Total number of images for full benchmark
BATCH_SIZE = 20             # Images per batch (must match device MAX_QUEUE_SIZE)
NUM_BATCHES = 50            # Number of batches (1000 / 20 = 50)
USE_TEST_SET = True         # True = test set, False = training set
TIMEOUT = 30                # Serial timeout in seconds (longer for batch processing)
ACT_MODE_8BIT = True        # True = [-128, 127], False = [-128/128, 127/128]
USE_CHW_FORMAT = True       # True = CHW (Channels,Height,Width), False = HWC (Height,Width,Channels)
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


def read_device_output(ser, timeout_seconds=60, batch_num=None):
    """
    Read and display all output from device until batch completes.
    Looks for [BATCH X/50 COMPLETE] marker for individual batches,
    or [DONE] for final completion.
    """
    start_time = time.time()
    device_response = bytearray()
    batch_done = False
    
    print("\n" + "="*70)
    print("DEVICE OUTPUT")
    print("="*70)
    
    while not batch_done and (time.time() - start_time) < timeout_seconds:
        if ser.in_waiting > 0:
            chunk = ser.read(ser.in_waiting)
            device_response.extend(chunk)
            
            # Decode and print incrementally
            try:
                text = chunk.decode('utf-8', errors='ignore')
                print(text, end='', flush=True)
                
                response_str = device_response.decode('utf-8', errors='ignore')
                
                # Check if batch is complete
                if batch_num is not None:
                    # Looking for specific batch completion
                    if f'[BATCH {batch_num}/50 COMPLETE]' in response_str:
                        batch_done = True
                else:
                    # Looking for final done marker
                    if '[DONE]' in response_str or 'BENCHMARK COMPLETE' in response_str:
                        batch_done = True
            except:
                pass
        else:
            time.sleep(0.01)  # Small delay to avoid busy-waiting
    
    print("\n" + "="*70)
    
    return batch_done


def send_cifar100_benchmark_multibatch(num_images=None, batch_size=None, use_test_set=None,
                                       serial_port=None, baudrate=None, timeout=None,
                                       act_mode_8bit=None, use_chw=None):
    """
    Send CIFAR-100 images in multiple batches for benchmarking 1000 images.
    Sends images in batches of BATCH_SIZE, device processes each batch, then requests next.
    Protocol: For each batch: [DEADBEEF][NUM_IMAGES(4)][Label0][Image0][Label1][Image1]...
    
    Args:
        num_images: total number of images to send (default: 1000)
        batch_size: number of images per batch (default: 20)
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
    if batch_size is None:
        batch_size = BATCH_SIZE
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
    
    # Calculate number of batches
    num_batches = (num_images + batch_size - 1) // batch_size  # Ceiling division
    
    print("="*70)
    print("CIFAR-100 Multi-Batch Benchmark Sender")
    print("="*70)
    print(f"Configuration:")
    print(f"  - Total images:      {num_images}")
    print(f"  - Batch size:        {batch_size} images")
    print(f"  - Number of batches: {num_batches}")
    print(f"  - Dataset:           {'Test' if use_test_set else 'Training'} set")
    print(f"  - Image size:        {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"  - Bytes per image:   {BYTES_PER_IMAGE} bytes")
    print(f"  - Total data:        {(BYTES_PER_IMAGE * num_images) / 1024:.2f} KB")
    print(f"  - Serial port:       {serial_port}")
    print(f"  - Baud rate:         {baudrate}")
    print(f"  - Normalization:     {'8-bit [-128, 127]' if act_mode_8bit else 'Float [-1, 1]'}")
    print(f"  - Data format:       {'CHW' if use_chw else 'HWC'}")
    
    # Load all images
    print(f"\nLoading {num_images} images from dataset...")
    images, labels = load_cifar100_images(num_images, use_test_set)
    
    # Preprocess all images first (do this before opening port)
    print(f"Preprocessing {num_images} images...")
    preprocessed_images = []
    for idx in tqdm(range(num_images), desc="Preprocessing"):
        image_data = preprocess_image_for_cnn(images[idx], act_mode_8bit, use_chw)
        preprocessed_images.append((image_data, labels[idx]))
    
    # Open serial port
    ser, port_name = open_serial_port(serial_port, baudrate, timeout)
    if ser is None:
        print("Failed to open serial port!")
        return False
    
    # Wait for device to be ready
    print("\nWaiting for device to initialize...")
    time.sleep(2)
    
    print("\n" + "="*70)
    print(f"Starting 1000-image benchmark ({num_batches} batches)")
    print("="*70 + "\n")
    
    overall_start_time = time.time()
    total_tx_time = 0
    
    try:
        # Send each batch
        for batch_idx in range(num_batches):
            batch_num = batch_idx + 1
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_images)
            current_batch_size = end_idx - start_idx
            
            print(f"\n{'='*70}")
            print(f"BATCH {batch_num}/{num_batches} (Images {start_idx+1}-{end_idx})")
            print(f"{'='*70}")
            
            # Clear buffers
            ser.reset_input_buffer()
            ser.reset_output_buffer()
            
            # Send batch header
            print(f"Sending batch header ({current_batch_size} images)...")
            if not send_batch_header(ser, current_batch_size):
                print(f"Failed to send batch {batch_num} header!")
                return False
            
            # Wait for device READY signal
            print("Waiting for device READY signal...")
            if not wait_for_ready(ser, timeout_seconds=10):
                print(f"Device did not respond for batch {batch_num}!")
                return False
            
            # Send images for this batch
            print(f"Transmitting {current_batch_size} images...")
            batch_tx_start = time.time()
            
            for img_idx in tqdm(range(start_idx, end_idx), desc=f"Batch {batch_num}/{num_batches}"):
                image_data, label = preprocessed_images[img_idx]
                if not send_image_data(ser, image_data, label):
                    print(f"\nFailed to send image {img_idx}!")
                    return False
            
            batch_tx_time = time.time() - batch_tx_start
            total_tx_time += batch_tx_time
            
            print(f"✓ Batch {batch_num} sent in {batch_tx_time:.2f}s " + 
                  f"({(BYTES_PER_IMAGE * current_batch_size) / batch_tx_time / 1024:.1f} KB/s)")
            
            # Wait for device to process this batch
            print(f"Device processing batch {batch_num}...")
            
            # Read device output for this batch (shortened timeout per batch)
            batch_complete = read_device_output(ser, timeout_seconds=60, batch_num=batch_num)
            
            if not batch_complete:
                print(f"⚠ Warning: Batch {batch_num} processing may have timed out")
            
            # Small delay between batches
            if batch_num < num_batches:
                time.sleep(0.5)
        
        # All batches sent, wait for final report
        print(f"\n{'='*70}")
        print("All batches sent! Waiting for final statistics...")
        print(f"{'='*70}\n")
        
        # Give device extra time to print final report
        time.sleep(2)
        
        # Read final statistics
        final_output_start = time.time()
        while (time.time() - final_output_start) < 30:
            if ser.in_waiting > 0:
                chunk = ser.read(ser.in_waiting)
                try:
                    text = chunk.decode('utf-8', errors='ignore')
                    print(text, end='', flush=True)
                except:
                    pass
            else:
                time.sleep(0.01)
        
        overall_time = time.time() - overall_start_time
        
        print(f"\n{'='*70}")
        print("BENCHMARK COMPLETE")
        print(f"{'='*70}")
        print(f"Total images:        {num_images}")
        print(f"Total batches:       {num_batches}")
        print(f"Total TX time:       {total_tx_time:.2f}s")
        print(f"Overall time:        {overall_time:.2f}s ({overall_time/60:.1f} min)")
        print(f"Avg TX rate:         {(BYTES_PER_IMAGE * num_images) / total_tx_time / 1024:.1f} KB/s")
        print(f"{'='*70}\n")
        
        return True
        
    except KeyboardInterrupt:
        print("\n\n" + "="*70)
        print("Keyboard interrupt received. Stopping...")
        print("="*70)
        return False
    
    finally:
        # Close serial port
        if ser is not None and ser.is_open:
            ser.close()
            print("Serial port closed")


def main():
    """
    Main entry point with command line arguments
    """
    parser = argparse.ArgumentParser(description='Send CIFAR-100 images to MAX78000 via UART (Multi-Batch Benchmark)')
    parser.add_argument('--num-images', type=int, default=None,
                        help=f'Total number of images to send (default: {NUM_IMAGES})')
    parser.add_argument('--batch-size', type=int, default=None,
                        help=f'Images per batch (default: {BATCH_SIZE})')
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
    print("CIFAR-100 Multi-Batch Benchmark Tool for MAX78000")
    print("="*70)
    print("Required packages: pyserial, tensorflow, numpy, tqdm")
    print("Install with: pip install pyserial tensorflow numpy tqdm\n")
    
    # Determine format settings
    use_chw = not args.hwc
    act_mode_8bit = not args.float_mode
    
    success = send_cifar100_benchmark_multibatch(
        num_images=args.num_images,
        batch_size=args.batch_size,
        use_test_set=not args.train_set,
        serial_port=args.port,
        baudrate=args.baudrate,
        timeout=args.timeout,
        act_mode_8bit=act_mode_8bit,
        use_chw=use_chw
    )
    
    if success:
        print("\n✓ Benchmark completed successfully")
    else:
        print("\n✗ Benchmark failed")


if __name__ == "__main__":
    main()

"""
CIFAR-10 Image Sender via UART (64x64 Scaled Version) - Queue-based Benchmark
Sends images from CIFAR-10 dataset scaled up to 64x64 to MAX78000 device via UART
For use with AlexNet-Small model that expects 64x64 input
Protocol: [DEADBEEF][NUM_IMAGES(4)][Label0][Image0][Label1][Image1]...
"""
import serial
import serial.tools.list_ports
import numpy as np
import tensorflow as tf
import time
import struct
from PIL import Image
from tqdm import tqdm
import argparse

# ============================================================================
# CONFIGURATION SETTINGS
# ============================================================================
SERIAL_PORT = 'COM6'        # Default COM port, will auto-detect if this fails
BAUD_RATE = 115200          # Baud rate for serial communication
NUM_IMAGES = 8              # Number of images to send (benchmark default)
USE_TEST_SET = True         # True = test set, False = training set
TIMEOUT = 30                # Serial timeout in seconds (longer for batch processing)
ACT_MODE_8BIT = True        # True = [-128, 127], False = [-128/128, 127/128]
USE_CHW_FORMAT = True       # True = CHW (Channels,Height,Width), False = HWC (Height,Width,Channels)
MAX_QUEUE_SIZE = 8          # Must match device MAX_QUEUE_SIZE (limited by SRAM for 64x64)
# ============================================================================

# Image Configuration - Scaled up CIFAR-10 for AlexNet-Small
ORIGINAL_SIZE = 32          # Original CIFAR-10 size
IMAGE_SIZE = 64             # Scaled size for AlexNet-Small
NUM_CHANNELS = 3
BYTES_PER_IMAGE = IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS  # 12288 bytes

# Protocol Configuration
START_MARKER = b'\xDE\xAD\xBE\xEF'  # DEADBEEF - Start of transmission marker


def load_cifar10_images(num_images=1000, use_test_set=True):
    """Load CIFAR-10 images from TensorFlow datasets"""
    print("Loading CIFAR-10 dataset from TensorFlow...")
    
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    
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


def scale_image_to_64x64(image, interpolation='bilinear'):
    """Scale a 32x32 image to 64x64 using PIL for high-quality interpolation"""
    interp_map = {
        'nearest': Image.NEAREST,
        'bilinear': Image.BILINEAR,
        'bicubic': Image.BICUBIC,
        'lanczos': Image.LANCZOS
    }
    
    interp_method = interp_map.get(interpolation.lower(), Image.BILINEAR)
    pil_image = Image.fromarray(image.astype(np.uint8))
    scaled_pil = pil_image.resize((IMAGE_SIZE, IMAGE_SIZE), interp_method)
    return np.array(scaled_pil)


def normalize_image(image, act_mode_8bit=True):
    """Normalize image using AI8X standard normalization"""
    img = image.astype(np.float32)
    img = img - 128.0
    img = np.clip(img, -128, 127)
    
    if not act_mode_8bit:
        img = img / 128.0
    
    return img


def preprocess_image_for_cnn(image, act_mode_8bit=None, use_chw=None, interpolation='bilinear'):
    """Preprocess image to match the CNN input format with proper normalization"""
    if act_mode_8bit is None:
        act_mode_8bit = ACT_MODE_8BIT
    if use_chw is None:
        use_chw = USE_CHW_FORMAT
    
    # Scale image from 32x32 to 64x64
    scaled_image = scale_image_to_64x64(image, interpolation)
    
    # Apply normalization
    normalized = normalize_image(scaled_image, act_mode_8bit)
    
    if act_mode_8bit:
        normalized = normalized.astype(np.int8)
        processed = normalized.view(np.uint8)
    else:
        quantized = (normalized * 128.0).astype(np.int8)
        processed = quantized.view(np.uint8)
    
    if use_chw:
        processed_chw = np.transpose(processed, (2, 0, 1))
        flattened = processed_chw.flatten(order='C')
    else:
        flattened = processed.flatten(order='C')
    
    return flattened.tobytes()


def send_batch_header(ser, num_images):
    """Send batch header with start marker and number of images."""
    header = bytearray()
    header.extend(START_MARKER)
    header.extend(struct.pack('>I', num_images))  # Big-endian to match device
    
    bytes_sent = ser.write(header)
    ser.flush()
    return bytes_sent == len(header)


def wait_for_ready(ser, timeout_seconds=10):
    """Wait for 'READY' signal from device before sending images."""
    start_time = time.time()
    response = ""
    
    while (time.time() - start_time) < timeout_seconds:
        if ser.in_waiting > 0:
            chunk = ser.read(ser.in_waiting)
            try:
                text = chunk.decode('utf-8', errors='ignore')
                response += text
                print(text, end='', flush=True)
                
                if 'READY' in response:
                    return True
            except:
                pass
        else:
            time.sleep(0.01)
    
    return False


def send_image_data(ser, image_data, label):
    """Send a single image (label + image data) without start marker."""
    packet = bytearray()
    packet.extend(struct.pack('<B', label))
    packet.extend(image_data)
    
    bytes_sent = ser.write(packet)
    ser.flush()
    return bytes_sent == len(packet)


def open_serial_port(port=SERIAL_PORT, baudrate=BAUD_RATE, timeout=TIMEOUT):
    """Open serial port with auto-detection fallback"""
    available_ports = []
    for port_info in serial.tools.list_ports.comports():
        available_ports.append(f"{port_info.device}: {port_info.description}")
    
    print("\nAvailable COM ports:")
    for port_info in available_ports:
        print(f"  - {port_info}")
    
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
            return None, None
    
    if ser is None:
        return None, None
    
    print(f"Serial port opened successfully")
    print(f"Port: {ser.name}")
    print(f"Baud rate: {ser.baudrate}")
    
    return ser, port_to_use


def read_device_output(ser, timeout_seconds=120):
    """Read and display all output from device until benchmark completes."""
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
            
            try:
                text = chunk.decode('utf-8', errors='ignore')
                print(text, end='', flush=True)
                
                if '[DONE]' in device_response.decode('utf-8', errors='ignore'):
                    benchmark_done = True
            except:
                pass
        else:
            time.sleep(0.01)
    
    print("\n" + "="*70)
    
    return benchmark_done


def send_cifar10_benchmark(num_images=None, use_test_set=None,
                           serial_port=None, baudrate=None, timeout=None,
                           act_mode_8bit=None, use_chw=None, interpolation='bilinear'):
    """Send CIFAR-10 images (64x64) as a batch for benchmarking."""
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
    
    if num_images > MAX_QUEUE_SIZE:
        print(f"Warning: Limiting images to {MAX_QUEUE_SIZE} (device queue limit)")
        num_images = MAX_QUEUE_SIZE
    
    print("="*70)
    print("CIFAR-10 Benchmark Sender (64x64 Queue-based Protocol)")
    print("="*70)
    print(f"Configuration:")
    print(f"  - Number of images: {num_images}")
    print(f"  - Dataset: {'Test' if use_test_set else 'Training'} set")
    print(f"  - Original size: {ORIGINAL_SIZE}x{ORIGINAL_SIZE}")
    print(f"  - Scaled size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"  - Interpolation: {interpolation}")
    print(f"  - Bytes per image: {BYTES_PER_IMAGE} bytes")
    print(f"  - Total data: {(BYTES_PER_IMAGE * num_images) / 1024:.2f} KB")
    print(f"  - Serial port: {serial_port}")
    print(f"  - Baud rate: {baudrate}")
    print(f"  - Normalization: {'8-bit [-128, 127]' if act_mode_8bit else 'Float [-1, 1]'}")
    print(f"  - Data format: {'CHW' if use_chw else 'HWC'}")
    
    # Load images
    images, labels = load_cifar10_images(num_images, use_test_set)
    
    # Preprocess all images first
    print(f"\nPreprocessing {num_images} images (scaling to 64x64)...")
    preprocessed_images = []
    for idx in tqdm(range(num_images), desc="Preprocessing"):
        image_data = preprocess_image_for_cnn(images[idx], act_mode_8bit, use_chw, interpolation)
        preprocessed_images.append((image_data, labels[idx]))
    
    # Main loop
    ser = None
    run_count = 0
    
    print("\n" + "="*70)
    print("Starting continuous benchmark mode (Press Ctrl+C to stop)")
    print("="*70)
    
    try:
        while True:
            if ser is None or not ser.is_open:
                ser, port_name = open_serial_port(serial_port, baudrate, timeout)
                if ser is None:
                    print("\nRetrying in 3 seconds...")
                    time.sleep(3)
                    continue
                
                print("\nWaiting for device to be ready...")
                time.sleep(2)
            
            run_count += 1
            print(f"\n{'='*70}")
            print(f"BENCHMARK RUN #{run_count}")
            print(f"{'='*70}")
            
            ser.reset_input_buffer()
            ser.reset_output_buffer()
            
            print(f"\nSending batch header (N={num_images})...")
            if not send_batch_header(ser, num_images):
                print("Failed to send batch header! Reconnecting...")
                ser.close()
                ser = None
                time.sleep(1)
                continue
            
            print("Waiting for device READY signal...")
            if not wait_for_ready(ser, timeout_seconds=10):
                print("Device did not respond with READY! Reconnecting...")
                ser.close()
                ser = None
                time.sleep(1)
                continue
            
            print("Device ready, sending images...")
            
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
            
            print("\nWaiting for device to complete benchmark...")
            benchmark_complete = read_device_output(ser, timeout_seconds=120)
            
            if benchmark_complete:
                print(f"\n✓ Benchmark run #{run_count} completed successfully!")
            else:
                print(f"\n⚠ Benchmark run #{run_count} may not have completed (timeout)")
            
            print("\nWaiting 10 seconds before next run... (Ctrl+C to stop)")
            time.sleep(10)
            
    except KeyboardInterrupt:
        print("\n\n" + "="*70)
        print("Keyboard interrupt received. Stopping...")
        print(f"Total benchmark runs completed: {run_count}")
        print("="*70)
    
    finally:
        if ser is not None and ser.is_open:
            ser.close()
            print("Serial port closed")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Send CIFAR-10 images (64x64) to MAX78000 via UART (Benchmark Mode)')
    parser.add_argument('-n', '--num-images', type=int, default=None,
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
    parser.add_argument('--interpolation', type=str, default='bilinear',
                        choices=['nearest', 'bilinear', 'bicubic', 'lanczos'],
                        help='Interpolation method for scaling (default: bilinear)')
    
    args = parser.parse_args()
    
    print("="*70)
    print("CIFAR-10 (64x64) Benchmark Tool for MAX78000 AlexNet-Small")
    print("="*70)
    
    use_chw = not args.hwc
    act_mode_8bit = not args.float_mode
    
    send_cifar10_benchmark(
        num_images=args.num_images,
        use_test_set=not args.train_set,
        serial_port=args.port,
        baudrate=args.baudrate,
        timeout=args.timeout,
        act_mode_8bit=act_mode_8bit,
        use_chw=use_chw,
        interpolation=args.interpolation
    )


if __name__ == "__main__":
    main()
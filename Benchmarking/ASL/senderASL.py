"""
ASL Alphabet Image Sender via UART (64x64 Version) - Queue-based Benchmark
Sends images from ASL Alphabet dataset to MAX78000 device via UART
Images are resized to 64x64 and sent in CHW format
Protocol: [DEADBEEF][NUM_IMAGES(4)][Label0][Image0][Label1][Image1]...
"""
import serial
import serial.tools.list_ports
import numpy as np
import os
import time
import struct
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import argparse
import random

# ============================================================================
# CONFIGURATION SETTINGS
# ============================================================================
SERIAL_PORT = 'COM6'        # Default COM port, will auto-detect if this fails
BAUD_RATE = 115200          # Baud rate for serial communication
NUM_IMAGES = 1000           # Total number of images for full benchmark
BATCH_SIZE = 5              # Images per batch (must match device MAX_QUEUE_SIZE for 64x64)
NUM_BATCHES = 200           # Number of batches (1000 / 5 = 200)
USE_TEST_SET = False        # True = test set, False = training set
TIMEOUT = 30                # Serial timeout in seconds (longer for batch processing)
ACT_MODE_8BIT = True        # True = [-128, 127], False = [-128/128, 127/128]
USE_CHW_FORMAT = True       # True = CHW (Channels,Height,Width), False = HWC (Height,Width,Channels)
DATASET_PATH = "D:/ASL_Alphabet_Dataset"  # Path to ASL dataset
# ============================================================================

# Image Configuration
IMAGE_SIZE = 64             # Target size for CNN input
NUM_CHANNELS = 3
BYTES_PER_IMAGE = IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS  # 12288 bytes

# Protocol Configuration
START_MARKER = b'\xDE\xAD\xBE\xEF'  # DEADBEEF - Start of transmission marker

# ASL Class Labels (A-Z = 0-25, del=26)
# Note: This dataset only has A-Z and del (27 classes total)
ASL_CLASSES = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z', 'unknown'
]

CLASS_TO_LABEL = {cls: idx for idx, cls in enumerate(ASL_CLASSES)}


def is_valid_image(image_path):
    """Check if an image file is valid and can be opened"""
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception:
        return False


def load_asl_images(dataset_path=DATASET_PATH, num_images=1000, use_test_set=True):
    """Load ASL Alphabet images from the dataset folder"""
    print("Loading ASL Alphabet dataset...")
    
    images = []
    labels = []
    image_paths = []
    skipped_count = 0
    
    if use_test_set:
        subset_path = os.path.join(dataset_path, "Test")
        print(f"Using test set from: {subset_path}")
        
        if not os.path.exists(subset_path):
            print(f"Error: Test set path does not exist: {subset_path}")
            return [], [], []
        
        # Test set has files named like "A_test.jpg", "B_test.jpg", etc.
        test_files = []
        for filename in os.listdir(subset_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Extract class name from filename (e.g., "A_test.jpg" -> "A")
                class_name = filename.split('_')[0].upper()
                if class_name in CLASS_TO_LABEL:
                    test_files.append((os.path.join(subset_path, filename), CLASS_TO_LABEL[class_name]))
        
        random.shuffle(test_files)
        
        # Keep loading until we have enough valid images
        idx = 0
        while len(images) < num_images and idx < len(test_files):
            img_path, label = test_files[idx]
            idx += 1
            
            if not is_valid_image(img_path):
                skipped_count += 1
                continue
            try:
                img = Image.open(img_path).convert('RGB')
                images.append(np.array(img))
                labels.append(label)
                image_paths.append(img_path)
            except Exception as e:
                skipped_count += 1
    else:
        subset_path = os.path.join(dataset_path, "Train")
        print(f"Using training set from: {subset_path}")
        
        if not os.path.exists(subset_path):
            print(f"Error: Training set path does not exist: {subset_path}")
            return [], [], []
        
        # Training set has subfolders per class (A, B, C, ..., Z, unknown)
        all_images = []
        for class_name in ASL_CLASSES:
            class_path = os.path.join(subset_path, class_name)
            if os.path.exists(class_path):
                for filename in os.listdir(class_path):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        all_images.append((os.path.join(class_path, filename), CLASS_TO_LABEL[class_name]))
        
        random.shuffle(all_images)
        
        # Keep loading until we have enough valid images
        idx = 0
        while len(images) < num_images and idx < len(all_images):
            img_path, label = all_images[idx]
            idx += 1
            
            if not is_valid_image(img_path):
                skipped_count += 1
                continue
            try:
                img = Image.open(img_path).convert('RGB')
                images.append(np.array(img))
                labels.append(label)
                image_paths.append(img_path)
            except Exception as e:
                skipped_count += 1
    
    labels = np.array(labels)
    
    print(f"Loaded {len(images)} valid images")
    if skipped_count > 0:
        print(f"Skipped {skipped_count} invalid images")
    
    return images, labels, image_paths


def resize_image_to_64x64(image, interpolation='bilinear'):
    """Resize an image to 64x64 using PIL for high-quality interpolation"""
    interp_map = {
        'nearest': Image.NEAREST,
        'bilinear': Image.BILINEAR,
        'bicubic': Image.BICUBIC,
        'lanczos': Image.LANCZOS
    }
    
    interp_method = interp_map.get(interpolation.lower(), Image.BILINEAR)
    pil_image = Image.fromarray(image.astype(np.uint8))
    resized_pil = pil_image.resize((IMAGE_SIZE, IMAGE_SIZE), interp_method)
    return np.array(resized_pil)


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
    
    resized_image = resize_image_to_64x64(image, interpolation)
    normalized = normalize_image(resized_image, act_mode_8bit)
    
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
    header.extend(struct.pack('>I', num_images))
    
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


def read_device_output(ser, timeout_seconds=120, batch_num=None):
    """
    Read and display all output from device until batch completes.
    Looks for [BATCH X/200 COMPLETE] marker for individual batches,
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
            
            try:
                text = chunk.decode('utf-8', errors='ignore')
                print(text, end='', flush=True)
                
                # Check for batch completion marker
                response_str = device_response.decode('utf-8', errors='ignore')
                if batch_num is not None:
                    # Looking for specific batch completion
                    if f'[BATCH {batch_num}/200 COMPLETE]' in response_str:
                        batch_done = True
                else:
                    # Looking for final completion
                    if '[DONE]' in response_str or 'BENCHMARK COMPLETE' in response_str:
                        batch_done = True
            except:
                pass
        else:
            time.sleep(0.01)
    
    print("\n" + "="*70)
    
    return batch_done


def send_asl_benchmark_multibatch(dataset_path=None, num_images=None, batch_size=None, use_test_set=None,
                                  serial_port=None, baudrate=None, timeout=None,
                                  act_mode_8bit=None, use_chw=None, interpolation='bilinear'):
    """
    Send ASL images (64x64) in multiple batches for benchmarking 1000 images.
    Sends images in batches of BATCH_SIZE, device processes each batch, then requests next.
    Protocol: For each batch: [DEADBEEF][NUM_IMAGES(4)][Label0][Image0][Label1][Image1]...
    
    Args:
        dataset_path: path to ASL dataset
        num_images: total number of images to send (default: 1000)
        batch_size: number of images per batch (default: 5 for 64x64)
        use_test_set: if True, use test set; otherwise use training set
        serial_port: COM port to use
        baudrate: baud rate for serial communication
        timeout: timeout in seconds
        act_mode_8bit: if True, use 8-bit mode; if False, use float mode
        use_chw: if True, convert to CHW format; if False, keep HWC format
        interpolation: resizing interpolation method
    """
    # Use global defaults if not specified
    if dataset_path is None:
        dataset_path = DATASET_PATH
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
    print("ASL Alphabet Multi-Batch Benchmark Sender (64x64)")
    print("="*70)
    print(f"Configuration:")
    print(f"  - Dataset path:      {dataset_path}")
    print(f"  - Total images:      {num_images}")
    print(f"  - Batch size:        {batch_size} images")
    print(f"  - Number of batches: {num_batches}")
    print(f"  - Dataset:           {'Test' if use_test_set else 'Training'} set")
    print(f"  - Image size:        {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"  - Interpolation:     {interpolation}")
    print(f"  - Bytes per image:   {BYTES_PER_IMAGE} bytes")
    print(f"  - Total data:        {(BYTES_PER_IMAGE * num_images) / 1024:.2f} KB")
    print(f"  - Serial port:       {serial_port}")
    print(f"  - Baud rate:         {baudrate}")
    print(f"  - Normalization:     {'8-bit [-128, 127]' if act_mode_8bit else 'Float [-1, 1]'}")
    print(f"  - Data format:       {'CHW' if use_chw else 'HWC'}")
    
    # Load all images
    print(f"\nLoading {num_images} images from dataset...")
    images, labels, image_paths = load_asl_images(dataset_path, num_images, use_test_set)
    
    if len(images) == 0:
        print("No images loaded! Check dataset path.")
        return False
    
    # Preprocess all images first (do this before opening port)
    print(f"Preprocessing {len(images)} images...")
    preprocessed_images = []
    for idx in tqdm(range(len(images)), desc="Preprocessing"):
        image_data = preprocess_image_for_cnn(images[idx], act_mode_8bit, use_chw, interpolation)
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
            end_idx = min(start_idx + batch_size, len(preprocessed_images))
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
                    print(chunk.decode('utf-8', errors='ignore'), end='', flush=True)
                except:
                    pass
            else:
                time.sleep(0.1)
        
        overall_time = time.time() - overall_start_time
        
        print(f"\n{'='*70}")
        print("BENCHMARK COMPLETE")
        print(f"{'='*70}")
        print(f"Total images:        {len(preprocessed_images)}")
        print(f"Total batches:       {num_batches}")
        print(f"Total TX time:       {total_tx_time:.2f}s")
        print(f"Overall time:        {overall_time:.2f}s ({overall_time/60:.1f} min)")
        print(f"Avg TX rate:         {(BYTES_PER_IMAGE * len(preprocessed_images)) / total_tx_time / 1024:.1f} KB/s")
        print(f"{'='*70}\n")
        
        return True
        
    except KeyboardInterrupt:
        print("\n\n⚠ Benchmark interrupted by user")
        return False
    except Exception as e:
        print(f"\n⚠ Error during benchmark: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if ser and ser.is_open:
            ser.close()
            print(f"Serial port {port_name} closed.")


def get_class_name(label):
    """Get class name from label index"""
    if 0 <= label < len(ASL_CLASSES):
        return ASL_CLASSES[label]
    return f"Unknown({label})"


def send_asl_benchmark(dataset_path=None, num_images=None, use_test_set=None,
                       serial_port=None, baudrate=None, timeout=None,
                       act_mode_8bit=None, use_chw=None, interpolation='bilinear'):
    """Send ASL images as a batch for benchmarking."""
    if dataset_path is None:
        dataset_path = DATASET_PATH
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
    print("ASL Alphabet Benchmark Sender (64x64 Queue-based Protocol)")
    print("="*70)
    print(f"Configuration:")
    print(f"  - Dataset path: {dataset_path}")
    print(f"  - Number of images: {num_images}")
    print(f"  - Dataset: {'Test' if use_test_set else 'Training'} set")
    print(f"  - Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"  - Interpolation: {interpolation}")
    print(f"  - Bytes per image: {BYTES_PER_IMAGE} bytes")
    print(f"  - Total data: {(BYTES_PER_IMAGE * num_images) / 1024:.2f} KB")
    print(f"  - Serial port: {serial_port}")
    print(f"  - Baud rate: {baudrate}")
    print(f"  - Normalization: {'8-bit [-128, 127]' if act_mode_8bit else 'Float [-1, 1]'}")
    print(f"  - Data format: {'CHW' if use_chw else 'HWC'}")
    
    # Load images
    images, labels, image_paths = load_asl_images(dataset_path, num_images, use_test_set)
    
    if len(images) == 0:
        print("No images loaded! Check dataset path.")
        return False
    
    # Preprocess all images first
    print(f"\nPreprocessing {len(images)} images...")
    preprocessed_images = []
    for idx in tqdm(range(len(images)), desc="Preprocessing"):
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
            
            actual_num_images = len(preprocessed_images)
            
            print(f"\nSending batch header (N={actual_num_images})...")
            if not send_batch_header(ser, actual_num_images):
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
            
            print(f"Sending {actual_num_images} images...")
            tx_start_time = time.time()
            
            send_failed = False
            for idx, (image_data, label) in enumerate(tqdm(preprocessed_images[:actual_num_images], desc="Transmitting")):
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
            
            print(f"\n✓ All {actual_num_images} images sent in {tx_time:.2f}s")
            print(f"  Transmission rate: {(BYTES_PER_IMAGE * actual_num_images) / tx_time / 1024:.2f} KB/s")
            
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
    parser = argparse.ArgumentParser(description='Send ASL images to MAX78000 via UART (Multi-Batch Benchmark Mode)')
    parser.add_argument('-n', '--num-images', type=int, default=None,
                        help=f'Total number of images to send (default: {NUM_IMAGES})')
    parser.add_argument('-b', '--batch-size', type=int, default=None,
                        help=f'Number of images per batch (default: {BATCH_SIZE})')
    parser.add_argument('--train-set', action='store_true',
                        help='Use training set instead of test set')
    parser.add_argument('--dataset', type=str, default=None,
                        help=f'Path to ASL dataset (default: {DATASET_PATH})')
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
                        help='Interpolation method for resizing (default: bilinear)')
    
    args = parser.parse_args()
    
    print("="*70)
    print("ASL Alphabet Multi-Batch Benchmark Tool for MAX78000")
    print("="*70)
    
    use_chw = not args.hwc
    act_mode_8bit = not args.float_mode
    
    success = send_asl_benchmark_multibatch(
        dataset_path=args.dataset,
        num_images=args.num_images,
        batch_size=args.batch_size,
        use_test_set=True if args.train_set else None,  # None = use default (False), --train-set = explicitly False
        serial_port=args.port,
        baudrate=args.baudrate,
        timeout=args.timeout,
        act_mode_8bit=act_mode_8bit,
        use_chw=use_chw,
        interpolation=args.interpolation
    )
    
    if success:
        print("\n✓ Multi-batch benchmark completed successfully")
    else:
        print("\n✗ Multi-batch benchmark failed")


if __name__ == "__main__":
    main()
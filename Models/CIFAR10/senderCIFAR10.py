"""
CIFAR-10 Image Sender via UART
Sends images from CIFAR-10 dataset to MAX78000 device via UART COM5
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
NUM_IMAGES = 100            # Number of images to send
USE_TEST_SET = True         # True = test set, False = training set
TIMEOUT = 5                 # Serial timeout in seconds
# ============================================================================

# Image Configuration (CIFAR-10 standard)
IMAGE_SIZE = 32
NUM_CHANNELS = 3
BYTES_PER_IMAGE = IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS  # 3072 bytes

# Protocol Configuration
START_MARKER = b'\xFF\xAA'  # Start of transmission marker
END_MARKER = b'\xBB\xFF'    # End of transmission marker


def load_cifar10_images(num_images=1000, use_test_set=True):
    """
    Load CIFAR-10 images from TensorFlow datasets
    
    Args:
        num_images: number of images to load
        use_test_set: if True, use test set; otherwise use training set
    
    Returns:
        images: numpy array of shape (num_images, 32, 32, 3)
        labels: numpy array of shape (num_images,)
    """
    print("Loading CIFAR-10 dataset from TensorFlow...")
    
    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    
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


def preprocess_image_for_cnn(image):
    """
    Preprocess image to match the CNN input format (HWC format with proper scaling)
    
    Args:
        image: numpy array of shape (32, 32, 3) with values 0-255
        
    Returns:
        processed_data: byte array ready for transmission
    """
    # Normalize to 0-255 range (already in this range for CIFAR-100)
    # Convert to uint8 if not already
    image = image.astype(np.uint8)
    
    # CNN expects HWC format (Height, Width, Channel)
    # CIFAR-10 is already in this format
    
    # Flatten the image to 1D array for transmission
    flattened = image.flatten()
    
    return flattened.tobytes()


def send_image_via_uart(ser, image_data, image_idx, label):
    """
    Send a single image via UART with protocol framing
    
    Args:
        ser: serial port object
        image_data: preprocessed image bytes
        image_idx: index of the image
        label: ground truth label
        
    Returns:
        success: boolean indicating if transmission was successful
    """
    try:
        # Create packet: [START_MARKER][IMAGE_IDX][LABEL][IMAGE_DATA][END_MARKER]
        packet = bytearray()
        packet.extend(START_MARKER)
        packet.extend(struct.pack('<I', image_idx))  # 4 bytes for image index
        packet.extend(struct.pack('<B', label))       # 1 byte for label
        packet.extend(image_data)                     # 3072 bytes for image
        packet.extend(END_MARKER)
        
        # Send packet
        bytes_sent = ser.write(packet)
        
        # Try to read response from device (non-blocking)
        time.sleep(0.05)  # Give device time to process
        if ser.in_waiting > 0:
            response = ser.read(ser.in_waiting)
            try:
                print(f"\n[Device]: {response.decode('utf-8', errors='ignore')}", end='')
            except:
                pass
        
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


def send_cifar10_images(num_images=NUM_IMAGES, use_test_set=USE_TEST_SET,
                        serial_port=SERIAL_PORT, baudrate=BAUD_RATE, timeout=TIMEOUT):
    """
    Main function to send CIFAR-10 images via UART
    
    Args:
        num_images: number of images to send
        use_test_set: if True, use test set; otherwise use training set
        serial_port: COM port to use
        baudrate: baud rate for serial communication
        timeout: timeout in seconds
    """
    print("="*60)
    print("CIFAR-10 to MAX78000 UART Sender")
    print("="*60)
    print(f"Configuration:")
    print(f"  - Number of images: {num_images}")
    print(f"  - Dataset: {'Test' if use_test_set else 'Training'} set")
    print(f"  - Serial port: {serial_port}")
    print(f"  - Baud rate: {baudrate}")
    
    # Load images
    images, labels = load_cifar10_images(num_images, use_test_set)
    
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
    print(f"\nSending {num_images} images...")
    success_count = 0
    fail_count = 0
    
    start_time = time.time()
    
    for idx in tqdm(range(num_images), desc="Sending images"):
        # Preprocess image
        image_data = preprocess_image_for_cnn(images[idx])
        
        # Send image
        if send_image_via_uart(ser, image_data, idx, labels[idx]):
            success_count += 1
        else:
            fail_count += 1
            print(f"\nFailed to send image {idx}")
        
        # Small delay between images to prevent buffer overflow
        time.sleep(0.01)  # 10ms delay
    
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
    parser = argparse.ArgumentParser(description='Send CIFAR-10 images to MAX78000 via UART')
    parser.add_argument('--num-images', type=int, default=NUM_IMAGES,
                        help=f'Number of images to send (default: {NUM_IMAGES})')
    parser.add_argument('--train-set', action='store_true',
                        help='Use training set instead of test set')
    parser.add_argument('--port', type=str, default=SERIAL_PORT,
                        help=f'Serial port to use (default: {SERIAL_PORT})')
    parser.add_argument('--baudrate', type=int, default=BAUD_RATE,
                        help=f'Baud rate (default: {BAUD_RATE})')
    parser.add_argument('--timeout', type=int, default=TIMEOUT,
                        help=f'Serial timeout in seconds (default: {TIMEOUT})')
    
    args = parser.parse_args()
    
    print("Required packages: pyserial, tensorflow, numpy, tqdm")
    print("Install with: pip install pyserial tensorflow numpy tqdm\n")
    
    success = send_cifar10_images(
        num_images=args.num_images,
        use_test_set=not args.train_set,
        serial_port=args.port,
        baudrate=args.baudrate,
        timeout=args.timeout
    )
    
    if success:
        print("\n✓ Program completed successfully")
    else:
        print("\n✗ Program failed")


if __name__ == "__main__":
    main()
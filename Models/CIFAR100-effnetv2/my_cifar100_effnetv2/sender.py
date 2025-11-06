"""
CIFAR-100 Image Sender via UART
Sends images from CIFAR-100 dataset to MAX78000 device via UART COM5
"""

import serial
import numpy as np
import tensorflow as tf
import time
import struct
from tqdm import tqdm

# UART Configuration
SERIAL_PORT = 'COM5'
BAUD_RATE = 115200
TIMEOUT = 5  # seconds

# Image Configuration
IMAGE_SIZE = 32
NUM_CHANNELS = 3
BYTES_PER_IMAGE = IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS  # 3072 bytes
NUM_IMAGES = 1000

# Protocol Configuration
START_MARKER = b'\xFF\xAA'  # Start of transmission marker
END_MARKER = b'\xBB\xFF'    # End of transmission marker


def load_cifar100_images(num_images=1000):
    """
    Load CIFAR-100 images from TensorFlow datasets
    
    Returns:
        images: numpy array of shape (num_images, 32, 32, 3)
        labels: numpy array of shape (num_images,)
    """
    print("Loading CIFAR-100 dataset from TensorFlow...")
    
    # Load CIFAR-100 dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
    
    # Use test set images (or you can use x_train if preferred)
    images = x_test[:num_images]
    labels = y_test[:num_images].flatten()
    
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
    # CIFAR-100 is already in this format
    
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
        
        # Wait for acknowledgment from device (optional)
        # You can add ACK/NACK protocol here if needed
        
        return bytes_sent == len(packet)
        
    except serial.SerialException as e:
        print(f"Serial error: {e}")
        return False


def send_cifar100_images():
    """
    Main function to send CIFAR-100 images via UART
    """
    print("="*60)
    print("CIFAR-100 to MAX78000 UART Sender")
    print("="*60)
    
    # Load images
    images, labels = load_cifar100_images(NUM_IMAGES)
    
    # Open serial port
    try:
        print(f"\nOpening serial port {SERIAL_PORT} at {BAUD_RATE} baud...")
        ser = serial.Serial(
            port=SERIAL_PORT,
            baudrate=BAUD_RATE,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=TIMEOUT,
            write_timeout=TIMEOUT
        )
        
        print(f"Serial port opened successfully")
        print(f"Port: {ser.name}")
        print(f"Baud rate: {ser.baudrate}")
        
        # Wait for device to be ready
        time.sleep(2)
        
        # Clear any existing data in buffers
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        
        # Send images
        print(f"\nSending {NUM_IMAGES} images...")
        success_count = 0
        fail_count = 0
        
        start_time = time.time()
        
        for idx in tqdm(range(NUM_IMAGES), desc="Sending images"):
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
        print(f"Total images sent: {success_count}/{NUM_IMAGES}")
        print(f"Failed transmissions: {fail_count}")
        print(f"Total time: {elapsed_time:.2f} seconds")
        print(f"Average time per image: {elapsed_time/NUM_IMAGES*1000:.2f} ms")
        print(f"Throughput: {NUM_IMAGES/elapsed_time:.2f} images/second")
        print(f"Data rate: {(BYTES_PER_IMAGE*NUM_IMAGES)/(elapsed_time*1024):.2f} KB/s")
        
        # Close serial port
        ser.close()
        print("\nSerial port closed")
        
    except serial.SerialException as e:
        print(f"\nSerial port error: {e}")
        print("Please check:")
        print("1. Device is connected to COM5")
        print("2. No other program is using COM5")
        print("3. Correct drivers are installed")
        return False
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        return False
    
    return True


if __name__ == "__main__":
    # Install required packages if not already installed
    print("Required packages: pyserial, tensorflow, numpy, tqdm")
    print("Install with: pip install pyserial tensorflow numpy tqdm\n")
    
    success = send_cifar100_images()
    
    if success:
        print("\n✓ Program completed successfully")
    else:
        print("\n✗ Program failed")
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

# UART Configuration
SERIAL_PORT = 'COM6'  # Default, will auto-detect if this fails
BAUD_RATE = 115200
TIMEOUT = 5  # seconds

# Image Configuration
IMAGE_SIZE = 32
NUM_CHANNELS = 3
BYTES_PER_IMAGE = IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS  # 3072 bytes
NUM_IMAGES = 100  # Start with 100 images to test quickly

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


def send_cifar100_images():
    """
    Main function to send CIFAR-100 images via UART
    """
    print("="*60)
    print("CIFAR-100 to MAX78000 UART Sender")
    print("="*60)
    
    # Load images
    images, labels = load_cifar100_images(NUM_IMAGES)
    
    # Auto-detect COM port if default fails
    available_ports = []
    for port in serial.tools.list_ports.comports():
        available_ports.append(f"{port.device}: {port.description}")
    
    print("\nAvailable COM ports:")
    for port_info in available_ports:
        print(f"  - {port_info}")
    
    # Try to open serial port
    port_to_use = SERIAL_PORT
    ser = None
    
    try:
        print(f"\nTrying to open serial port {port_to_use} at {BAUD_RATE} baud...")
        ser = serial.Serial(
            port=port_to_use,
            baudrate=BAUD_RATE,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=TIMEOUT,
            write_timeout=TIMEOUT
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
                        baudrate=BAUD_RATE,
                        bytesize=serial.EIGHTBITS,
                        parity=serial.PARITY_NONE,
                        stopbits=serial.STOPBITS_ONE,
                        timeout=TIMEOUT,
                        write_timeout=TIMEOUT
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
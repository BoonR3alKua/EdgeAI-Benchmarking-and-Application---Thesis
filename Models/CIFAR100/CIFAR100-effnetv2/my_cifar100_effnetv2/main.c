/*******************************************************************************
* Modified main.c to receive CIFAR-100 images via UART and perform inference
* 
* IMPORTANT NOTES:
* 1. Verify CNN input memory address (0x51018000) matches your CNN configuration
* 2. softmax_q17p14_q15 function must be available (check softmax.c/h)
* 3. Ensure UART0 is properly connected to sender on COM5
* 4. Baud rate: 115200
*******************************************************************************/

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include "mxc.h"
#include "cnn.h"
#include "uart.h"
#include "dma.h"
#include "led.h"

// Softmax function prototype (should be in softmax.h if it exists)
extern void softmax_q17p14_q15(const q31_t *vec_in, const uint16_t dim_vec, q15_t *p_out);

// UART Configuration
#define UART_BAUD 115200
#define RX_UART MXC_UART0  // Console UART (COM5) - used for both data and console

// Image Configuration
#define IMAGE_SIZE 32
#define NUM_CHANNELS 3
#define BYTES_PER_IMAGE (IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS)  // 3072 bytes

// Protocol markers
#define START_MARKER_1 0xFF
#define START_MARKER_2 0xAA
#define END_MARKER_1 0xBB
#define END_MARKER_2 0xFF

// Buffers
static uint8_t rx_buffer[BYTES_PER_IMAGE + 16];  // Extra space for protocol overhead
static uint32_t input_data[BYTES_PER_IMAGE / 4];  // 768 words for CNN input

// CNN outputs
static int32_t ml_data[CNN_NUM_OUTPUTS];
static q15_t ml_softmax[CNN_NUM_OUTPUTS];

volatile uint32_t cnn_time;

/**
 * Receive image via UART with protocol framing
 */
int receive_image_uart(uint32_t *image_idx, uint8_t *label)
{
    int bytes_received = 0;
    int byte_read;
    uint8_t byte;
    int state = 0;  // 0: waiting for start, 1: reading data, 2: verifying end
    
    printf("Waiting for image...\n");
    
    // State machine to receive packet
    while (1) {
        // Read one byte
        byte_read = MXC_UART_ReadCharacter(RX_UART);
        if (byte_read < 0) {
            continue;  // No data available or error
        }
        byte = (uint8_t)byte_read;
        
        switch (state) {
            case 0:  // Waiting for start marker
                if (byte == START_MARKER_1) {
                    state = 1;
                }
                break;
                
            case 1:  // Got first start marker, check second
                if (byte == START_MARKER_2) {
                    bytes_received = 0;
                    state = 2;
                } else {
                    state = 0;
                }
                break;
                
            case 2:  // Reading packet data
                // Check for buffer overflow before writing
                if (bytes_received >= sizeof(rx_buffer)) {
                    printf("Error: Buffer overflow, resetting\n");
                    state = 0;
                    bytes_received = 0;
                    break;
                }
                
                rx_buffer[bytes_received++] = byte;
                
                // Check if we've received all data (4 bytes idx + 1 byte label + 3072 bytes image + 2 bytes end marker)
                if (bytes_received >= (4 + 1 + BYTES_PER_IMAGE + 2)) {
                    // Verify end markers
                    if (rx_buffer[bytes_received - 2] == END_MARKER_1 &&
                        rx_buffer[bytes_received - 1] == END_MARKER_2) {
                        
                        // Extract image index
                        *image_idx = *((uint32_t *)&rx_buffer[0]);
                        
                        // Extract label
                        *label = rx_buffer[4];
                        
                        // Copy image data
                        memcpy(input_data, &rx_buffer[5], BYTES_PER_IMAGE);
                        
                        printf("Received image %lu, label %d\n", *image_idx, *label);
                        return E_NO_ERROR;
                    } else {
                        printf("Error: Invalid end markers\n");
                        return E_FAIL;
                    }
                }
                break;
        }
    }
    
    return E_FAIL;
}

/**
 * Load input data from received buffer into CNN
 * CIFAR-100: 3 channels, 32x32 HWC format
 * Total: 3072 bytes (32x32x3)
 * 
 * For HWC format with 3 channels, data needs to be loaded to CNN input memory.
 * Based on MAX78000 architecture, input is typically at 0x51018000 for streaming,
 * but may need distribution across memory banks depending on CNN configuration.
 * 
 * TODO: Verify this matches the CNN configuration in cnn.c
 */
void load_input_from_buffer(void)
{
    // For CIFAR-100 with 3 input channels (RGB), 32x32 image
    // Input format is HWC (Height, Width, Channel)
    // Based on typical MAX78000 configuration for small RGB images:
    
    // Option 1: Single memory write - NOT WORKING! (0.80% accuracy)
    memcpy32((uint32_t *)0x50400000, input_data, 1024);  // Load all 3072 bytes as 1024 words
    
    // Option 2: Multi-bank distribution (TRYING THIS NOW)
    // Split the data across memory banks similar to KWS20
    // int words_per_bank = (BYTES_PER_IMAGE / 4) / 4;  // 768 / 4 = 192 words per bank
    // memcpy32((uint32_t *)0x50400000, &input_data[0 * words_per_bank], words_per_bank);
    // memcpy32((uint32_t *)0x50408000, &input_data[1 * words_per_bank], words_per_bank);
    // memcpy32((uint32_t *)0x50410000, &input_data[2 * words_per_bank], words_per_bank);
    // memcpy32((uint32_t *)0x50418000, &input_data[3 * words_per_bank], words_per_bank);
}

/**
 * Perform CNN inference and return classification
 */
int classify_image(void)
{
    int i, max_idx = 0;
    int32_t max_val = ml_data[0];
    
    cnn_time = 0;  // Reset timer before starting CNN
    cnn_start();
    
    while (cnn_time == 0)
        MXC_LP_EnterSleepMode();
    
    // Unload and compute softmax
    cnn_unload((uint32_t *)ml_data);
    softmax_q17p14_q15((const q31_t *)ml_data, CNN_NUM_OUTPUTS, ml_softmax);
    
    // Find maximum
    for (i = 1; i < CNN_NUM_OUTPUTS; i++) {
        if (ml_data[i] > max_val) {
            max_val = ml_data[i];
            max_idx = i;
        }
    }
    
    return max_idx;
}

/**
 * Get confidence percentage for a given class
 */
float get_confidence(int class_idx)
{
    // Convert q15 softmax value to percentage
    // q15 format: value / 32768 gives the probability
    // Multiply by 100 to get percentage
    int digs = (1000 * ml_softmax[class_idx] + 0x4000) >> 15;
    return digs / 10.0f;
}

int main(void)
{
    int error;
    uint32_t image_idx;
    uint8_t ground_truth_label;
    int predicted_class;
    int correct_count = 0;
    int total_count = 0;
    float total_confidence = 0.0f;
    float avg_confidence = 0.0f;
    
    MXC_ICC_Enable(MXC_ICC0);
    MXC_SYS_Clock_Select(MXC_SYS_CLOCK_IPO);
    SystemCoreClockUpdate();
    
    printf("\n*** CIFAR-100 CNN Inference via UART (EfficientNetV2) ***\n");
    printf("Ready to receive images on UART0 at %d baud\n", UART_BAUD);
    printf("NOTE: Close this console before running sender.py\n");
    printf("      Results will be visible after sender completes.\n\n");
    
    // Initialize UART for data reception
    error = MXC_UART_Init(RX_UART, UART_BAUD, MXC_UART_APB_CLK);
    if (error < E_NO_ERROR) {
        printf("Error initializing UART: %d\n", error);
        return error;
    }
    
    // Initialize CNN
    cnn_enable(MXC_S_GCR_PCLKDIV_CNNCLKSEL_PCLK, MXC_S_GCR_PCLKDIV_CNNCLKDIV_DIV1);
    cnn_init();
    cnn_load_weights();
    cnn_load_bias();
    cnn_configure();
    
    printf("CNN initialized. Ready to receive images.\n\n");
    
    LED_On(0);  // Indicate ready
    
    // Main loop: receive and classify images
    while (1) {
        // Receive image via UART
        if (receive_image_uart(&image_idx, &ground_truth_label) != E_NO_ERROR) {
            printf("Error receiving image\n");
            continue;
        }
        
        // Load image into CNN
        load_input_from_buffer();
        
        // Perform inference
        LED_On(1);  // Indicate processing
        predicted_class = classify_image();
        LED_Off(1);
        
        // Get confidence for predicted class
        float confidence = get_confidence(predicted_class);
        
        // Check accuracy
        total_count++;
        total_confidence += confidence;
        avg_confidence = total_confidence / total_count;
        
        if (predicted_class == ground_truth_label) {
            correct_count++;
        }
        
        // Print results with confidence
        printf("Image %lu: Predicted=%d (%.2f%%), Actual=%d, %s\n",
               image_idx, predicted_class, confidence, ground_truth_label,
               (predicted_class == ground_truth_label) ? "CORRECT" : "WRONG");
        
        printf("Accuracy: %d/%d (%.2f%%), Avg Confidence: %.2f%%\n\n",
               correct_count, total_count,
               (double)correct_count / total_count * 100.0,
               avg_confidence);
        
        // Reconfigure CNN for next inference
        cnn_init();
        cnn_configure();
    }
    
    return E_NO_ERROR;
}

/*
  SUMMARY OF OPS
  Hardware: 9,578,752 ops (9,486,848 macc; 89,856 comp; 2,048 add; 0 mul; 0 bitwise)
    Layer 0: 232,448 ops (221,184 macc; 11,264 comp; 0 add; 0 mul; 0 bitwise)
    Layer 1: 1,775,616 ops (1,769,472 macc; 6,144 comp; 0 add; 0 mul; 0 bitwise)
    Layer 2: 614,400 ops (589,824 macc; 24,576 comp; 0 add; 0 mul; 0 bitwise)
    Layer 3: 5,339,136 ops (5,308,416 macc; 30,720 comp; 0 add; 0 mul; 0 bitwise)
    Layer 4: 248,320 ops (245,760 macc; 2,560 comp; 0 add; 0 mul; 0 bitwise)
    Layer 5: 209,920 ops (204,800 macc; 5,120 comp; 0 add; 0 mul; 0 bitwise)
    Layer 6: 928,000 ops (921,600 macc; 6,400 comp; 0 add; 0 mul; 0 bitwise)
    Layer 7: 82,944 ops (81,920 macc; 1,024 comp; 0 add; 0 mul; 0 bitwise)
    Layer 8: 133,120 ops (131,072 macc; 2,048 comp; 0 add; 0 mul; 0 bitwise)
    Layer 9: 2,048 ops (0 macc; 0 comp; 2,048 add; 0 mul; 0 bitwise)
    Layer 10: 12,800 ops (12,800 macc; 0 comp; 0 add; 0 mul; 0 bitwise)

  RESOURCE USAGE
  Weight memory: 183,776 bytes out of 442,368 bytes total (41.5%)
  Bias memory:   740 bytes out of 2,048 bytes total (36.1%)
*/


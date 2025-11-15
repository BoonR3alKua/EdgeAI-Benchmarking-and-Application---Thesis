/*******************************************************************************
* Copyright (C) 2019-2024 Maxim Integrated Products, Inc., All rights Reserved.
*
* This software is protected by copyright laws of the United States and
* of foreign countries. This material may also be protected by patent laws
* and technology transfer regulations of the United States and of foreign
* countries. This software is furnished under a license agreement and/or a
* nondisclosure agreement and may only be used or reproduced in accordance
* with the terms of those agreements. Dissemination of this information to
* any party or parties not specified in the license agreement and/or
* nondisclosure agreement is expressly prohibited.
*
* The above copyright notice and this permission notice shall be included
* in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
* OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL MAXIM INTEGRATED BE LIABLE FOR ANY CLAIM, DAMAGES
* OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
* ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
* OTHER DEALINGS IN THE SOFTWARE.
*
* Except as contained in this notice, the name of Maxim Integrated
* Products, Inc. shall not be used except as stated in the Maxim Integrated
* Products, Inc. Branding Policy.
*
* The mere transfer of this software does not imply any licenses
* of trade secrets, proprietary technology, copyrights, patents,
* trademarks, maskwork rights, or any other form of intellectual
* property whatsoever. Maxim Integrated Products, Inc. retains all
* ownership rights.
*******************************************************************************/

// my_cifar100_mobilenetv2
// Modified to receive CIFAR-100 images via UART and perform inference
// 
// IMPORTANT NOTES:
// 1. CNN input memory address (0x50400000) verified from original load_input()
// 2. softmax_q17p14_q15 function available (check softmax.c/h)
// 3. Ensure UART0 is properly connected to sender on COM5
// 4. Baud rate: 115200

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
 * Original load_input() used: memcpy32((uint32_t *) 0x50400000, input_0, 1024);
 * This matches the generated code exactly.
 */
void load_input_from_buffer(void)
{
    // Load all 3072 bytes as 1024 words to CNN input memory
    // Address confirmed from original generated load_input() function
    memcpy32((uint32_t *)0x50400000, input_data, 1024);
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
    
    printf("\n*** CIFAR-100 CNN Inference via UART (MobileNetV2) ***\n");
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
  Hardware: 4,312,320 ops (4,243,712 macc; 67,584 comp; 1,024 add; 0 mul; 0 bitwise)
    Layer 0: 344,064 ops (331,776 macc; 12,288 comp; 0 add; 0 mul; 0 bitwise)
    Layer 1: 1,339,392 ops (1,327,104 macc; 12,288 comp; 0 add; 0 mul; 0 bitwise)
    Layer 2: 12,288 ops (0 macc; 12,288 comp; 0 add; 0 mul; 0 bitwise)
    Layer 3: 79,872 ops (73,728 macc; 6,144 comp; 0 add; 0 mul; 0 bitwise)
    Layer 4: 1,333,248 ops (1,327,104 macc; 6,144 comp; 0 add; 0 mul; 0 bitwise)
    Layer 5: 102,400 ops (98,304 macc; 4,096 comp; 0 add; 0 mul; 0 bitwise)
    Layer 6: 4,096 ops (0 macc; 4,096 comp; 0 add; 0 mul; 0 bitwise)
    Layer 7: 34,816 ops (32,768 macc; 2,048 comp; 0 add; 0 mul; 0 bitwise)
    Layer 8: 591,872 ops (589,824 macc; 2,048 comp; 0 add; 0 mul; 0 bitwise)
    Layer 9: 50,688 ops (49,152 macc; 1,536 comp; 0 add; 0 mul; 0 bitwise)
    Layer 10: 1,536 ops (0 macc; 1,536 comp; 0 add; 0 mul; 0 bitwise)
    Layer 11: 19,200 ops (18,432 macc; 768 comp; 0 add; 0 mul; 0 bitwise)
    Layer 12: 332,544 ops (331,776 macc; 768 comp; 0 add; 0 mul; 0 bitwise)
    Layer 13: 25,088 ops (24,576 macc; 512 comp; 0 add; 0 mul; 0 bitwise)
    Layer 14: 33,792 ops (32,768 macc; 1,024 comp; 0 add; 0 mul; 0 bitwise)
    Layer 15: 1,024 ops (0 macc; 0 comp; 1,024 add; 0 mul; 0 bitwise)
    Layer 16: 6,400 ops (6,400 macc; 0 comp; 0 add; 0 mul; 0 bitwise)

  RESOURCE USAGE
  Weight memory: 49,844 bytes out of 442,368 bytes total (11.3%)
  Bias memory:   468 bytes out of 2,048 bytes total (22.9%)
*/


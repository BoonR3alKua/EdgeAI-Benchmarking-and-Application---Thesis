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

// my_cifar10_alexnet_small
// Modified to receive CIFAR-10 images (scaled to 64x64) via UART and perform inference
// 
// IMPORTANT NOTES:
// 1. CNN input memory address (0x50400000) verified from original load_input()
// 2. softmax_q17p14_q15 function available (check softmax.c/h)
// 3. Ensure UART0 is properly connected to sender on COM5
// 4. Baud rate: 115200
// 5. Image size: 64x64x3 = 12288 bytes (scaled up from CIFAR-10 32x32)

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include "mxc_device.h"
#include "board.h"
#include "mxc.h"
#include "cnn.h"
#include "uart.h"
#include "dma.h"
#include "led.h"
#include "nvic_table.h"

// Softmax function prototype (should be in softmax.h if it exists)
extern void softmax_q17p14_q15(const q31_t *vec_in, const uint16_t dim_vec, q15_t *p_out);

// UART Configuration
#define UART_BAUD 115200
#define RX_UART MXC_UART0  // Console UART (COM5) - used for both data and console

// UART reception flags and callback
volatile int UART_READ_FLAG = 0;

// Image Configuration - 64x64 scaled CIFAR-10 images
#define IMAGE_SIZE 64
#define NUM_CHANNELS 3
#define BYTES_PER_IMAGE (IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS)  // 12288 bytes

// Protocol markers - Using unlikely byte sequences for image data
// Start: 0xDE 0xAD 0xBE 0xEF (DEADBEEF - classic debug marker)
// End:   0xCA 0xFE 0xBA 0xBE (CAFEBABE - Java class file marker)
#define START_MARKER_1 0xDE
#define START_MARKER_2 0xAD
#define START_MARKER_3 0xBE
#define START_MARKER_4 0xEF
#define END_MARKER_1 0xCA
#define END_MARKER_2 0xFE
#define END_MARKER_3 0xBA
#define END_MARKER_4 0xBE

// Buffers
static uint8_t rx_buffer[BYTES_PER_IMAGE + 16];  // Extra space for protocol overhead
static uint32_t input_data[BYTES_PER_IMAGE / 4];  // 3072 words for CNN input (12288 bytes)

// CNN outputs
static int32_t ml_data[CNN_NUM_OUTPUTS];
static q15_t ml_softmax[CNN_NUM_OUTPUTS];

volatile uint32_t cnn_time; // Stopwatch

/**
 * UART DMA callback (called when DMA transfer completes)
 */
void uartReadCallback(mxc_uart_req_t *req, int error)
{
    UART_READ_FLAG = error;
}

/**
 * Receive a single byte with timeout
 * Returns: byte value (0-255) on success, negative on error/timeout
 */
int uart_read_byte_timeout(mxc_uart_regs_t *uart, uint32_t timeout_loops)
{
    uint32_t loops = 0;
    int byte_read;
    
    while (loops < timeout_loops) {
        byte_read = MXC_UART_ReadCharacter(uart);
        if (byte_read >= 0) {
            return byte_read;
        }
        loops++;
    }
    
    return E_TIME_OUT;
}

/**
 * Receive image via UART with protocol framing
 * For 64x64 scaled CIFAR-10 images (12288 bytes)
 */
int receive_image_uart(uint32_t *image_idx, uint8_t *label)
{
    int bytes_received = 0;
    int byte_read;
    uint8_t byte;
    int state = 0;  // 0-3: waiting for 4-byte start marker, 4: reading data
    uint32_t timeout_loops = 10000000;  // Large number of loops for timeout (~30 sec at 60MHz)
    
    printf("Waiting for image...\n");
    
    // State machine to receive packet with 4-byte start marker
    while (1) {
        // Read one byte with timeout
        byte_read = uart_read_byte_timeout(RX_UART, timeout_loops);
        if (byte_read < 0) {
            if (byte_read == E_TIME_OUT) {
                printf("Error: UART timeout waiting for data\n");
            }
            return byte_read;  // Error or timeout
        }
        byte = (uint8_t)byte_read;
        
        switch (state) {
            case 0:  // Waiting for first start marker byte (0xDE)
                if (byte == START_MARKER_1) {
                    state = 1;
                } else {
                    state = 0;
                }
                break;
                
            case 1:  // Waiting for second start marker byte (0xAD)
                if (byte == START_MARKER_2) {
                    state = 2;
                } else if (byte == START_MARKER_1) {
                    state = 1;  // Stay in state 1 if we get another 0xDE
                } else {
                    state = 0;
                }
                break;
                
            case 2:  // Waiting for third start marker byte (0xBE)
                if (byte == START_MARKER_3) {
                    state = 3;
                } else if (byte == START_MARKER_1) {
                    state = 1;
                } else {
                    state = 0;
                }
                break;
                
            case 3:  // Waiting for fourth start marker byte (0xEF)
                if (byte == START_MARKER_4) {
                    bytes_received = 0;
                    state = 4;
                    // Do NOT print here - it blocks UART reception!
                } else if (byte == START_MARKER_1) {
                    state = 1;
                } else {
                    state = 0;
                }
                break;
                
            case 4:  // Reading packet data
                // Check for buffer overflow before writing
                if (bytes_received >= sizeof(rx_buffer)) {
                    printf("Error: Buffer overflow, resetting\n");
                    state = 0;
                    bytes_received = 0;
                    break;
                }
                
                rx_buffer[bytes_received++] = byte;
                
                // NOTE: Do NOT print during reception - it causes data loss!
                // printf() to UART0 blocks incoming data reception
                
                // Check if we've received all data (4 bytes idx + 1 byte label + 12288 bytes image)
                // No end markers needed since packet size is fixed
                // Total expected in buffer: 4 + 1 + 12288 = 12293 bytes
                if (bytes_received >= (4 + 1 + BYTES_PER_IMAGE)) {
                    // Do NOT print during reception - wait until complete!
                    
                    // Extract image index (little-endian)
                    *image_idx = ((uint32_t)rx_buffer[0]) |
                                 ((uint32_t)rx_buffer[1] << 8) |
                                 ((uint32_t)rx_buffer[2] << 16) |
                                 ((uint32_t)rx_buffer[3] << 24);
                    
                    // Extract label
                    *label = rx_buffer[4];
                    
                    // Copy image data (starts at index 5, 12288 bytes)
                    memcpy(input_data, &rx_buffer[5], BYTES_PER_IMAGE);
                    
                    // Reception complete - print simple confirmation
                    printf("Received Image %lu (Label: %d)\n", *image_idx, *label);
                    return E_NO_ERROR;
                }
                break;
        }
    }
    
    return E_FAIL;
}

/**
 * Load input data from received buffer into CNN
 * CIFAR-10 scaled to 64x64: 3 channels, 64x64 HWC format
 * Total: 12288 bytes (64x64x3)
 * 
 * Original load_input() used: memcpy32((uint32_t *) 0x50400000, input_0, 4096);
 * This matches the generated code exactly.
 */
void load_input_from_buffer(void)
{
    // Load all 12288 bytes as 4096 words to CNN input memory
    // Address confirmed from original generated load_input() function
    memcpy32((uint32_t *)0x50400000, input_data, 4096);
}

/**
 * Perform CNN inference and return classification
 * Returns the CNN inference time in microseconds via cnn_time global
 */
int classify_image(void)
{
    int i, max_idx = 0;
    int32_t max_val = ml_data[0];
    
    cnn_time = 0;  // Reset timer before starting CNN
    cnn_start();
    
    // Disable Deep Sleep mode to allow CNN interrupt to wake CPU
    SCB->SCR &= ~SCB_SCR_SLEEPDEEP_Msk;
    
    // Wait for CNN to complete using WFI (Wait For Interrupt)
    while (cnn_time == 0) {
        asm volatile("wfi");  // More power efficient than busy wait
    }
    
    // cnn_time now contains inference time in microseconds (set by CNN_ISR)
    
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
    uint32_t total_cnn_time = 0;  // Accumulate CNN inference times
    float avg_cnn_time = 0.0f;
    uint32_t min_cnn_time = 0xFFFFFFFF;
    uint32_t max_cnn_time = 0;
    
    // Initialize system clocks and cache
    MXC_ICC_Enable(MXC_ICC0);
    MXC_SYS_Clock_Select(MXC_SYS_CLOCK_IPO);
    SystemCoreClockUpdate();
    
    printf("\n\n**************** CIFAR-10 CNN Inference via UART ******************\n");
    printf("Model: AlexNet-Small (64x64 input)\n");
    printf("UART: %d baud on UART0\n", UART_BAUD);
    printf("NOTE: Close this console before running sender.py\n");
    printf("      Results will be visible after sender completes.\n");
    printf("********************************************************************\n\n");
    
    // Initialize UART for data reception
    printf("Initializing UART...\n");
    error = MXC_UART_Init(RX_UART, UART_BAUD, MXC_UART_APB_CLK);
    if (error < E_NO_ERROR) {
        printf("-->Error initializing UART: %d\n", error);
        printf("-->Example Failed\n");
        return error;
    }
    printf("-->UART Initialized\n\n");
    
    // Initialize CNN
    printf("Initializing CNN...\n");
    cnn_enable(MXC_S_GCR_PCLKDIV_CNNCLKSEL_PCLK, MXC_S_GCR_PCLKDIV_CNNCLKDIV_DIV1);
    cnn_init();
    cnn_load_weights();
    cnn_load_bias();
    cnn_configure();
    printf("-->CNN initialized\n\n");
    
    printf("Ready to receive images. Waiting for data...\n\n");
    
    LED_On(0);  // Indicate ready
    
    // Main loop: receive and classify images
    while (1) {
        // Receive image via UART
        error = receive_image_uart(&image_idx, &ground_truth_label);
        if (error != E_NO_ERROR) {
            printf("-->Error receiving image: %d\n", error);
            printf("Resetting and waiting for next image...\n\n");
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
        
        // Track CNN inference timing
        total_cnn_time += cnn_time;
        if (cnn_time < min_cnn_time) min_cnn_time = cnn_time;
        if (cnn_time > max_cnn_time) max_cnn_time = cnn_time;
        avg_cnn_time = (float)total_cnn_time / (total_count + 1);
        
        // Check accuracy
        total_count++;
        total_confidence += confidence;
        avg_confidence = total_confidence / total_count;
        
        if (predicted_class == ground_truth_label) {
            correct_count++;
        }
        
        // Print results with confidence and timing
        printf("----------------------------------------\n");
        printf("Image %lu Results:\n", image_idx);
        printf("  Predicted:    %d (%.2f%% confidence)\n", predicted_class, (double)confidence);
        printf("  Actual:       %d\n", ground_truth_label);
        printf("  Status:       %s\n", (predicted_class == ground_truth_label) ? "CORRECT" : "WRONG");
        printf("  CNN Time:     %lu us (%.2f ms)\n", cnn_time, (double)cnn_time / 1000.0);
        printf("Overall Statistics:\n");
        printf("  Accuracy:     %d/%d (%.2f%%)\n",
               correct_count, total_count,
               ((double)correct_count / (double)total_count) * 100.0);
        printf("  Avg Conf:     %.2f%%\n", (double)avg_confidence);
        printf("  Avg CNN Time: %.2f us (%.2f ms)\n", (double)avg_cnn_time, (double)avg_cnn_time / 1000.0);
        printf("  Min CNN Time: %lu us (%.2f ms)\n", min_cnn_time, (double)min_cnn_time / 1000.0);
        printf("  Max CNN Time: %lu us (%.2f ms)\n", max_cnn_time, (double)max_cnn_time / 1000.0);
        printf("  Throughput:   %.2f images/sec\n", 1000000.0 / (double)avg_cnn_time);
        printf("----------------------------------------\n\n");
        
        // Ensure all output is transmitted before continuing
        MXC_UART_ClearRXFIFO(RX_UART);  // Clear any stale data
        MXC_Delay(MXC_DELAY_MSEC(100));  // Small delay to ensure transmission complete
        
        // Reconfigure CNN for next inference
        cnn_init();
        cnn_configure();
    }
    
    return E_NO_ERROR;
}

/*
  SUMMARY OF OPS
  Hardware: 27,039,360 ops (26,887,424 macc; 147,840 comp; 4,096 add; 0 mul; 0 bitwise)
    Layer 0: 929,792 ops (884,736 macc; 45,056 comp; 0 add; 0 mul; 0 bitwise)
    Layer 1: 4,767,744 ops (4,718,592 macc; 49,152 comp; 0 add; 0 mul; 0 bitwise)
    Layer 2: 9,453,568 ops (9,437,184 macc; 16,384 comp; 0 add; 0 mul; 0 bitwise)
    Layer 3: 9,453,568 ops (9,437,184 macc; 16,384 comp; 0 add; 0 mul; 0 bitwise)
    Layer 4: 2,379,776 ops (2,359,296 macc; 20,480 comp; 0 add; 0 mul; 0 bitwise)
    Layer 5: 4,096 ops (0 macc; 0 comp; 4,096 add; 0 mul; 0 bitwise)
    Layer 6: 16,640 ops (16,384 macc; 256 comp; 0 add; 0 mul; 0 bitwise)
    Layer 7: 32,896 ops (32,768 macc; 128 comp; 0 add; 0 mul; 0 bitwise)
    Layer 8: 1,280 ops (1,280 macc; 0 comp; 0 add; 0 mul; 0 bitwise)

  RESOURCE USAGE
  Weight memory: 180,320 bytes out of 442,368 bytes total (40.8%)
  Bias memory:   682 bytes out of 2,048 bytes total (33.3%)
*/


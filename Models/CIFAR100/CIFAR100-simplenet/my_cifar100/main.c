/*******************************************************************************
* Modified main.c to receive CIFAR-100 images via UART and perform inference
* 
* IMPORTANT NOTES:
* 1. CNN input memory address (0x50400000) verified from original load_input()
* 2. softmax_q17p14_q15 function available (check softmax.c/h)
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

// Protocol markers - Using unlikely byte sequences for image data
// Start: 0xDE 0xAD 0xBE 0xEF (DEADBEEF - classic debug marker)
#define START_MARKER_1 0xDE
#define START_MARKER_2 0xAD
#define START_MARKER_3 0xBE
#define START_MARKER_4 0xEF

// Buffers
static uint8_t rx_buffer[BYTES_PER_IMAGE + 16];  // Extra space for protocol overhead
static uint32_t input_data[BYTES_PER_IMAGE / 4];  // 768 words for CNN input

// CNN outputs
static int32_t ml_data[CNN_NUM_OUTPUTS];
static q15_t ml_softmax[CNN_NUM_OUTPUTS];

volatile uint32_t cnn_time;

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
 * Fixed packet size: START_MARKER(4) + IDX(4) + LABEL(1) + IMAGE(3072) = 3081 bytes
 */
int receive_image_uart(uint32_t *image_idx, uint8_t *label)
{
    int bytes_received = 0;
    int byte_read;
    uint8_t byte;
    int state = 0;  // 0-3: waiting for 4-byte start marker, 4: reading data
    uint32_t timeout_loops = 10000000;  // Large number of loops for timeout (~30 sec at 60MHz)
    
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
                
                // Fixed packet: 4 bytes idx + 1 byte label + 3072 bytes image = 3077 bytes
                if (bytes_received >= (4 + 1 + BYTES_PER_IMAGE)) {
                    // Extract image index
                    *image_idx = *((uint32_t *)&rx_buffer[0]);
                    
                    // Extract label
                    *label = rx_buffer[4];
                    
                    // Copy image data
                    memcpy(input_data, &rx_buffer[5], BYTES_PER_IMAGE);
                    
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
    
    printf("\n\n**************** CIFAR-100 CNN Inference via UART ******************\n");
    printf("Model: SimpleNet\n");
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
  Hardware: 18,607,744 ops (18,461,184 macc; 146,560 comp; 0 add; 0 mul; 0 bitwise)
    Layer 0: 458,752 ops (442,368 macc; 16,384 comp; 0 add; 0 mul; 0 bitwise)
    Layer 1: 2,969,600 ops (2,949,120 macc; 20,480 comp; 0 add; 0 mul; 0 bitwise)
    Layer 2: 3,706,880 ops (3,686,400 macc; 20,480 comp; 0 add; 0 mul; 0 bitwise)
    Layer 3: 3,706,880 ops (3,686,400 macc; 20,480 comp; 0 add; 0 mul; 0 bitwise)
    Layer 4: 947,200 ops (921,600 macc; 25,600 comp; 0 add; 0 mul; 0 bitwise)
    Layer 5: 926,720 ops (921,600 macc; 5,120 comp; 0 add; 0 mul; 0 bitwise)
    Layer 6: 2,038,784 ops (2,027,520 macc; 11,264 comp; 0 add; 0 mul; 0 bitwise)
    Layer 7: 1,230,848 ops (1,216,512 macc; 14,336 comp; 0 add; 0 mul; 0 bitwise)
    Layer 8: 1,330,176 ops (1,327,104 macc; 3,072 comp; 0 add; 0 mul; 0 bitwise)
    Layer 9: 668,160 ops (663,552 macc; 4,608 comp; 0 add; 0 mul; 0 bitwise)
    Layer 10: 200,192 ops (196,608 macc; 3,584 comp; 0 add; 0 mul; 0 bitwise)
    Layer 11: 262,656 ops (262,144 macc; 512 comp; 0 add; 0 mul; 0 bitwise)
    Layer 12: 148,096 ops (147,456 macc; 640 comp; 0 add; 0 mul; 0 bitwise)
    Layer 13: 12,800 ops (12,800 macc; 0 comp; 0 add; 0 mul; 0 bitwise)

  RESOURCE USAGE
  Weight memory: 381,792 bytes out of 442,368 bytes total (86.3%)
  Bias memory:   1,220 bytes out of 2,048 bytes total (59.6%)
*/


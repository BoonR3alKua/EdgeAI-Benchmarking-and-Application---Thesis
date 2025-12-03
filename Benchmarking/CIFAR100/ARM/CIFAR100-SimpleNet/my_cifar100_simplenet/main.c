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

// my_cifar100_simplenet - Queue-based Benchmark Version
// 
// Features:
// 1. Queue-based image receiving (receive all, then process)
// 2. Warmup iterations before benchmark
// 3. Continuous CNN inference until queue is empty
// 4. Detailed benchmark statistics with formatted output
//
// Memory mapping:
//   Channel 0 (R): 0x50400000 - 256 words (1024 bytes)
//   Channel 1 (G): 0x50800000 - 256 words (1024 bytes)
//   Channel 2 (B): 0x50c00000 - 256 words (1024 bytes)
//
// Protocol: [DEADBEEF][NUM_IMAGES(4)][Label0][Image0][Label1][Image1]...

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include "mxc.h"
#include "cnn.h"
#include "led.h"

// Softmax function prototype
extern void softmax_q17p14_q15(const q31_t *vec_in, const uint16_t dim_vec, q15_t *p_out);

// UART Configuration
#define UART_BAUD 115200
#define RX_UART MXC_UART0  // Same UART for printf and receiving data

// Image Configuration - 32x32 CHW format
#define IMAGE_SIZE 32
#define NUM_CHANNELS 3
#define BYTES_PER_CHANNEL (IMAGE_SIZE * IMAGE_SIZE)  // 1024 bytes per channel
#define BYTES_PER_IMAGE (BYTES_PER_CHANNEL * NUM_CHANNELS)  // 3072 bytes total

// Queue Configuration
#define MAX_QUEUE_SIZE 30  // Maximum images in queue (limited by SRAM)

// Benchmark Configuration
#define WARMUP_ITERATIONS 5

// Protocol markers - DEADBEEF
#define START_MARKER_1 0xDE
#define START_MARKER_2 0xAD
#define START_MARKER_3 0xBE
#define START_MARKER_4 0xEF

// Image Queue Entry
typedef struct {
    uint32_t input_ch0[256];  // Channel 0 (R) - 1024 bytes
    uint32_t input_ch1[256];  // Channel 1 (G) - 1024 bytes
    uint32_t input_ch2[256];  // Channel 2 (B) - 1024 bytes
    uint8_t label;
    uint8_t valid;
} ImageQueueEntry;

// Image Queue (circular buffer)
static ImageQueueEntry image_queue[MAX_QUEUE_SIZE];
static volatile uint32_t queue_head = 0;  // Next position to write
static volatile uint32_t queue_tail = 0;  // Next position to read
static volatile uint32_t queue_count = 0; // Number of items in queue

// Receive buffer
static uint8_t rx_buffer[BYTES_PER_IMAGE + 8];

// CNN outputs
static int32_t ml_data[CNN_NUM_OUTPUTS];
static q15_t ml_softmax[CNN_NUM_OUTPUTS];

volatile uint32_t cnn_time;

// Benchmark results storage
typedef struct {
    uint32_t cnn_time_us;
    int predicted_class;
    uint8_t ground_truth;
    uint8_t correct;
    float confidence;
} BenchmarkResult;

static BenchmarkResult benchmark_results[MAX_QUEUE_SIZE];

/**
 * Check if queue is full
 */
static inline int queue_is_full(void) {
    return queue_count >= MAX_QUEUE_SIZE;
}

/**
 * Check if queue is empty
 */
static inline int queue_is_empty(void) {
    return queue_count == 0;
}

/**
 * Add image to queue
 */
static int queue_push(uint8_t *image_data, uint8_t label) {
    if (queue_is_full()) {
        return -1;
    }
    
    ImageQueueEntry *entry = &image_queue[queue_head];
    
    // Copy channel data (CHW format)
    memcpy(entry->input_ch0, image_data, BYTES_PER_CHANNEL);
    memcpy(entry->input_ch1, image_data + BYTES_PER_CHANNEL, BYTES_PER_CHANNEL);
    memcpy(entry->input_ch2, image_data + 2 * BYTES_PER_CHANNEL, BYTES_PER_CHANNEL);
    entry->label = label;
    entry->valid = 1;
    
    queue_head = (queue_head + 1) % MAX_QUEUE_SIZE;
    queue_count++;
    
    return 0;
}

/**
 * Get next image from queue (does not remove)
 */
static ImageQueueEntry* queue_peek(void) {
    if (queue_is_empty()) {
        return NULL;
    }
    return &image_queue[queue_tail];
}

/**
 * Remove front image from queue
 */
static void queue_pop(void) {
    if (!queue_is_empty()) {
        image_queue[queue_tail].valid = 0;
        queue_tail = (queue_tail + 1) % MAX_QUEUE_SIZE;
        queue_count--;
    }
}

/**
 * Receive a single byte with timeout
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
 * Receive a single byte (blocking)
 */
int uart_read_byte_blocking(mxc_uart_regs_t *uart)
{
    int byte_read;
    while ((byte_read = MXC_UART_ReadCharacter(uart)) < 0);
    return byte_read;
}

/**
 * Wait for start marker and receive number of images
 * Returns: E_NO_ERROR on success, negative on error
 */
int wait_for_benchmark_start(uint32_t *num_images)
{
    int byte_read;
    int state = 0;
    uint32_t timeout_loops = 100000000;  // Long timeout for initial connection
    
    printf("[WAIT] Waiting for benchmark data...\n");
    
    while (1) {
        byte_read = uart_read_byte_timeout(RX_UART, timeout_loops);
        if (byte_read < 0) {
            return byte_read;
        }
        
        switch (state) {
            case 0: if (byte_read == START_MARKER_1) state = 1; break;
            case 1: state = (byte_read == START_MARKER_2) ? 2 : (byte_read == START_MARKER_1) ? 1 : 0; break;
            case 2: state = (byte_read == START_MARKER_3) ? 3 : (byte_read == START_MARKER_1) ? 1 : 0; break;
            case 3:
                if (byte_read == START_MARKER_4) {
                    // Read number of images (4 bytes, little-endian)
                    uint8_t num_bytes[4];
                    for (int i = 0; i < 4; i++) {
                        num_bytes[i] = uart_read_byte_blocking(RX_UART);
                    }
                    *num_images = ((uint32_t)num_bytes[0]) |
                                  ((uint32_t)num_bytes[1] << 8) |
                                  ((uint32_t)num_bytes[2] << 16) |
                                  ((uint32_t)num_bytes[3] << 24);
                    return E_NO_ERROR;
                }
                state = (byte_read == START_MARKER_1) ? 1 : 0;
                break;
        }
    }
}

/**
 * Receive all images into queue
 */
int receive_all_images(uint32_t num_images)
{
    uint32_t received = 0;
    
    // Send READY signal to sender - must flush before starting to receive
    printf("READY\n");
    fflush(stdout);
    
    while (received < num_images && !queue_is_full()) {
        // Read label (1 byte)
        uint8_t label = uart_read_byte_blocking(RX_UART);
        
        // Read image data (3072 bytes)
        for (int i = 0; i < BYTES_PER_IMAGE; i++) {
            rx_buffer[i] = uart_read_byte_blocking(RX_UART);
        }
        
        // Add to queue
        if (queue_push(rx_buffer, label) == 0) {
            received++;
        } else {
            printf("[ERROR] Queue full at image %lu\n", received);
            break;
        }
    }
    
    printf("[RECV] Received %lu images into queue.\n", received);
    return received;
}

/**
 * Load image from queue entry into CNN memory
 */
void load_input_from_queue(ImageQueueEntry *entry)
{
    memcpy32((uint32_t *)0x50400000, entry->input_ch0, 256);
    memcpy32((uint32_t *)0x50800000, entry->input_ch1, 256);
    memcpy32((uint32_t *)0x50c00000, entry->input_ch2, 256);
}

/**
 * Perform CNN inference and return classification
 * Also stores confidence in the provided pointer
 */
int classify_image(float *confidence)
{
    int i, max_idx = 0;
    int32_t max_val;
    
    cnn_time = 0;
    cnn_start();
    
    SCB->SCR &= ~SCB_SCR_SLEEPDEEP_Msk;
    while (cnn_time == 0) {
        asm volatile("wfi");
    }
    
    cnn_unload((uint32_t *)ml_data);
    softmax_q17p14_q15((const q31_t *)ml_data, CNN_NUM_OUTPUTS, ml_softmax);
    
    max_val = ml_data[0];
    for (i = 1; i < CNN_NUM_OUTPUTS; i++) {
        if (ml_data[i] > max_val) {
            max_val = ml_data[i];
            max_idx = i;
        }
    }
    
    // Calculate confidence as percentage (ml_softmax is Q15 format: 0x7FFF = 100%)
    if (confidence != NULL) {
        *confidence = (float)ml_softmax[max_idx] / 32768.0f * 100.0f;
    }
    
    return max_idx;
}

/**
 * Run warmup iterations using first image in queue
 */
void run_warmup(int iterations)
{
    ImageQueueEntry *entry = queue_peek();
    if (entry == NULL) {
        printf("[WARMUP] No images in queue for warmup!\n");
        return;
    }
    
    printf("[WARMUP] Running %d iterations...\n", iterations);
    
    for (int i = 0; i < iterations; i++) {
        load_input_from_queue(entry);
        cnn_init();
        cnn_configure();
        classify_image(NULL);  // Don't need confidence for warmup
    }
    
    printf("[WARMUP] %d iterations done.\n", iterations);
}

/**
 * Run benchmark on all queued images
 */
void run_benchmark(uint32_t num_images)
{
    uint32_t total_time = 0;
    uint32_t min_time = 0xFFFFFFFF;
    uint32_t max_time = 0;
    int correct_count = 0;
    float total_confidence = 0.0f;
    uint32_t idx = 0;
    
    printf("[BENCH] Start Loop (N=%lu)\n", num_images);
    
    // Process all images in queue
    while (!queue_is_empty() && idx < num_images) {
        ImageQueueEntry *entry = queue_peek();
        if (entry == NULL) break;
        
        // Load and run inference
        load_input_from_queue(entry);
        cnn_init();
        cnn_configure();
        
        float confidence;
        int predicted = classify_image(&confidence);
        
        // Store result
        benchmark_results[idx].cnn_time_us = cnn_time;
        benchmark_results[idx].predicted_class = predicted;
        benchmark_results[idx].ground_truth = entry->label;
        benchmark_results[idx].correct = (predicted == entry->label) ? 1 : 0;
        benchmark_results[idx].confidence = confidence;
        
        // Update statistics
        total_time += cnn_time;
        total_confidence += confidence;
        if (cnn_time < min_time) min_time = cnn_time;
        if (cnn_time > max_time) max_time = cnn_time;
        if (predicted == entry->label) correct_count++;
        
        // Print iteration result
        printf("Iter %02lu: %.1fms | Class: %d | GT: %d | Conf: %.1f%% | %s\n", 
               idx + 1, 
               (double)cnn_time / 1000.0,
               predicted,
               entry->label,
               (double)confidence,
               (predicted == entry->label) ? "OK" : "FAIL");
        
        queue_pop();
        idx++;
    }
    
    printf("[BENCH] Loop End.\n");
    
    // =========================================================================
    // BENCHMARK STATISTICS CALCULATION
    // =========================================================================
    
    // Basic timing statistics
    float avg_time = (float)total_time / (float)idx;
    float fps = 1000000.0f / avg_time;
    float accuracy = ((float)correct_count / (float)idx) * 100.0f;
    float avg_confidence = total_confidence / (float)idx;
    
    // Model constants
    const uint32_t total_ops = 18607744;      // Total operations per inference
    const uint32_t total_macs = 18461184;     // Total MACs per inference
    const uint32_t weight_bytes = 381792;     // Weight memory used
    const uint32_t bias_bytes = 1220;         // Bias memory used
    const uint32_t input_bytes = 3072;        // Input size (32x32x3)
    const uint32_t output_bytes = 400;        // Output size (100 classes × 4 bytes)
    
    // MAX78000 NPU specs
    const float npu_freq_mhz = 50.0f;         // NPU clock frequency (PCLK = 100MHz / 2)
    const uint32_t num_processors = 64;       // Number of parallel processors
    const float peak_tops = (npu_freq_mhz * 1e6f * num_processors * 2) / 1e12f;  // Peak TOPS (×2 for MAC = 2 ops)
    
    // Latency calculations
    float core_latency_us = avg_time;         // CNN accelerator time (what we measure)
    float load_time_us = 10.0f;               // Estimated: cnn_init + cnn_configure (~10us)
    float unload_time_us = 5.0f;              // Estimated: cnn_unload + softmax (~5us)
    float system_latency_us = core_latency_us + load_time_us + unload_time_us;
    
    // TOPS calculations
    float effective_tops = (float)total_ops / (core_latency_us / 1e6f) / 1e12f;
    float effective_macs_tops = (float)total_macs / (core_latency_us / 1e6f) / 1e12f;
    float utilization_pct = (effective_tops / peak_tops) * 100.0f;
    
    // Memory bandwidth estimation
    // Data moved per inference: weights (read once, cached) + input + intermediate activations + output
    // For streaming mode, weights are pre-loaded to CNN memory, so only input/output moves through SRAM
    uint32_t data_moved_bytes = input_bytes + output_bytes;  // Minimum data movement
    float bandwidth_mb_per_sec = ((float)data_moved_bytes * fps) / (1024.0f * 1024.0f);
    
    // Memory efficiency: how much computation per byte of data moved
    float ops_per_byte = (float)total_ops / (float)data_moved_bytes;
    float arithmetic_intensity = (float)total_macs / (float)(weight_bytes + input_bytes);
    
    // =========================================================================
    // PRINT REPORT
    // =========================================================================
    
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║              BENCHMARK REPORT - SimpleNet                    ║\n");
    printf("╠══════════════════════════════════════════════════════════════╣\n");
    
    // Basic Statistics
    printf("║ [BASIC STATISTICS]                                           ║\n");
    printf("║   Total Images:        %-6lu                                ║\n", idx);
    printf("║   Correct:             %d/%lu (%.1f%%)                         \n", 
           correct_count, idx, (double)accuracy);
    printf("║   Avg Confidence:      %.1f%%                                 \n", (double)avg_confidence);
    printf("╠══════════════════════════════════════════════════════════════╣\n");
    
    // Latency Metrics
    printf("║ [LATENCY METRICS]                                            ║\n");
    printf("║   Core Latency (NPU):  %.2f us (%.3f ms)                     \n", 
           (double)core_latency_us, (double)core_latency_us / 1000.0);
    printf("║   System Latency:      %.2f us (%.3f ms)                     \n", 
           (double)system_latency_us, (double)system_latency_us / 1000.0);
    printf("║   Min Latency:         %lu us                                \n", min_time);
    printf("║   Max Latency:         %lu us                                \n", max_time);
    printf("║   Latency Jitter:      %lu us                                \n", max_time - min_time);
    printf("╠══════════════════════════════════════════════════════════════╣\n");
    
    // Throughput Metrics
    printf("║ [THROUGHPUT METRICS]                                         ║\n");
    printf("║   Throughput:          %.2f FPS                              \n", (double)fps);
    printf("║   Total Time:          %.2f ms (%lu images)                  \n", 
           (double)total_time / 1000.0, idx);
    printf("╠══════════════════════════════════════════════════════════════╣\n");
    
    // TOPS Analysis
    printf("║ [TOPS ANALYSIS]                                              ║\n");
    printf("║   Peak TOPS:           %.3f TOPS (theoretical max)           \n", (double)peak_tops);
    printf("║   Effective TOPS:      %.3f TOPS (actual)                    \n", (double)effective_tops);
    printf("║   Effective MACs:      %.3f TMACs/s                          \n", (double)effective_macs_tops);
    printf("║   NPU Utilization:     %.1f%%                                \n", (double)utilization_pct);
    printf("║   Ops/Inference:       %.2f M ops                            \n", (double)total_ops / 1e6);
    printf("║   MACs/Inference:      %.2f M MACs                           \n", (double)total_macs / 1e6);
    printf("╠══════════════════════════════════════════════════════════════╣\n");
    
    // Memory Efficiency
    printf("║ [MEMORY EFFICIENCY]                                          ║\n");
    printf("║   Weight Memory:       %lu bytes (%.1f KB)                   \n", 
           weight_bytes, (double)weight_bytes / 1024.0);
    printf("║   Input Size:          %lu bytes                             \n", input_bytes);
    printf("║   Output Size:         %lu bytes                             \n", output_bytes);
    printf("║   Data Movement:       %lu bytes/inference                   \n", data_moved_bytes);
    printf("║   Bandwidth Used:      %.2f MB/s                             \n", (double)bandwidth_mb_per_sec);
    printf("║   Arithmetic Intensity:%.1f ops/byte                         \n", (double)arithmetic_intensity);
    printf("║   Ops/Byte Moved:      %.1f ops/byte                         \n", (double)ops_per_byte);
    printf("╠══════════════════════════════════════════════════════════════╣\n");
    
    // Final Verdict
    printf("║ [VERDICT]                                                    ║\n");
    if (utilization_pct >= 40.0f) {
        printf("║   NPU Efficiency:      GOOD (>=40%%)                         ║\n");
    } else if (utilization_pct >= 20.0f) {
        printf("║   NPU Efficiency:      MODERATE (20-40%%)                    ║\n");
    } else {
        printf("║   NPU Efficiency:      LOW (<20%%)                           ║\n");
    }
    printf("║   Accuracy:            %s                                  \n",
           (correct_count == (int)idx) ? "PERFECT" : "PARTIAL");
    printf("╚══════════════════════════════════════════════════════════════╝\n");
    
    // Layer-by-layer breakdown (estimated from ops count)
    static const struct {
        const char *name;
        uint32_t ops;
        uint32_t macc;
    } layer_info[] = {
        {"Conv2D-64-3x3",      458752,   442368},   // Layer 0
        {"Conv2D-128-3x3",    2969600,  2949120},   // Layer 1
        {"Conv2D-128-3x3",    3706880,  3686400},   // Layer 2
        {"Conv2D-128-3x3",    3706880,  3686400},   // Layer 3
        {"Conv2D-128-3x3",     947200,   921600},   // Layer 4
        {"Conv2D-256-3x3",     926720,   921600},   // Layer 5
        {"Conv2D-256-3x3",    2038784,  2027520},   // Layer 6
        {"Conv2D-256-3x3",    1230848,  1216512},   // Layer 7
        {"Conv2D-512-3x3",    1330176,  1327104},   // Layer 8
        {"Conv2D-512-3x3",     668160,   663552},   // Layer 9
        {"Conv2D-512-3x3",     200192,   196608},   // Layer 10
        {"Conv2D-512-1x1",     262656,   262144},   // Layer 11
        {"Conv2D-512-1x1",     148096,   147456},   // Layer 12
        {"Dense-100",           12800,    12800},   // Layer 13
    };
    const int num_layers = 14;
    
    printf("\n");
    printf("============================================================\n");
    printf("[LAYERS] Layer-by-Layer Breakdown (SimpleNet)\n");
    printf("============================================================\n");
    printf("Layer | Name              |    Ops    |   MACC   | Est.Time | %%Total\n");
    printf("------|-------------------|-----------|----------|----------|-------\n");
    
    for (int l = 0; l < num_layers; l++) {
        float pct = (float)layer_info[l].ops / (float)total_ops * 100.0f;
        float est_time_us = (float)avg_time * pct / 100.0f;
        printf("  %2d  | %-17s | %9lu | %8lu | %6.1fus | %5.1f%%\n",
               l, layer_info[l].name, layer_info[l].ops, layer_info[l].macc,
               (double)est_time_us, (double)pct);
    }
    
    printf("------|-------------------|-----------|----------|----------|-------\n");
    printf(" ALL  | Total             | %9u | %8u | %6.1fus | 100.0%%\n",
           total_ops, 18461184, (double)avg_time);
    printf("============================================================\n");
}

int main(void)
{
    int error;
    uint32_t num_images;
    
    // Initialize system
    MXC_ICC_Enable(MXC_ICC0);
    MXC_SYS_Clock_Select(MXC_SYS_CLOCK_IPO);
    SystemCoreClockUpdate();
    
    printf("\n\n");
    printf("============================================================\n");
    printf("[INIT] CIFAR-100 SimpleNet Benchmark System\n");
    printf("============================================================\n");
    printf("[INIT] Image Size: 32x32x3 (CHW format)\n");
    printf("[INIT] Queue Size: %d images max\n", MAX_QUEUE_SIZE);
    printf("[INIT] UART: %d baud\n", UART_BAUD);
    
    // Initialize UART
    printf("[INIT] Initializing UART...");
    error = MXC_UART_Init(RX_UART, UART_BAUD, MXC_UART_APB_CLK);
    if (error < E_NO_ERROR) {
        printf(" FAILED (%d)\n", error);
        return error;
    }
    printf(" OK\n");
    
    // Initialize CNN
    printf("[INIT] Loading SimpleNet...");
    cnn_enable(MXC_S_GCR_PCLKDIV_CNNCLKSEL_PCLK, MXC_S_GCR_PCLKDIV_CNNCLKDIV_DIV1);
    cnn_init();
    cnn_load_weights();
    cnn_load_bias();
    cnn_configure();
    printf(" OK\n");
    
    // Print NPU frequency
    uint32_t pclk_freq = SystemCoreClock / 2;  // APB clock is typically half of system clock
    printf("[INIT] NPU Freq: %lu MHz (PCLK)\n", pclk_freq / 1000000);
    printf("============================================================\n\n");
    
    LED_On(0);  // Ready indicator
    
    // Main benchmark loop
    while (1) {
        // Reset queue
        queue_head = 0;
        queue_tail = 0;
        queue_count = 0;
        
        // Wait for benchmark start command
        error = wait_for_benchmark_start(&num_images);
        if (error != E_NO_ERROR) {
            printf("[ERROR] Failed to receive start command: %d\n", error);
            continue;
        }
        
        if (num_images > MAX_QUEUE_SIZE) {
            printf("[WARN] Requested %lu images, limiting to %d\n", num_images, MAX_QUEUE_SIZE);
            num_images = MAX_QUEUE_SIZE;
        }
        
        printf("[INFO] Benchmark requested for %lu images\n", num_images);
        
        // Receive all images into queue
        LED_On(1);
        uint32_t received = receive_all_images(num_images);
        LED_Off(1);
        
        if (received == 0) {
            printf("[ERROR] No images received!\n");
            continue;
        }
        
        // Send ACK to PC
        printf("[ACK] Ready to process %lu images\n", received);
        MXC_UART_ClearRXFIFO(RX_UART);
        
        // Run warmup
        run_warmup(WARMUP_ITERATIONS);
        
        // Run benchmark
        LED_On(1);
        run_benchmark(received);
        LED_Off(1);
        
        printf("\n[DONE]\n\n");
        
        // Clear UART and prepare for next batch
        MXC_UART_ClearRXFIFO(RX_UART);
        MXC_Delay(MXC_DELAY_MSEC(100));
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


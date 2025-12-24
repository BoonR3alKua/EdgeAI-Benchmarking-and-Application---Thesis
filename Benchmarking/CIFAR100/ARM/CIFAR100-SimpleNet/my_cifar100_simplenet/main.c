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
#define MAX_QUEUE_SIZE 20  // Maximum images in queue (limited by SRAM)
#define TOTAL_BATCHES 50   // Total number of batches for 1000 images

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

// Timing measurements for load/unload operations
static uint32_t measured_init_time_us = 0;
static uint32_t measured_configure_time_us = 0;
static uint32_t measured_unload_time_us = 0;
static uint32_t measured_softmax_time_us = 0;
static uint32_t measured_load_input_time_us = 0;
static uint32_t measured_load_weights_time_us = 0;
static uint32_t measured_load_bias_time_us = 0;

// Multi-batch cumulative statistics
static uint32_t total_batches_processed = 0;
static uint32_t total_images_processed = 0;
static uint32_t cumulative_correct = 0;
static uint64_t cumulative_time_us = 0;  // Use 64-bit to avoid overflow
static uint32_t cumulative_min_time = 0xFFFFFFFF;
static uint32_t cumulative_max_time = 0;
static float cumulative_confidence = 0.0f;
static uint8_t warmup_done = 0;  // Flag to track if warmup has been performed

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
 * Measure load/unload times using a calibration run
 */
void measure_overhead_times(ImageQueueEntry *entry)
{
    // Measure load_input time
    MXC_TMR_SW_Start(CNN_INFERENCE_TIMER);
    load_input_from_queue(entry);
    measured_load_input_time_us = MXC_TMR_SW_Stop(CNN_INFERENCE_TIMER);
    
    // Measure cnn_init time
    MXC_TMR_SW_Start(CNN_INFERENCE_TIMER);
    cnn_init();
    measured_init_time_us = MXC_TMR_SW_Stop(CNN_INFERENCE_TIMER);
    
    // Measure cnn_configure time
    MXC_TMR_SW_Start(CNN_INFERENCE_TIMER);
    cnn_configure();
    measured_configure_time_us = MXC_TMR_SW_Stop(CNN_INFERENCE_TIMER);
    
    // Run inference to get valid output data for softmax measurement
    cnn_time = 0;
    cnn_start();
    SCB->SCR &= ~SCB_SCR_SLEEPDEEP_Msk;
    while (cnn_time == 0) {
        asm volatile("wfi");
    }
    
    // Measure cnn_unload time
    MXC_TMR_SW_Start(CNN_INFERENCE_TIMER);
    cnn_unload((uint32_t *)ml_data);
    measured_unload_time_us = MXC_TMR_SW_Stop(CNN_INFERENCE_TIMER);
    
    // Measure softmax time
    MXC_TMR_SW_Start(CNN_INFERENCE_TIMER);
    softmax_q17p14_q15((const q31_t *)ml_data, CNN_NUM_OUTPUTS, ml_softmax);
    measured_softmax_time_us = MXC_TMR_SW_Stop(CNN_INFERENCE_TIMER);
    
    printf("[MEASURE] load_input: %lu us\n", measured_load_input_time_us);
    printf("[MEASURE] cnn_init: %lu us\n", measured_init_time_us);
    printf("[MEASURE] cnn_configure: %lu us\n", measured_configure_time_us);
    printf("[MEASURE] cnn_unload: %lu us\n", measured_unload_time_us);
    printf("[MEASURE] softmax: %lu us\n", measured_softmax_time_us);
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
    
    // Measure overhead times before warmup
    printf("[WARMUP] Measuring overhead times...\n");
    measure_overhead_times(entry);
    
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
 * Accumulates statistics across batches for final report
 */
void run_benchmark(uint32_t num_images, uint32_t batch_num)
{
    uint32_t total_time = 0;
    uint32_t min_time = 0xFFFFFFFF;
    uint32_t max_time = 0;
    int correct_count = 0;
    float total_confidence = 0.0f;
    uint32_t idx = 0;
    
    printf("[BENCH] Batch %lu/%d - Start Loop (N=%lu)\n", batch_num, TOTAL_BATCHES, num_images);
    
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
        
        // Store result (within MAX_QUEUE_SIZE bounds)
        if (idx < MAX_QUEUE_SIZE) {
            benchmark_results[idx].cnn_time_us = cnn_time;
            benchmark_results[idx].predicted_class = predicted;
            benchmark_results[idx].ground_truth = entry->label;
            benchmark_results[idx].correct = (predicted == entry->label) ? 1 : 0;
            benchmark_results[idx].confidence = confidence;
        }
        
        // Update statistics
        total_time += cnn_time;
        total_confidence += confidence;
        if (cnn_time < min_time) min_time = cnn_time;
        if (cnn_time > max_time) max_time = cnn_time;
        if (predicted == entry->label) correct_count++;
        
        // Print iteration result (show global image number)
        uint32_t global_idx = total_images_processed + idx + 1;
        printf("Img %04lu: %.1fms | Class: %d | GT: %d | Conf: %.1f%% | %s\n", 
               global_idx,
               (double)cnn_time / 1000.0,
               predicted,
               entry->label,
               (double)confidence,
               (predicted == entry->label) ? "OK" : "FAIL");
        
        queue_pop();
        idx++;
    }
    
    printf("[BENCH] Batch %lu/%d - Loop End.\n", batch_num, TOTAL_BATCHES);
    
    // Update cumulative statistics
    total_batches_processed++;
    total_images_processed += idx;
    cumulative_correct += correct_count;
    cumulative_time_us += total_time;
    cumulative_confidence += total_confidence;
    if (min_time < cumulative_min_time) cumulative_min_time = min_time;
    if (max_time > cumulative_max_time) cumulative_max_time = max_time;
    
    // Print batch summary
    float batch_accuracy = ((float)correct_count / (float)idx) * 100.0f;
    float batch_avg_time = (float)total_time / (float)idx;
    
    printf("\n[BATCH %lu SUMMARY]\n", batch_num);
    printf("  Images:     %lu\n", idx);
    printf("  Correct:    %d/%lu (%.1f%%)\n", correct_count, idx, (double)batch_accuracy);
    printf("  Avg Time:   %.2f us (%.3f ms)\n", (double)batch_avg_time, (double)batch_avg_time / 1000.0);
    printf("  Min/Max:    %lu / %lu us\n", min_time, max_time);
    printf("------------------------------------------------------------\n");
    
    // Print cumulative progress
    float cumulative_accuracy = ((float)cumulative_correct / (float)total_images_processed) * 100.0f;
    printf("[CUMULATIVE PROGRESS]\n");
    printf("  Batches:    %lu/%d\n", total_batches_processed, TOTAL_BATCHES);
    printf("  Images:     %lu/1000\n", total_images_processed);
    printf("  Accuracy:   %lu/%lu (%.1f%%)\n", cumulative_correct, total_images_processed, (double)cumulative_accuracy);
    printf("  Avg Time:   %.2f us\n", (double)((float)cumulative_time_us / (float)total_images_processed));
    printf("============================================================\n");
}

/**
 * Print final consolidated statistics after all batches complete
 */
void print_final_statistics(void)
{
    printf("\n\n");
    printf("############################################################\n");
    printf("##                                                        ##\n");
    printf("##      FINAL BENCHMARK REPORT - 1000 Images             ##\n");
    printf("##                 SimpleNet CIFAR-100                    ##\n");
    printf("##                                                        ##\n");
    printf("############################################################\n");
    printf("\n");
    
    // Calculate aggregate statistics
    float avg_time = (float)cumulative_time_us / (float)total_images_processed;
    float fps = 1000000.0f / avg_time;
    float accuracy = ((float)cumulative_correct / (float)total_images_processed) * 100.0f;
    float avg_confidence = cumulative_confidence / (float)total_images_processed;
    
    // Model constants
    const uint32_t total_ops = 18607744;
    const uint32_t total_macs = 18461184;
    const uint32_t weight_bytes = 381792;
    const uint32_t bias_bytes = 1220;
    const uint32_t input_bytes = 3072;
    const uint32_t output_bytes = 400;
    
    // NPU specs
    const float npu_freq_mhz = 50.0f;
    const uint32_t num_processors = 64;
    const uint32_t ops_per_cycle = 18;
    const float peak_tops = (npu_freq_mhz * 1e6f * num_processors * ops_per_cycle) / 1e12f;
    
    // Latency calculations
    float core_latency_us = avg_time;
    float load_time_us = (float)(measured_load_input_time_us + measured_init_time_us + measured_configure_time_us);
    float unload_time_us = (float)(measured_unload_time_us + measured_softmax_time_us);
    float system_latency_us = core_latency_us + load_time_us + unload_time_us;
    
    // TOPS calculations
    float effective_tops = (float)total_ops / (core_latency_us / 1e6f) / 1e12f;
    float effective_macs_tops = (float)total_macs / (core_latency_us / 1e6f) / 1e12f;
    float utilization_pct = (effective_tops / peak_tops) * 100.0f;
    
    // Memory bandwidth
    uint32_t data_moved_bytes = input_bytes + output_bytes;
    float bandwidth_mb_per_sec = ((float)data_moved_bytes * fps) / (1024.0f * 1024.0f);
    float ops_per_byte = (float)total_ops / (float)data_moved_bytes;
    float arithmetic_intensity = (float)total_macs / (float)(weight_bytes + input_bytes);
    
    // Total time in seconds
    float total_time_sec = (float)cumulative_time_us / 1e6f;
    
    printf("[BASIC STATISTICS]\n");
    printf("  Total Batches:       %lu/%d\n", total_batches_processed, TOTAL_BATCHES);
    printf("  Total Images:        %lu\n", total_images_processed);
    printf("  Correct:             %lu/%lu (%.2f%%)\n", cumulative_correct, total_images_processed, (double)accuracy);
    printf("  Avg Confidence:      %.1f%%\n", (double)avg_confidence);
    printf("------------------------------------------------------------\n");
    
    printf("[LATENCY METRICS]\n");
    printf("  Core Latency (NPU):  %.2f us (%.3f ms)\n", (double)core_latency_us, (double)core_latency_us / 1000.0);
    printf("  System Latency:      %.2f us (%.3f ms)\n", (double)system_latency_us, (double)system_latency_us / 1000.0);
    printf("  Min Latency:         %lu us\n", cumulative_min_time);
    printf("  Max Latency:         %lu us\n", cumulative_max_time);
    printf("  Latency Jitter:      %lu us\n", cumulative_max_time - cumulative_min_time);
    printf("  --- Measured Overhead ---\n");
    printf("  Load Weights:        %lu us (one-time)\n", measured_load_weights_time_us);
    printf("  Load Bias:           %lu us (one-time)\n", measured_load_bias_time_us);
    printf("  Load Input:          %lu us (per inference)\n", measured_load_input_time_us);
    printf("  CNN Init:            %lu us (per inference)\n", measured_init_time_us);
    printf("  CNN Configure:       %lu us (per inference)\n", measured_configure_time_us);
    printf("  CNN Unload:          %lu us (per inference)\n", measured_unload_time_us);
    printf("  Softmax:             %lu us (per inference)\n", measured_softmax_time_us);
    printf("  Total Overhead:      %.2f us (per inference)\n", (double)(load_time_us + unload_time_us));
    printf("------------------------------------------------------------\n");
    
    printf("[THROUGHPUT METRICS]\n");
    printf("  Throughput:          %.2f FPS\n", (double)fps);
    printf("  Total Time:          %.2f seconds (%.2f minutes)\n", (double)total_time_sec, (double)total_time_sec / 60.0);
    printf("  Time per Image:      %.2f ms\n", (double)avg_time / 1000.0);
    printf("  Images per Second:   %.2f\n", (double)fps);
    printf("------------------------------------------------------------\n");
    
    printf("[TOPS ANALYSIS]\n");
    printf("  Peak TOPS:           %.4f TOPS (theoretical max)\n", (double)peak_tops);
    printf("  Effective TOPS:      %.4f TOPS (actual)\n", (double)effective_tops);
    printf("  Effective MACs:      %.4f TMACs/s\n", (double)effective_macs_tops);
    printf("  NPU Utilization:     %.1f%%\n", (double)utilization_pct);
    printf("  Ops/Inference:       %.2f M ops\n", (double)total_ops / 1e6);
    printf("  MACs/Inference:      %.2f M MACs\n", (double)total_macs / 1e6);
    printf("  Total Ops (1000):    %.2f G ops\n", (double)(total_ops * total_images_processed) / 1e9);
    printf("  Total MACs (1000):   %.2f G MACs\n", (double)(total_macs * total_images_processed) / 1e9);
    printf("------------------------------------------------------------\n");
    
    printf("[MEMORY EFFICIENCY]\n");
    printf("  Weight Memory:       %lu bytes (%.1f KB)\n", weight_bytes, (double)weight_bytes / 1024.0);
    printf("  Bias Memory:         %lu bytes\n", bias_bytes);
    printf("  Input Size:          %lu bytes\n", input_bytes);
    printf("  Output Size:         %lu bytes\n", output_bytes);
    printf("  Data Movement:       %lu bytes/inference\n", data_moved_bytes);
    printf("  Total Data Moved:    %.2f MB (1000 images)\n", 
           (double)(data_moved_bytes * total_images_processed) / (1024.0 * 1024.0));
    printf("  Bandwidth Used:      %.2f MB/s\n", (double)bandwidth_mb_per_sec);
    printf("  Arithmetic Intensity:%.1f ops/byte\n", (double)arithmetic_intensity);
    printf("  Ops/Byte Moved:      %.1f ops/byte\n", (double)ops_per_byte);
    printf("------------------------------------------------------------\n");
    
    printf("[ENERGY ESTIMATION]\n");
    printf("  Avg Power (NPU):     ~100 mW (estimated)\n");
    printf("  Energy per Image:    ~%.2f mJ\n", (double)(0.1 * avg_time / 1000.0));
    printf("  Total Energy (1000): ~%.2f J\n", (double)(0.1 * total_time_sec));
    printf("  Energy Efficiency:   %.2f TOPS/W (estimated)\n", (double)(effective_tops / 0.1));
    printf("------------------------------------------------------------\n");
    
    printf("[VERDICT]\n");
    if (utilization_pct >= 40.0f) {
        printf("  NPU Efficiency:      ✓ EXCELLENT (>=40%%)\n");
    } else if (utilization_pct >= 20.0f) {
        printf("  NPU Efficiency:      ✓ GOOD (20-40%%)\n");
    } else {
        printf("  NPU Efficiency:      ⚠ MODERATE (<20%%)\n");
    }
    
    if (accuracy >= 99.0f) {
        printf("  Accuracy:            ✓ EXCELLENT (>=99%%)\n");
    } else if (accuracy >= 95.0f) {
        printf("  Accuracy:            ✓ GOOD (>=95%%)\n");
    } else if (accuracy >= 90.0f) {
        printf("  Accuracy:            ✓ ACCEPTABLE (>=90%%)\n");
    } else {
        printf("  Accuracy:            ⚠ NEEDS IMPROVEMENT (<90%%)\n");
    }
    
    if (fps >= 30.0f) {
        printf("  Throughput:          ✓ REAL-TIME CAPABLE (>=30 FPS)\n");
    } else if (fps >= 10.0f) {
        printf("  Throughput:          ✓ GOOD (>=10 FPS)\n");
    } else {
        printf("  Throughput:          ⚠ LIMITED (<10 FPS)\n");
    }
    
    printf("############################################################\n");
    printf("##                  BENCHMARK COMPLETE                    ##\n");
    printf("############################################################\n\n");
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
    MXC_TMR_SW_Start(CNN_INFERENCE_TIMER);
    cnn_load_weights();
    measured_load_weights_time_us = MXC_TMR_SW_Stop(CNN_INFERENCE_TIMER);
    MXC_TMR_SW_Start(CNN_INFERENCE_TIMER);
    cnn_load_bias();
    measured_load_bias_time_us = MXC_TMR_SW_Stop(CNN_INFERENCE_TIMER);
    cnn_configure();
    printf(" OK\n");
    printf("[INIT] Load Weights Time: %lu us\n", measured_load_weights_time_us);
    printf("[INIT] Load Bias Time: %lu us\n", measured_load_bias_time_us);
    
    // Print NPU frequency
    uint32_t pclk_freq = SystemCoreClock / 2;  // APB clock is typically half of system clock
    printf("[INIT] NPU Freq: %lu MHz (PCLK)\n", pclk_freq / 1000000);
    printf("============================================================\n\n");
    
    LED_On(0);  // Ready indicator
    
    // Main benchmark loop
    while (1) {
        // Reset cumulative statistics for new benchmark run
        total_batches_processed = 0;
        total_images_processed = 0;
        cumulative_correct = 0;
        cumulative_time_us = 0;
        cumulative_min_time = 0xFFFFFFFF;
        cumulative_max_time = 0;
        cumulative_confidence = 0.0f;
        warmup_done = 0;
        
        printf("\n");
        printf("============================================================\n");
        printf("[START] New Benchmark Session - 1000 Images (50 Batches)\n");
        printf("============================================================\n\n");
        
        // Process all 50 batches
        for (uint32_t batch = 1; batch <= TOTAL_BATCHES; batch++) {
            // Reset queue for new batch
            queue_head = 0;
            queue_tail = 0;
            queue_count = 0;
            
            // Only print wait message for first batch
            if (batch == 1) {
                printf("[WAIT] Waiting for benchmark data...\n");
            } else {
                printf("[WAIT] Waiting for batch %lu/%d...\n", batch, TOTAL_BATCHES);
            }
            
            // Wait for batch start command
            error = wait_for_benchmark_start(&num_images);
            if (error != E_NO_ERROR) {
                printf("[ERROR] Failed to receive batch %lu: %d\n", batch, error);
                break;
            }
            
            if (num_images > MAX_QUEUE_SIZE) {
                printf("[WARN] Batch %lu: Requested %lu images, limiting to %d\n", 
                       batch, num_images, MAX_QUEUE_SIZE);
                num_images = MAX_QUEUE_SIZE;
            }
            
            printf("[BATCH %lu/%d] Receiving %lu images...\n", batch, TOTAL_BATCHES, num_images);
            
            // Receive all images into queue
            LED_On(1);
            uint32_t received = receive_all_images(num_images);
            LED_Off(1);
            
            if (received == 0) {
                printf("[ERROR] No images received in batch %lu!\n", batch);
                break;
            }
            
            // Run warmup only once on first batch
            if (!warmup_done && batch == 1) {
                printf("[WARMUP] Running warmup (one-time)...\n");
                run_warmup(WARMUP_ITERATIONS);
                warmup_done = 1;
            }
            
            // Run benchmark for this batch
            LED_On(1);
            run_benchmark(received, batch);
            LED_Off(1);
            
            printf("\n[BATCH %lu/%d COMPLETE]\n", batch, TOTAL_BATCHES);
            
            // Clear UART for next batch
            MXC_UART_ClearRXFIFO(RX_UART);
            MXC_Delay(MXC_DELAY_MSEC(50));
        }
        
        // Print final consolidated report
        if (total_batches_processed == TOTAL_BATCHES) {
            print_final_statistics();
        } else {
            printf("\n[INCOMPLETE] Only %lu/%d batches completed.\n", 
                   total_batches_processed, TOTAL_BATCHES);
        }
        
        printf("\n[DONE] Benchmark session complete.\n\n");
        
        // Clear UART and wait for next session
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


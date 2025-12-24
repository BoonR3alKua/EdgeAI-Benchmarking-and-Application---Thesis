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

// my_cifar100_effnetv2 - Queue-based UART Benchmark System
// CIFAR-100 classification with EfficientNetV2 on MAX78000
//
// Memory layout for 32x32 RGB image (CHW format):
//   Channel 0 (R): 0x50400000 - 256 words (1024 bytes)
//   Channel 1 (G): 0x50800000 - 256 words (1024 bytes)
//   Channel 2 (B): 0x50c00000 - 256 words (1024 bytes)
//
// Protocol: [DEADBEEF][NUM_IMAGES(4 bytes)][Label0][Image0][Label1][Image1]...

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include "mxc.h"
#include "cnn.h"
#include "led.h"

// ============================================================================
// Configuration
// ============================================================================
#define UART_BAUD           115200
#define RX_UART             MXC_UART0
#define START_MARKER        0xDEADBEEF
#define IMAGE_SIZE          3072    // 32x32x3 bytes
#define CHANNEL_SIZE        1024    // 32x32 bytes per channel
#define CHANNEL_WORDS       256     // 1024 bytes / 4
#define MAX_QUEUE_SIZE      20      // Maximum images in queue (limited by SRAM)
#define TOTAL_BATCHES       50      // Total number of batches for 1000 images
#define WARMUP_ITERATIONS   5       // Warmup runs before benchmark
#define TIMEOUT_LOOPS       100000000  // Receive timeout (loop count)

// ============================================================================
// Data Structures
// ============================================================================
typedef struct {
    uint32_t ch0[CHANNEL_WORDS];    // Channel 0 (R) - 256 words
    uint32_t ch1[CHANNEL_WORDS];    // Channel 1 (G) - 256 words
    uint32_t ch2[CHANNEL_WORDS];    // Channel 2 (B) - 256 words
    uint8_t label;                  // Ground truth label
} ImageQueueEntry;

typedef struct {
    uint32_t latency_us;
    int predicted_class;
    uint8_t true_label;
    int correct;
    float confidence;
} BenchmarkResult;

// ============================================================================
// Global Variables
// ============================================================================
volatile uint32_t cnn_time;
static ImageQueueEntry image_queue[MAX_QUEUE_SIZE];
static volatile uint32_t queue_head = 0;
static volatile uint32_t queue_tail = 0;
static volatile uint32_t queue_count = 0;
static BenchmarkResult results[MAX_QUEUE_SIZE];
static int32_t ml_data[CNN_NUM_OUTPUTS];
static q15_t ml_softmax[CNN_NUM_OUTPUTS];

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

// ============================================================================
// Queue Functions
// ============================================================================
static inline int queue_push(ImageQueueEntry *entry) {
    if (queue_count >= MAX_QUEUE_SIZE) return -1;
    memcpy(&image_queue[queue_tail], entry, sizeof(ImageQueueEntry));
    queue_tail = (queue_tail + 1) % MAX_QUEUE_SIZE;
    queue_count++;
    return 0;
}

static inline ImageQueueEntry* queue_peek(uint32_t index) {
    if (index >= queue_count) return NULL;
    return &image_queue[(queue_head + index) % MAX_QUEUE_SIZE];
}

static inline void queue_pop(void) {
    if (queue_count == 0) return;
    queue_head = (queue_head + 1) % MAX_QUEUE_SIZE;
    queue_count--;
}

// ============================================================================
// UART Functions
// ============================================================================
static int uart_read_byte_timeout(uint8_t *byte, uint32_t timeout_loops) {
    uint32_t loops = 0;
    while (MXC_UART_GetRXFIFOAvailable(RX_UART) == 0) {
        if (++loops > timeout_loops) return -1;
    }
    *byte = MXC_UART_ReadCharacter(RX_UART);
    return 0;
}

static uint8_t uart_read_byte_blocking(void) {
    while (MXC_UART_GetRXFIFOAvailable(RX_UART) == 0);
    return MXC_UART_ReadCharacter(RX_UART);
}

static int uart_read_bytes(uint8_t *buffer, uint32_t length, uint32_t timeout_loops) {
    for (uint32_t i = 0; i < length; i++) {
        if (uart_read_byte_timeout(&buffer[i], timeout_loops) != 0) return i;
    }
    return length;
}

// ============================================================================
// Protocol Functions
// ============================================================================
static int wait_for_benchmark_start(uint32_t *num_images) {
    uint8_t marker_buf[4];
    uint8_t count_buf[4];
    uint32_t marker;
    
    while (1) {
        if (uart_read_byte_timeout(&marker_buf[0], TIMEOUT_LOOPS) != 0) continue;
        if (marker_buf[0] != 0xDE) continue;
        
        if (uart_read_bytes(&marker_buf[1], 3, TIMEOUT_LOOPS) != 3) continue;
        
        marker = ((uint32_t)marker_buf[0] << 24) | ((uint32_t)marker_buf[1] << 16) |
                 ((uint32_t)marker_buf[2] << 8)  | ((uint32_t)marker_buf[3]);
        
        if (marker == START_MARKER) {
            if (uart_read_bytes(count_buf, 4, TIMEOUT_LOOPS) != 4) {
                printf("[ERROR] Failed to read image count\n");
                return -1;
            }
            *num_images = ((uint32_t)count_buf[0] << 24) | ((uint32_t)count_buf[1] << 16) |
                          ((uint32_t)count_buf[2] << 8)  | ((uint32_t)count_buf[3]);
            printf("[INFO] Start marker received. Images: %lu\n", *num_images);
            return E_NO_ERROR;
        }
    }
}

static int receive_single_image(ImageQueueEntry *entry) {
    uint8_t label_byte;
    uint8_t image_data[IMAGE_SIZE];
    
    if (uart_read_byte_timeout(&label_byte, TIMEOUT_LOOPS) != 0) return -1;
    entry->label = label_byte;
    
    if (uart_read_bytes(image_data, IMAGE_SIZE, TIMEOUT_LOOPS) != IMAGE_SIZE) return -2;
    
    // Parse CHW format into separate channels (pack 4 bytes into each uint32_t)
    for (int i = 0; i < CHANNEL_WORDS; i++) {
        entry->ch0[i] = ((uint32_t)image_data[i*4 + 0]) |
                        ((uint32_t)image_data[i*4 + 1] << 8) |
                        ((uint32_t)image_data[i*4 + 2] << 16) |
                        ((uint32_t)image_data[i*4 + 3] << 24);
        
        entry->ch1[i] = ((uint32_t)image_data[CHANNEL_SIZE + i*4 + 0]) |
                        ((uint32_t)image_data[CHANNEL_SIZE + i*4 + 1] << 8) |
                        ((uint32_t)image_data[CHANNEL_SIZE + i*4 + 2] << 16) |
                        ((uint32_t)image_data[CHANNEL_SIZE + i*4 + 3] << 24);
        
        entry->ch2[i] = ((uint32_t)image_data[2*CHANNEL_SIZE + i*4 + 0]) |
                        ((uint32_t)image_data[2*CHANNEL_SIZE + i*4 + 1] << 8) |
                        ((uint32_t)image_data[2*CHANNEL_SIZE + i*4 + 2] << 16) |
                        ((uint32_t)image_data[2*CHANNEL_SIZE + i*4 + 3] << 24);
    }
    
    return E_NO_ERROR;
}

static uint32_t receive_all_images(uint32_t num_images) {
    ImageQueueEntry entry;
    uint32_t received = 0;
    
    // Send READY signal to sender - must flush before starting to receive
    printf("READY\n");
    fflush(stdout);
    
    for (uint32_t i = 0; i < num_images; i++) {
        if (receive_single_image(&entry) != E_NO_ERROR) {
            printf("[RX] Failed at image %lu\n", i);
            break;
        }
        if (queue_push(&entry) != 0) {
            printf("[RX] Queue full at image %lu\n", i);
            break;
        }
        received++;
    }
    
    printf("[RECV] Received %lu images into queue.\n", received);
    return received;
}

// ============================================================================
// CNN Functions
// ============================================================================
static void load_image_to_cnn(ImageQueueEntry *img) {
    memcpy32((uint32_t *)0x50400000, img->ch0, CHANNEL_WORDS);
    memcpy32((uint32_t *)0x50800000, img->ch1, CHANNEL_WORDS);
    memcpy32((uint32_t *)0x50c00000, img->ch2, CHANNEL_WORDS);
}

/**
 * Measure load/unload times using a calibration run
 */
static void measure_overhead_times(ImageQueueEntry *img)
{
    // Measure load_input time
    MXC_TMR_SW_Start(CNN_INFERENCE_TIMER);
    load_image_to_cnn(img);
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
    while (cnn_time == 0) { __WFI(); }
    
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

static int run_cnn_inference(float *confidence) {
    cnn_init();
    cnn_configure();
    cnn_start();
    while (cnn_time == 0) { __WFI(); }
    cnn_unload((uint32_t *)ml_data);
    softmax_q17p14_q15((const q31_t *)ml_data, CNN_NUM_OUTPUTS, ml_softmax);
    
    int max_idx = 0;
    int32_t max_val = ml_data[0];
    for (int i = 1; i < CNN_NUM_OUTPUTS; i++) {
        if (ml_data[i] > max_val) { max_val = ml_data[i]; max_idx = i; }
    }
    
    // Calculate confidence as percentage (ml_softmax is Q15 format: 0x7FFF = 100%)
    if (confidence != NULL) {
        *confidence = (float)ml_softmax[max_idx] / 32768.0f * 100.0f;
    }
    
    return max_idx;
}

// ============================================================================
// Benchmark Functions
// ============================================================================
static void run_warmup(int iterations) {
    ImageQueueEntry *first_img = queue_peek(0);
    if (first_img == NULL) { printf("[WARMUP] No images in queue!\n"); return; }
    
    // Measure overhead times before warmup
    printf("[WARMUP] Measuring overhead times...\n");
    measure_overhead_times(first_img);
    
    printf("[WARMUP] Running %d iterations...\n", iterations);
    
    load_image_to_cnn(first_img);
    
    for (int i = 0; i < iterations; i++) {
        cnn_time = 0;
        run_cnn_inference(NULL);
    }
    
    printf("[WARMUP] %d iterations done.\n", iterations);
}

static void run_benchmark(uint32_t num_images, uint32_t batch_num) {
    uint32_t total_time_us = 0;
    uint32_t min_time = 0xFFFFFFFF, max_time = 0;
    uint32_t correct_count = 0;
    float total_confidence = 0.0f;
    
    printf("[BENCH] Batch %lu/%d - Start Loop (N=%lu)\n", batch_num, TOTAL_BATCHES, num_images);
    
    for (uint32_t i = 0; i < num_images; i++) {
        ImageQueueEntry *img = queue_peek(i);
        if (img == NULL) break;
        
        load_image_to_cnn(img);
        cnn_time = 0;
        
        float confidence;
        int predicted = run_cnn_inference(&confidence);
        uint32_t latency = cnn_time;
        
        if (i < MAX_QUEUE_SIZE) {
            results[i].latency_us = latency;
            results[i].predicted_class = predicted;
            results[i].true_label = img->label;
            results[i].correct = (predicted == img->label);
            results[i].confidence = confidence;
        }
        
        total_time_us += latency;
        total_confidence += confidence;
        if (latency < min_time) min_time = latency;
        if (latency > max_time) max_time = latency;
        if (predicted == img->label) correct_count++;
        
        uint32_t global_idx = total_images_processed + i + 1;
        printf("Img %04lu: %.1fms | Class: %d | GT: %d | Conf: %.1f%% | %s\n",
               global_idx,
               (double)latency / 1000.0,
               predicted, img->label,
               (double)confidence,
               (predicted == img->label) ? "OK" : "FAIL");
    }
    
    printf("[BENCH] Batch %lu/%d - Loop End.\n", batch_num, TOTAL_BATCHES);
    
    // Update cumulative statistics
    total_batches_processed++;
    total_images_processed += num_images;
    cumulative_correct += correct_count;
    cumulative_time_us += total_time_us;
    cumulative_confidence += total_confidence;
    if (min_time < cumulative_min_time) cumulative_min_time = min_time;
    if (max_time > cumulative_max_time) cumulative_max_time = max_time;
    
    // Print batch summary
    float batch_accuracy = ((float)correct_count / (float)num_images) * 100.0f;
    float batch_avg_time = (float)total_time_us / (float)num_images;
    
    printf("\n[BATCH %lu SUMMARY]\n", batch_num);
    printf("  Images:     %lu\n", num_images);
    printf("  Correct:    %lu/%lu (%.1f%%)\n", correct_count, num_images, (double)batch_accuracy);
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
static void print_final_statistics(void)
{
    printf("\n\n");
    printf("############################################################\n");
    printf("##                                                        ##\n");
    printf("##      FINAL BENCHMARK REPORT - 1000 Images             ##\n");
    printf("##              EfficientNetV2 CIFAR-100                  ##\n");
    printf("##                                                        ##\n");
    printf("############################################################\n");
    printf("\n");
    
    // Calculate aggregate statistics
    float avg_time = (float)cumulative_time_us / (float)total_images_processed;
    float fps = 1000000.0f / avg_time;
    float accuracy = ((float)cumulative_correct / (float)total_images_processed) * 100.0f;
    float avg_confidence = cumulative_confidence / (float)total_images_processed;
    
    // Model constants
    const uint32_t total_ops = 14893568;
    const uint32_t total_macs = 14733824;
    const uint32_t weight_bytes = 280136;
    const uint32_t bias_bytes = 948;
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

// ============================================================================
// Main
// ============================================================================
int main(void) {
    int error;
    uint32_t num_images;
    
    MXC_ICC_Enable(MXC_ICC0);
    MXC_SYS_Clock_Select(MXC_SYS_CLOCK_IPO);
    SystemCoreClockUpdate();
    
    printf("\n\n============================================================\n");
    printf("[INIT] CIFAR-100 EfficientNetV2 Benchmark System\n");
    printf("============================================================\n");
    printf("[INIT] Image Size: 32x32x3 (CHW format)\n");
    printf("[INIT] Queue Size: %d images max\n", MAX_QUEUE_SIZE);
    printf("[INIT] UART: %d baud\n", UART_BAUD);
    
    printf("[INIT] Initializing UART...");
    error = MXC_UART_Init(RX_UART, UART_BAUD, MXC_UART_APB_CLK);
    if (error < E_NO_ERROR) { printf(" FAILED (%d)\n", error); return error; }
    printf(" OK\n");
    
    printf("[INIT] Loading EfficientNetV2...");
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
    
    uint32_t pclk_freq = SystemCoreClock / 2;
    printf("[INIT] NPU Freq: %lu MHz (PCLK)\n", pclk_freq / 1000000);
    printf("============================================================\n\n");
    
    LED_On(0);
    
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
                printf("[INFO] Waiting for start marker (0xDEADBEEF)...\n");
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
                printf("[WARN] Limiting to %d images\n", MAX_QUEUE_SIZE);
                num_images = MAX_QUEUE_SIZE;
            }
            
            printf("[BATCH %lu/%d] Receiving %lu images...\n", batch, TOTAL_BATCHES, num_images);
            
            // Receive all images into queue
            LED_On(1);
            uint32_t received = receive_all_images(num_images);
            LED_Off(1);
            
            if (received == 0) {
                printf("[ERROR] No images received!\n");
                break;
            }
            
            // Run warmup only once on first batch
            if (!warmup_done && batch == 1) {
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
  Hardware: 14,893,568 ops (14,733,824 macc; 132,096 comp; 27,648 add; 0 mul; 0 bitwise)
    Layer 0 (stem): 688,128 ops (663,552 macc; 24,576 comp; 0 add; 0 mul; 0 bitwise)
    Layer 1 (stage1_conv): 5,332,992 ops (5,308,416 macc; 24,576 comp; 0 add; 0 mul; 0 bitwise)
    Layer 2 (stage1_short): 0 ops (0 macc; 0 comp; 0 add; 0 mul; 0 bitwise)
    Layer 3 (stage1_add): 24,576 ops (0 macc; 0 comp; 24,576 add; 0 mul; 0 bitwise)
    Layer 4 (pool1): 24,576 ops (0 macc; 24,576 comp; 0 add; 0 mul; 0 bitwise)
    Layer 5 (stage2_fused): 2,666,496 ops (2,654,208 macc; 12,288 comp; 0 add; 0 mul; 0 bitwise)
    Layer 6 (stage2_proj): 602,112 ops (589,824 macc; 12,288 comp; 0 add; 0 mul; 0 bitwise)
    Layer 7 (pool2): 12,288 ops (0 macc; 12,288 comp; 0 add; 0 mul; 0 bitwise)
    Layer 8 (stage3_fused): 1,773,568 ops (1,769,472 macc; 4,096 comp; 0 add; 0 mul; 0 bitwise)
    Layer 9 (stage3_proj): 266,240 ops (262,144 macc; 4,096 comp; 0 add; 0 mul; 0 bitwise)
    Layer 10 (pool3): 4,096 ops (0 macc; 4,096 comp; 0 add; 0 mul; 0 bitwise)
    Layer 11 (stage4_expand): 66,560 ops (65,536 macc; 1,024 comp; 0 add; 0 mul; 0 bitwise)
    Layer 12 (stage4_spatial): 590,848 ops (589,824 macc; 1,024 comp; 0 add; 0 mul; 0 bitwise)
    Layer 13 (stage4_proj): 66,560 ops (65,536 macc; 1,024 comp; 0 add; 0 mul; 0 bitwise)
    Layer 14 (stage4_short): 0 ops (0 macc; 0 comp; 0 add; 0 mul; 0 bitwise)
    Layer 15 (stage4_add): 1,024 ops (0 macc; 0 comp; 1,024 add; 0 mul; 0 bitwise)
    Layer 16 (stage5_expand): 133,120 ops (131,072 macc; 2,048 comp; 0 add; 0 mul; 0 bitwise)
    Layer 17 (stage5_spatial): 2,361,344 ops (2,359,296 macc; 2,048 comp; 0 add; 0 mul; 0 bitwise)
    Layer 18 (stage5_proj): 264,192 ops (262,144 macc; 2,048 comp; 0 add; 0 mul; 0 bitwise)
    Layer 19 (avg_pool): 2,048 ops (0 macc; 0 comp; 2,048 add; 0 mul; 0 bitwise)
    Layer 20 (fc): 12,800 ops (12,800 macc; 0 comp; 0 add; 0 mul; 0 bitwise)

  RESOURCE USAGE
  Weight memory: 280,136 bytes out of 442,368 bytes total (63.3%)
  Bias memory:   948 bytes out of 2,048 bytes total (46.3%)
*/
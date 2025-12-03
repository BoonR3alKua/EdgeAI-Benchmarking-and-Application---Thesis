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

// my_cifar100_mobilenetv2 - Queue-based Benchmark Version
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
#define RX_UART MXC_UART0

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
static volatile uint32_t queue_head = 0;
static volatile uint32_t queue_tail = 0;
static volatile uint32_t queue_count = 0;

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

static inline int queue_is_full(void) { return queue_count >= MAX_QUEUE_SIZE; }
static inline int queue_is_empty(void) { return queue_count == 0; }

static int queue_push(uint8_t *image_data, uint8_t label) {
    if (queue_is_full()) return -1;
    
    ImageQueueEntry *entry = &image_queue[queue_head];
    memcpy(entry->input_ch0, image_data, BYTES_PER_CHANNEL);
    memcpy(entry->input_ch1, image_data + BYTES_PER_CHANNEL, BYTES_PER_CHANNEL);
    memcpy(entry->input_ch2, image_data + 2 * BYTES_PER_CHANNEL, BYTES_PER_CHANNEL);
    entry->label = label;
    entry->valid = 1;
    
    queue_head = (queue_head + 1) % MAX_QUEUE_SIZE;
    queue_count++;
    return 0;
}

static ImageQueueEntry* queue_peek(void) {
    if (queue_is_empty()) return NULL;
    return &image_queue[queue_tail];
}

static void queue_pop(void) {
    if (!queue_is_empty()) {
        image_queue[queue_tail].valid = 0;
        queue_tail = (queue_tail + 1) % MAX_QUEUE_SIZE;
        queue_count--;
    }
}

int uart_read_byte_timeout(mxc_uart_regs_t *uart, uint32_t timeout_loops) {
    uint32_t loops = 0;
    int byte_read;
    while (loops < timeout_loops) {
        byte_read = MXC_UART_ReadCharacter(uart);
        if (byte_read >= 0) return byte_read;
        loops++;
    }
    return E_TIME_OUT;
}

int uart_read_byte_blocking(mxc_uart_regs_t *uart) {
    int byte_read;
    while ((byte_read = MXC_UART_ReadCharacter(uart)) < 0);
    return byte_read;
}

int wait_for_benchmark_start(uint32_t *num_images) {
    int byte_read;
    int state = 0;
    uint32_t timeout_loops = 100000000;
    
    printf("[WAIT] Waiting for benchmark data...\n");
    
    while (1) {
        byte_read = uart_read_byte_timeout(RX_UART, timeout_loops);
        if (byte_read < 0) return byte_read;
        
        switch (state) {
            case 0: if (byte_read == START_MARKER_1) state = 1; break;
            case 1: state = (byte_read == START_MARKER_2) ? 2 : (byte_read == START_MARKER_1) ? 1 : 0; break;
            case 2: state = (byte_read == START_MARKER_3) ? 3 : (byte_read == START_MARKER_1) ? 1 : 0; break;
            case 3:
                if (byte_read == START_MARKER_4) {
                    uint8_t num_bytes[4];
                    for (int i = 0; i < 4; i++) num_bytes[i] = uart_read_byte_blocking(RX_UART);
                    *num_images = ((uint32_t)num_bytes[0]) | ((uint32_t)num_bytes[1] << 8) |
                                  ((uint32_t)num_bytes[2] << 16) | ((uint32_t)num_bytes[3] << 24);
                    return E_NO_ERROR;
                }
                state = (byte_read == START_MARKER_1) ? 1 : 0;
                break;
        }
    }
}

int receive_all_images(uint32_t num_images) {
    uint32_t received = 0;
    
    // Send READY signal to sender - must flush before starting to receive
    printf("READY\n");
    fflush(stdout);
    
    while (received < num_images && !queue_is_full()) {
        uint8_t label = uart_read_byte_blocking(RX_UART);
        for (int i = 0; i < BYTES_PER_IMAGE; i++) rx_buffer[i] = uart_read_byte_blocking(RX_UART);
        if (queue_push(rx_buffer, label) == 0) received++;
        else { printf("[ERROR] Queue full at image %lu\n", received); break; }
    }
    
    printf("[RECV] Received %lu images into queue.\n", received);
    return received;
}

void load_input_from_queue(ImageQueueEntry *entry) {
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

int classify_image(float *confidence) {
    int i, max_idx = 0;
    int32_t max_val;
    
    cnn_time = 0;
    cnn_start();
    SCB->SCR &= ~SCB_SCR_SLEEPDEEP_Msk;
    while (cnn_time == 0) asm volatile("wfi");
    
    cnn_unload((uint32_t *)ml_data);
    softmax_q17p14_q15((const q31_t *)ml_data, CNN_NUM_OUTPUTS, ml_softmax);
    
    max_val = ml_data[0];
    for (i = 1; i < CNN_NUM_OUTPUTS; i++) {
        if (ml_data[i] > max_val) { max_val = ml_data[i]; max_idx = i; }
    }
    
    // Calculate confidence as percentage (ml_softmax is Q15 format: 0x7FFF = 100%)
    if (confidence != NULL) {
        *confidence = (float)ml_softmax[max_idx] / 32768.0f * 100.0f;
    }
    
    return max_idx;
}

void run_warmup(int iterations) {
    ImageQueueEntry *entry = queue_peek();
    if (entry == NULL) { printf("[WARMUP] No images in queue!\n"); return; }
    
    // Measure overhead times before warmup
    printf("[WARMUP] Measuring overhead times...\n");
    measure_overhead_times(entry);
    
    printf("[WARMUP] Running %d iterations...\n", iterations);
    for (int i = 0; i < iterations; i++) {
        load_input_from_queue(entry);
        cnn_init();
        cnn_configure();
        classify_image(NULL);
    }
    printf("[WARMUP] %d iterations done.\n", iterations);
}

void run_benchmark(uint32_t num_images) {
    uint32_t total_time = 0, min_time = 0xFFFFFFFF, max_time = 0;
    int correct_count = 0;
    float total_confidence = 0.0f;
    uint32_t idx = 0;
    
    printf("[BENCH] Start Loop (N=%lu)\n", num_images);
    
    while (!queue_is_empty() && idx < num_images) {
        ImageQueueEntry *entry = queue_peek();
        if (entry == NULL) break;
        
        load_input_from_queue(entry);
        cnn_init();
        cnn_configure();
        
        float confidence;
        int predicted = classify_image(&confidence);
        
        benchmark_results[idx].cnn_time_us = cnn_time;
        benchmark_results[idx].predicted_class = predicted;
        benchmark_results[idx].ground_truth = entry->label;
        benchmark_results[idx].correct = (predicted == entry->label) ? 1 : 0;
        benchmark_results[idx].confidence = confidence;
        
        total_time += cnn_time;
        total_confidence += confidence;
        if (cnn_time < min_time) min_time = cnn_time;
        if (cnn_time > max_time) max_time = cnn_time;
        if (predicted == entry->label) correct_count++;
        
        printf("Iter %02lu: %.1fms | Class: %d | GT: %d | Conf: %.1f%% | %s\n", 
               idx + 1, (double)cnn_time / 1000.0, predicted, entry->label,
               (double)confidence,
               (predicted == entry->label) ? "OK" : "FAIL");
        
        queue_pop();
        idx++;
    }
    
    printf("[BENCH] Loop End.\n");
    
    // =========================================================================
    // BENCHMARK STATISTICS CALCULATION
    // =========================================================================
    
    float avg_time = (float)total_time / (float)idx;
    float fps = 1000000.0f / avg_time;
    float accuracy = ((float)correct_count / (float)idx) * 100.0f;
    float avg_confidence = total_confidence / (float)idx;
    
    // MobileNetV2 Model constants (from synthesis output)
    const uint32_t total_ops = 5343744;       // Total operations per inference
    const uint32_t total_macs = 5284544;      // Total MACs per inference
    const uint32_t weight_bytes = 121136;     // Weight memory used (27.4%)
    const uint32_t bias_bytes = 764;          // Bias memory used (37.3%)
    const uint32_t input_bytes = 3072;
    const uint32_t output_bytes = 400;
    
    // MAX78000 NPU specs
    const float npu_freq_mhz = 50.0f;        // NPU clock frequency (Up to 50MHz)
    const uint32_t num_processors = 64;      // Number of parallel processors
    
    // CORRECTION:
    // The accelerator outputs "Nominal 1 output channel per clock"
    // For a 3x3 kernel, this requires 9 MACs per clock cycle.
    // 1 MAC = 2 Operations (Multiply + Add)
    // Therefore: 9 MACs * 2 Ops = 18 Ops per cycle.
    const uint32_t ops_per_cycle = 18;       
    
    // Peak TOPS Calculation
    // 50 MHz * 64 Processors * 18 Ops/cycle = 57,600 Million Ops = 57.6 GOPS
    const float peak_tops = (npu_freq_mhz * 1e6f * num_processors * ops_per_cycle) / 1e12f;
    // Result: ~0.0576 TOPS (or 57.6 GOPS)
    
    // Latency calculations using MEASURED values
    float core_latency_us = avg_time;
    float load_time_us = (float)(measured_load_input_time_us + measured_init_time_us + measured_configure_time_us);
    float unload_time_us = (float)(measured_unload_time_us + measured_softmax_time_us);
    float system_latency_us = core_latency_us + load_time_us + unload_time_us;
    
    float effective_tops = (float)total_ops / (core_latency_us / 1e6f) / 1e12f;
    float effective_macs_tops = (float)total_macs / (core_latency_us / 1e6f) / 1e12f;
    float utilization_pct = (effective_tops / peak_tops) * 100.0f;
    
    uint32_t data_moved_bytes = input_bytes + output_bytes;
    float bandwidth_mb_per_sec = ((float)data_moved_bytes * fps) / (1024.0f * 1024.0f);
    float ops_per_byte = (float)total_ops / (float)data_moved_bytes;
    float arithmetic_intensity = (float)total_macs / (float)(weight_bytes + input_bytes);
    
    // =========================================================================
    // PRINT REPORT
    // =========================================================================
    
    printf("\n");
    printf("============================================================\n");
    printf("          BENCHMARK REPORT - MobileNetV2                    \n");
    printf("============================================================\n");
    
    printf("[BASIC STATISTICS]\n");
    printf("  Total Images:        %lu\n", idx);
    printf("  Correct:             %d/%lu (%.1f%%)\n", correct_count, idx, (double)accuracy);
    printf("  Avg Confidence:      %.1f%%\n", (double)avg_confidence);
    printf("------------------------------------------------------------\n");
    
    printf("[LATENCY METRICS]\n");
    printf("  Core Latency (NPU):  %.2f us (%.3f ms)\n", (double)core_latency_us, (double)core_latency_us / 1000.0);
    printf("  System Latency:      %.2f us (%.3f ms)\n", (double)system_latency_us, (double)system_latency_us / 1000.0);
    printf("  Min Latency:         %lu us\n", min_time);
    printf("  Max Latency:         %lu us\n", max_time);
    printf("  Latency Jitter:      %lu us\n", max_time - min_time);
    printf("  --- Measured Overhead ---\n");
    printf("  Load Input:          %lu us\n", measured_load_input_time_us);
    printf("  CNN Init:            %lu us\n", measured_init_time_us);
    printf("  CNN Configure:       %lu us\n", measured_configure_time_us);
    printf("  CNN Unload:          %lu us\n", measured_unload_time_us);
    printf("  Softmax:             %lu us\n", measured_softmax_time_us);
    printf("  Total Overhead:      %.2f us\n", (double)(load_time_us + unload_time_us));
    printf("------------------------------------------------------------\n");
    
    printf("[THROUGHPUT METRICS]\n");
    printf("  Throughput:          %.2f FPS\n", (double)fps);
    printf("  Total Time:          %.2f ms (%lu images)\n", (double)total_time / 1000.0, idx);
    printf("------------------------------------------------------------\n");
    
    printf("[TOPS ANALYSIS]\n");
    printf("  Peak TOPS:           %.4f TOPS (theoretical max)\n", (double)peak_tops);
    printf("  Effective TOPS:      %.4f TOPS (actual)\n", (double)effective_tops);
    printf("  Effective MACs:      %.4f TMACs/s\n", (double)effective_macs_tops);
    printf("  NPU Utilization:     %.1f%%\n", (double)utilization_pct);
    printf("  Ops/Inference:       %.2f M ops\n", (double)total_ops / 1e6);
    printf("  MACs/Inference:      %.2f M MACs\n", (double)total_macs / 1e6);
    printf("------------------------------------------------------------\n");
    
    printf("[MEMORY EFFICIENCY]\n");
    printf("  Weight Memory:       %lu bytes (%.1f KB)\n", weight_bytes, (double)weight_bytes / 1024.0);
    printf("  Bias Memory:         %lu bytes\n", bias_bytes);
    printf("  Input Size:          %lu bytes\n", input_bytes);
    printf("  Output Size:         %lu bytes\n", output_bytes);
    printf("  Data Movement:       %lu bytes/inference\n", data_moved_bytes);
    printf("  Bandwidth Used:      %.2f MB/s\n", (double)bandwidth_mb_per_sec);
    printf("  Arithmetic Intensity:%.1f ops/byte\n", (double)arithmetic_intensity);
    printf("  Ops/Byte Moved:      %.1f ops/byte\n", (double)ops_per_byte);
    printf("------------------------------------------------------------\n");
    
    printf("[VERDICT]\n");
    if (utilization_pct >= 40.0f) {
        printf("  NPU Efficiency:      GOOD (>=40%%)\n");
    } else if (utilization_pct >= 20.0f) {
        printf("  NPU Efficiency:      MODERATE (20-40%%)\n");
    } else {
        printf("  NPU Efficiency:      LOW (<20%%)\n");
    }
    printf("  Accuracy:            %s\n", (correct_count == (int)idx) ? "PERFECT" : "PARTIAL");
    printf("============================================================\n");
}

int main(void) {
    int error;
    uint32_t num_images;
    
    MXC_ICC_Enable(MXC_ICC0);
    MXC_SYS_Clock_Select(MXC_SYS_CLOCK_IPO);
    SystemCoreClockUpdate();
    
    printf("\n\n============================================================\n");
    printf("[INIT] CIFAR-100 MobileNetV2 Benchmark System\n");
    printf("============================================================\n");
    printf("[INIT] Image Size: 32x32x3 (CHW format)\n");
    printf("[INIT] Queue Size: %d images max\n", MAX_QUEUE_SIZE);
    printf("[INIT] UART: %d baud\n", UART_BAUD);
    
    printf("[INIT] Initializing UART...");
    error = MXC_UART_Init(RX_UART, UART_BAUD, MXC_UART_APB_CLK);
    if (error < E_NO_ERROR) { printf(" FAILED (%d)\n", error); return error; }
    printf(" OK\n");
    
    printf("[INIT] Loading MobileNetV2...");
    cnn_enable(MXC_S_GCR_PCLKDIV_CNNCLKSEL_PCLK, MXC_S_GCR_PCLKDIV_CNNCLKDIV_DIV1);
    cnn_init();
    cnn_load_weights();
    cnn_load_bias();
    cnn_configure();
    printf(" OK\n");
    
    uint32_t pclk_freq = SystemCoreClock / 2;
    printf("[INIT] NPU Freq: %lu MHz (PCLK)\n", pclk_freq / 1000000);
    printf("============================================================\n\n");
    
    LED_On(0);
    
    while (1) {
        queue_head = 0; queue_tail = 0; queue_count = 0;
        
        error = wait_for_benchmark_start(&num_images);
        if (error != E_NO_ERROR) { printf("[ERROR] Failed to receive start: %d\n", error); continue; }
        
        if (num_images > MAX_QUEUE_SIZE) {
            printf("[WARN] Limiting to %d images\n", MAX_QUEUE_SIZE);
            num_images = MAX_QUEUE_SIZE;
        }
        
        printf("[INFO] Benchmark for %lu images\n", num_images);
        
        LED_On(1);
        uint32_t received = receive_all_images(num_images);
        LED_Off(1);
        
        if (received == 0) { printf("[ERROR] No images received!\n"); continue; }
        
        printf("[ACK] Ready to process %lu images\n", received);
        MXC_UART_ClearRXFIFO(RX_UART);
        
        run_warmup(WARMUP_ITERATIONS);
        
        LED_On(1);
        run_benchmark(received);
        LED_Off(1);
        
        printf("\n[DONE] Benchmark complete. Waiting for next batch...\n\n");
        MXC_UART_ClearRXFIFO(RX_UART);
        MXC_Delay(MXC_DELAY_MSEC(100));
    }
    
    return E_NO_ERROR;
}

/*
  SUMMARY OF OPS
  Hardware: 5,343,744 ops (5,284,544 macc; 58,432 comp; 768 add; 0 mul; 0 bitwise)
    Layer 0: 458,752 ops (442,368 macc; 16,384 comp; 0 add; 0 mul; 0 bitwise)
    Layer 1: 1,187,840 ops (1,179,648 macc; 8,192 comp; 0 add; 0 mul; 0 bitwise)
    Layer 2: 73,728 ops (65,536 macc; 8,192 comp; 0 add; 0 mul; 0 bitwise)
    Layer 3: 8,192 ops (0 macc; 8,192 comp; 0 add; 0 mul; 0 bitwise)
    Layer 4: 65,536 ops (65,536 macc; 0 comp; 0 add; 0 mul; 0 bitwise)
    Layer 5: 1,179,648 ops (1,179,648 macc; 0 comp; 0 add; 0 mul; 0 bitwise)
    Layer 6: 49,152 ops (49,152 macc; 0 comp; 0 add; 0 mul; 0 bitwise)
    Layer 7: 3,072 ops (0 macc; 3,072 comp; 0 add; 0 mul; 0 bitwise)
    Layer 8: 39,936 ops (36,864 macc; 3,072 comp; 0 add; 0 mul; 0 bitwise)
    Layer 9: 665,088 ops (663,552 macc; 1,536 comp; 0 add; 0 mul; 0 bitwise)
    Layer 10: 25,600 ops (24,576 macc; 1,024 comp; 0 add; 0 mul; 0 bitwise)
    Layer 11: 1,024 ops (0 macc; 1,024 comp; 0 add; 0 mul; 0 bitwise)
    Layer 12: 17,408 ops (16,384 macc; 1,024 comp; 0 add; 0 mul; 0 bitwise)
    Layer 13: 295,424 ops (294,912 macc; 512 comp; 0 add; 0 mul; 0 bitwise)
    Layer 14: 10,560 ops (10,240 macc; 320 comp; 0 add; 0 mul; 0 bitwise)
    Layer 15: 26,880 ops (25,600 macc; 1,280 comp; 0 add; 0 mul; 0 bitwise)
    Layer 16: 461,440 ops (460,800 macc; 640 comp; 0 add; 0 mul; 0 bitwise)
    Layer 17: 15,744 ops (15,360 macc; 384 comp; 0 add; 0 mul; 0 bitwise)
    Layer 18: 38,400 ops (36,864 macc; 1,536 comp; 0 add; 0 mul; 0 bitwise)
    Layer 19: 664,320 ops (663,552 macc; 768 comp; 0 add; 0 mul; 0 bitwise)
    Layer 20: 25,088 ops (24,576 macc; 512 comp; 0 add; 0 mul; 0 bitwise)
    Layer 21: 25,344 ops (24,576 macc; 768 comp; 0 add; 0 mul; 0 bitwise)
    Layer 22: 768 ops (0 macc; 0 comp; 768 add; 0 mul; 0 bitwise)
    Layer 23: 4,800 ops (4,800 macc; 0 comp; 0 add; 0 mul; 0 bitwise)

  RESOURCE USAGE
  Weight memory: 121,136 bytes out of 442,368 bytes total (27.4%)
  Bias memory:   764 bytes out of 2,048 bytes total (37.3%)
*/
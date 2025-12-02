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

// my_asl_alexnet_small - Queue-based UART Benchmark System
// ASL (American Sign Language) classification with AlexNet-Small on MAX78000
//
// Memory layout for 64x64 RGB image (CHW format):
//   Channel 0 (R): 0x50400000 - 1024 words (4096 bytes)
//   Channel 1 (G): 0x50800000 - 1024 words (4096 bytes)
//   Channel 2 (B): 0x50c00000 - 1024 words (4096 bytes)
//
// Protocol: [DEADBEEF][NUM_IMAGES(4 bytes)][Label0][Image0][Label1][Image1]...

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include "mxc.h"
#include "cnn.h"
#include "led.h"

// Softmax function prototype
extern void softmax_q17p14_q15(const q31_t *vec_in, const uint16_t dim_vec, q15_t *p_out);

// ============================================================================
// Configuration
// ============================================================================
#define UART_BAUD           115200
#define RX_UART             MXC_UART0
#define START_MARKER        0xDEADBEEF
#define IMAGE_SIZE_DIM      64      // 64x64 image
#define CHANNEL_SIZE        4096    // 64x64 bytes per channel
#define CHANNEL_WORDS       1024    // 4096 bytes / 4
#define IMAGE_BYTES         12288   // 64x64x3 bytes total
#define MAX_QUEUE_SIZE      8       // Limited by SRAM (64x64 images are larger)
#define WARMUP_ITERATIONS   2
#define TIMEOUT_LOOPS       100000000

// ============================================================================
// Data Structures
// ============================================================================
typedef struct {
    uint32_t ch0[CHANNEL_WORDS];    // Channel 0 (R) - 1024 words
    uint32_t ch1[CHANNEL_WORDS];    // Channel 1 (G) - 1024 words
    uint32_t ch2[CHANNEL_WORDS];    // Channel 2 (B) - 1024 words
    uint8_t label;
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
    
    printf("[INFO] Waiting for start marker (0xDEADBEEF)...\n");
    
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
    static uint8_t image_data[IMAGE_BYTES];
    
    if (uart_read_byte_timeout(&label_byte, TIMEOUT_LOOPS) != 0) return -1;
    entry->label = label_byte;
    
    if (uart_read_bytes(image_data, IMAGE_BYTES, TIMEOUT_LOOPS) != IMAGE_BYTES) return -2;
    
    // Parse CHW format into separate channels (each channel 4096 bytes = 1024 words)
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

static int run_cnn_inference(float *confidence) {
    cnn_init();
    cnn_configure();
    cnn_start();
    
    SCB->SCR &= ~SCB_SCR_SLEEPDEEP_Msk;
    while (cnn_time == 0) { __WFI(); }
    
    cnn_unload((uint32_t *)ml_data);
    softmax_q17p14_q15((const q31_t *)ml_data, CNN_NUM_OUTPUTS, ml_softmax);
    
    int max_idx = 0;
    int32_t max_val = ml_data[0];
    for (int i = 1; i < CNN_NUM_OUTPUTS; i++) {
        if (ml_data[i] > max_val) { max_val = ml_data[i]; max_idx = i; }
    }
    
    if (confidence != NULL) {
        *confidence = (float)ml_softmax[max_idx] / 32768.0f * 100.0f;
    }
    
    return max_idx;
}

// ============================================================================
// Benchmark Functions
// ============================================================================
static void run_warmup(int iterations) {
    printf("[WARMUP] Running %d iterations...\n", iterations);
    
    ImageQueueEntry *first_img = queue_peek(0);
    if (first_img == NULL) { printf("[WARMUP] No images!\n"); return; }
    
    load_image_to_cnn(first_img);
    
    for (int i = 0; i < iterations; i++) {
        cnn_time = 0;
        run_cnn_inference(NULL);
    }
    
    printf("[WARMUP] %d iterations done.\n", iterations);
}

static void run_benchmark(uint32_t num_images) {
    uint32_t total_time_us = 0;
    uint32_t min_time = 0xFFFFFFFF, max_time = 0;
    uint32_t correct_count = 0;
    float total_confidence = 0.0f;
    
    printf("[BENCH] Start Loop (N=%lu)\n", num_images);
    
    for (uint32_t i = 0; i < num_images; i++) {
        ImageQueueEntry *img = queue_peek(i);
        if (img == NULL) break;
        
        load_image_to_cnn(img);
        cnn_time = 0;
        
        float confidence;
        int predicted = run_cnn_inference(&confidence);
        uint32_t latency = cnn_time;
        
        results[i].latency_us = latency;
        results[i].predicted_class = predicted;
        results[i].true_label = img->label;
        results[i].correct = (predicted == img->label);
        results[i].confidence = confidence;
        
        total_time_us += latency;
        total_confidence += confidence;
        if (latency < min_time) min_time = latency;
        if (latency > max_time) max_time = latency;
        if (results[i].correct) correct_count++;
        
        printf("Iter %02lu: %.1fms | Class: %d | GT: %d | Conf: %.1f%% | %s\n",
               i + 1, (double)latency / 1000.0, predicted, img->label,
               (double)confidence, results[i].correct ? "OK" : "FAIL");
    }
    
    printf("[BENCH] Loop End.\n");
    
    // Statistics
    float avg_time = (float)total_time_us / (float)num_images;
    float fps = 1000000.0f / avg_time;
    float accuracy = ((float)correct_count / (float)num_images) * 100.0f;
    float avg_confidence = total_confidence / (float)num_images;
    
    // ASL-AlexNet-Small constants (from cnn.h)
    const uint32_t total_ops = 27041536;
    const uint32_t total_macs = 26889600;
    const uint32_t weight_bytes = 182496;
    const uint32_t input_bytes = IMAGE_BYTES;
    const uint32_t output_bytes = CNN_NUM_OUTPUTS * 4;
    
    const float npu_freq_mhz = 50.0f;
    const uint32_t num_processors = 64;
    const float peak_tops = (npu_freq_mhz * 1e6f * num_processors * 2) / 1e12f;
    
    float core_latency_us = avg_time;
    float system_latency_us = core_latency_us + 15.0f;
    float effective_tops = (float)total_ops / (core_latency_us / 1e6f) / 1e12f;
    float effective_macs_tops = (float)total_macs / (core_latency_us / 1e6f) / 1e12f;
    float utilization_pct = (effective_tops / peak_tops) * 100.0f;
    
    uint32_t data_moved_bytes = input_bytes + output_bytes;
    float bandwidth_mb_per_sec = ((float)data_moved_bytes * fps) / (1024.0f * 1024.0f);
    float ops_per_byte = (float)total_ops / (float)data_moved_bytes;
    float arithmetic_intensity = (float)total_macs / (float)(weight_bytes + input_bytes);
    
    // Print Report
    printf("\n");
    printf("============================================================\n");
    printf("        BENCHMARK REPORT - ASL AlexNet-Small (64x64)        \n");
    printf("============================================================\n");
    
    printf("[BASIC STATISTICS]\n");
    printf("  Total Images:        %lu\n", num_images);
    printf("  Correct:             %lu/%lu (%.1f%%)\n", correct_count, num_images, (double)accuracy);
    printf("  Avg Confidence:      %.1f%%\n", (double)avg_confidence);
    printf("------------------------------------------------------------\n");
    
    printf("[LATENCY METRICS]\n");
    printf("  Core Latency (NPU):  %.2f us (%.3f ms)\n", (double)core_latency_us, (double)core_latency_us / 1000.0);
    printf("  System Latency:      %.2f us (%.3f ms)\n", (double)system_latency_us, (double)system_latency_us / 1000.0);
    printf("  Min Latency:         %lu us\n", min_time);
    printf("  Max Latency:         %lu us\n", max_time);
    printf("  Latency Jitter:      %lu us\n", max_time - min_time);
    printf("------------------------------------------------------------\n");
    
    printf("[THROUGHPUT METRICS]\n");
    printf("  Throughput:          %.2f FPS\n", (double)fps);
    printf("  Total Time:          %.2f ms (%lu images)\n", (double)total_time_us / 1000.0, num_images);
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
    printf("  Accuracy:            %s\n", (correct_count == num_images) ? "PERFECT" : "PARTIAL");
    printf("============================================================\n");
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
    printf("[INIT] ASL AlexNet-Small Benchmark System\n");
    printf("============================================================\n");
    printf("[INIT] Image Size: 64x64x3 (CHW) = %d bytes\n", IMAGE_BYTES);
    printf("[INIT] Queue Size: %d images max\n", MAX_QUEUE_SIZE);
    printf("[INIT] UART: %d baud\n", UART_BAUD);
    printf("[INIT] Classes: %d (ASL signs)\n", CNN_NUM_OUTPUTS);
    printf("[INIT] Memory Banks:\n");
    printf("         Channel 0 (R): 0x50400000\n");
    printf("         Channel 1 (G): 0x50800000\n");
    printf("         Channel 2 (B): 0x50c00000\n");
    
    printf("[INIT] Initializing UART...");
    error = MXC_UART_Init(RX_UART, UART_BAUD, MXC_UART_APB_CLK);
    if (error < E_NO_ERROR) { printf(" FAILED (%d)\n", error); return error; }
    printf(" OK\n");
    
    printf("[INIT] Loading ASL AlexNet-Small...");
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
  Hardware: 27,041,536 ops (26,889,600 macc; 147,840 comp; 4,096 add; 0 mul; 0 bitwise)
    Layer 0: 929,792 ops (884,736 macc; 45,056 comp; 0 add; 0 mul; 0 bitwise)
    Layer 1: 4,767,744 ops (4,718,592 macc; 49,152 comp; 0 add; 0 mul; 0 bitwise)
    Layer 2: 9,453,568 ops (9,437,184 macc; 16,384 comp; 0 add; 0 mul; 0 bitwise)
    Layer 3: 9,453,568 ops (9,437,184 macc; 16,384 comp; 0 add; 0 mul; 0 bitwise)
    Layer 4: 2,379,776 ops (2,359,296 macc; 20,480 comp; 0 add; 0 mul; 0 bitwise)
    Layer 5: 4,096 ops (0 macc; 0 comp; 4,096 add; 0 mul; 0 bitwise)
    Layer 6: 16,640 ops (16,384 macc; 256 comp; 0 add; 0 mul; 0 bitwise)
    Layer 7: 32,896 ops (32,768 macc; 128 comp; 0 add; 0 mul; 0 bitwise)
    Layer 8: 3,456 ops (3,456 macc; 0 comp; 0 add; 0 mul; 0 bitwise)

  RESOURCE USAGE
  Weight memory: 182,496 bytes out of 442,368 bytes total (41.3%)
  Bias memory:   699 bytes out of 2,048 bytes total (34.1%)
*/


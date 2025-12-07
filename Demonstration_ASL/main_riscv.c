/*******************************************************************************
* Copyright (C) 2019-2024 Maxim Integrated Products, Inc., All rights Reserved.
*
* ASL Recognition Demo - RISC-V Core
* 
* Dual-Core Architecture:
* ========================
* ARM Cortex-M4 (main.c):
*   - System clock and power initialization
*   - UART initialization for printf
*   - Camera power enable (GPIO)
*   - DMA clock enable
*   - Wakes up RISC-V core
*   - Waits for completion signal via semaphore
*
* RISC-V (this file):
*   - CNN accelerator control (init, load weights, inference)
*   - Camera capture and image preprocessing
*   - Classification and result output
*   - Signals ARM when done
*
* State Machine (Simplified):
*   INIT -> RUNNING -> DONE
*
* Target: MAX78000 FTHR_RevA
* CNN Model: ResNetV2-18 for ASL27 (A-Z + Unknown)
*******************************************************************************/

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <limits.h>
#include "mxc.h"
#include "mxc_device.h"
#include "mxc_delay.h"
#include "fcr_regs.h"
#include "sema_regs.h"
#include "icc.h"
#include "dma.h"
#include "led.h"
#include "pb.h"
#include "cnn.h"
#include "camera.h"

/* **** Definitions **** */
#define IMAGE_SIZE_X        64
#define IMAGE_SIZE_Y        64
#define CAMERA_SIZE_X       168
#define CAMERA_SIZE_Y       120
#define CAMERA_FREQ         10000000

#define NUM_CLASSES         CNN_NUM_OUTPUTS
#define INFERENCE_THRESHOLD 50
#define MAX_INFERENCES      20
#define CAPTURE_TIMEOUT     1000000  /* Loop iterations - increased */

/* **** Simplified State Machine **** */
typedef enum {
    STATE_INIT,     /* One-time initialization */
    STATE_RUNNING,  /* Capture -> Inference -> Result (loop) */
    STATE_DONE      /* Cleanup and halt */
} system_state_t;

/* **** Globals **** */
volatile uint32_t cnn_time;
static system_state_t current_state = STATE_INIT;

/* CNN buffers */
static int32_t ml_data[NUM_CLASSES];
static q15_t ml_softmax[NUM_CLASSES];

/* Statistics */
static uint32_t total_time_sum = 0;
static uint32_t min_time = 0xFFFFFFFF;
static uint32_t max_time = 0;
static uint16_t inference_count = 0;
static uint16_t recognized_count = 0;

/* DMA channel */
static int dma_channel = -1;

/* Class labels */
__attribute__((section(".rvflash_section"))) 
static const char labels[NUM_CLASSES][12] = {
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z", "Unknown"
};

/* **** Function Prototypes **** */
static int init_system(void);
static int run_single_inference(void);
static void print_summary(void);
static void load_image_to_cnn(void);
static void get_top3_predictions(int *idx, int *prob);

/*******************************************************************************
 * STATE: INIT - One-time system initialization
 * RISC-V responsibilities: CNN, DMA, Camera
 ******************************************************************************/
static int init_system(void)
{
    int ret;
    
    /* Enable RISC-V cache */
    MXC_ICC_Enable(MXC_ICC1);
    
    /* Wait for ARM core to finish its initialization */
    MXC_Delay(MXC_DELAY_MSEC(500));

    /* Print banner */
    printf("\n========================================================\n");
    printf("     ASL Recognition Demo - RISC-V Core (MAX78000)\n");
    printf("========================================================\n");
    printf("System Configuration:\n");
    printf("  Core:           RISC-V\n");
    printf("  Image Size:     %dx%d (RGB)\n", IMAGE_SIZE_X, IMAGE_SIZE_Y);
    printf("  Input Format:   CHW (3x%dx%d)\n", IMAGE_SIZE_X, IMAGE_SIZE_Y);
    printf("  Num Classes:    %d (A-Z + Unknown)\n", NUM_CLASSES);
    printf("  CNN Clock:      50 MHz (APB div 1)\n");
    printf("  Confidence:     %d%% threshold\n", INFERENCE_THRESHOLD);
    printf("========================================================\n\n");

    /* --- CNN Initialization --- */
    printf("Initializing CNN accelerator...\n");
    cnn_enable(MXC_S_GCR_PCLKDIV_CNNCLKSEL_PCLK, MXC_S_GCR_PCLKDIV_CNNCLKDIV_DIV1);
    cnn_boost_enable(MXC_GPIO2, MXC_GPIO_PIN_5);
    cnn_init();
    cnn_load_weights();
    cnn_load_bias();
    printf("  Weights loaded: 336,688 bytes (76.1%% of capacity)\n");
    printf("  Bias loaded:    811 bytes (39.6%% of capacity)\n");
    printf("  Total ops:      19,893,632 (19.58M MACC)\n\n");

    /* --- DMA Initialization --- */
    printf("Initializing camera...\n");
    MXC_DMA_Init();
    dma_channel = MXC_DMA_AcquireChannel();
    if (dma_channel < 0) {
        printf("  ERROR: DMA channel acquisition failed!\n");
        return -1;
    }
    printf("  DMA channel:    %d\n", dma_channel);

    /* --- Camera Initialization --- */
    camera_init(CAMERA_FREQ);
    camera_set_hmirror(0);
    camera_set_vflip(0);
    MXC_Delay(MXC_DELAY_MSEC(100));
    
    ret = camera_setup(CAMERA_SIZE_X, CAMERA_SIZE_Y, PIXFORMAT_BAYER,
                       FIFO_FOUR_BYTE, USE_DMA, dma_channel);
    if (ret != E_NO_ERROR) {
        printf("  ERROR: Camera setup failed (%d)!\n", ret);
        return -1;
    }
    printf("  Resolution:     %dx%d (capture) -> %dx%d (CNN)\n", 
           CAMERA_SIZE_X, CAMERA_SIZE_Y, IMAGE_SIZE_X, IMAGE_SIZE_Y);
    printf("  Pixel format:   BAYER\n\n");

    printf("========================================================\n");
    printf("*** READY - Running %d inferences ***\n", MAX_INFERENCES);
    printf("========================================================\n\n");
    return 0;
}

/*******************************************************************************
 * STATE: RUNNING - Single inference cycle
 * Capture -> Preprocess -> CNN -> Result
 ******************************************************************************/
static int run_single_inference(void)
{
    uint32_t t_capture, t_load, t_infer, t_unload, t_softmax, t_total;
    uint32_t timeout;
    int top3_idx[3], top3_prob[3];
    
    inference_count++;
    printf("=========== Inference #%d ===========\n", inference_count);

    /* --- Prepare CNN --- */
    cnn_init();
    cnn_load_bias();
    cnn_configure();
    cnn_time = 0;

    /* --- Capture Image --- */
    LED_On(LED1);
    MXC_Delay(MXC_DELAY_MSEC(10));
    MXC_TMR_SW_Start(MXC_TMR1);
    camera_start_capture_image();
    
    timeout = CAPTURE_TIMEOUT;
    while (!camera_is_image_rcv() && --timeout > 0) {}
    t_capture = MXC_TMR_SW_Stop(MXC_TMR1);
    
    if (timeout == 0) {
        printf("ERROR: Capture timeout\n");
        printf("=====================================\n\n");
        LED_Off(LED1);
        return -1;
    }

    /* --- Preprocess & Load to CNN --- */
    MXC_TMR_SW_Start(MXC_TMR1);
    load_image_to_cnn();
    t_load = MXC_TMR_SW_Stop(MXC_TMR1);

    /* --- Run CNN Inference --- */
    cnn_start();
    while (cnn_time == 0) {
        asm volatile("wfi");
    }
    t_infer = cnn_time;
    LED_Off(LED1);

    /* --- Unload Results --- */
    MXC_TMR_SW_Start(MXC_TMR1);
    cnn_unload((uint32_t *)ml_data);
    t_unload = MXC_TMR_SW_Stop(MXC_TMR1);

    /* --- Softmax --- */
    MXC_TMR_SW_Start(MXC_TMR1);
    softmax_q17p14_q15((const q31_t *)ml_data, NUM_CLASSES, ml_softmax);
    get_top3_predictions(top3_idx, top3_prob);
    t_softmax = MXC_TMR_SW_Stop(MXC_TMR1);

    /* --- Calculate Total Time --- */
    t_total = t_capture + t_load + t_infer + t_unload + t_softmax;

    /* --- Print Timing Breakdown --- */
    printf("Timing Breakdown:\n");
    printf("  Capture:        %lu us\n", t_capture);
    printf("  Preprocess/Load:%lu us\n", t_load);
    printf("  CNN Inference:  %lu us\n", t_infer);
    printf("  Unload:         %lu us\n", t_unload);
    printf("  Softmax:        %lu us\n", t_softmax);
    printf("  Total:          %lu us (%.2f ms)\n\n", t_total, (double)t_total / 1000.0);

    /* --- Print Top-3 Predictions --- */
    printf("Top-3 Predictions:\n");
    printf("  1. '%s': %d%% [raw: %ld]\n", labels[top3_idx[0]], top3_prob[0], ml_data[top3_idx[0]]);
    printf("  2. '%s': %d%% [raw: %ld]\n", labels[top3_idx[1]], top3_prob[1], ml_data[top3_idx[1]]);
    printf("  3. '%s': %d%% [raw: %ld]\n\n", labels[top3_idx[2]], top3_prob[2], ml_data[top3_idx[2]]);

    /* --- Print Result --- */
    if (top3_prob[0] >= INFERENCE_THRESHOLD && top3_idx[0] < NUM_CLASSES - 1) {
        printf(">>> RESULT: '%s' (%d%%)\n", labels[top3_idx[0]], top3_prob[0]);
        recognized_count++;
        LED_On(LED2);
        MXC_Delay(MXC_DELAY_MSEC(100));
        LED_Off(LED2);
    } else {
        printf(">>> RESULT: Low confidence (below %d%%)\n", INFERENCE_THRESHOLD);
    }

    /* --- Update Statistics --- */
    total_time_sum += t_total;
    if (t_total < min_time) min_time = t_total;
    if (t_total > max_time) max_time = t_total;

    printf("\nSession Stats: %d recognized, %d unknown\n", 
           recognized_count, inference_count - recognized_count);
    printf("Timing: min=%lu us, max=%lu us, avg=%lu us\n",
           min_time, max_time, total_time_sum / inference_count);
    printf("=====================================\n\n");
    
    return 0;
}

/*******************************************************************************
 * STATE: DONE - Print summary and signal ARM
 ******************************************************************************/
static void print_summary(void)
{
    uint16_t unknown_count = inference_count - recognized_count;
    uint32_t avg_time = inference_count > 0 ? total_time_sum / inference_count : 0;
    
    printf("\n============ SESSION SUMMARY ============\n");
    printf("Total Inferences: %d\n", inference_count);
    printf("  Recognized: %d (%d%%)\n", recognized_count,
           inference_count > 0 ? (100 * recognized_count) / inference_count : 0);
    printf("  Unknown:    %d (%d%%)\n", unknown_count,
           inference_count > 0 ? (100 * unknown_count) / inference_count : 0);
    printf("Timing Statistics:\n");
    printf("  Min: %lu us (%.2f ms)\n", min_time, (double)min_time / 1000.0);
    printf("  Max: %lu us (%.2f ms)\n", max_time, (double)max_time / 1000.0);
    printf("  Avg: %lu us (%.2f ms)\n", avg_time, (double)avg_time / 1000.0);
    printf("=========================================\n");
    
    /* Cleanup */
    cnn_disable();
    LED_Off(LED1);
    LED_Off(LED2);
    
    /* Signal ARM core that RISC-V is done */
    MXC_SEMA->irq0 = MXC_F_SEMA_IRQ0_EN | MXC_F_SEMA_IRQ0_CM4_IRQ;
}

/*******************************************************************************
 * Main - Simplified State Machine
 ******************************************************************************/
int main(void)
{
    current_state = STATE_INIT;

    while (1) {
        switch (current_state) {
            
        case STATE_INIT:
            if (init_system() == 0) {
                current_state = STATE_RUNNING;
            } else {
                current_state = STATE_DONE;  /* Fatal error */
            }
            break;

        case STATE_RUNNING:
            run_single_inference();
            MXC_Delay(MXC_DELAY_MSEC(500));  /* Inter-frame delay */
            
            if (inference_count >= MAX_INFERENCES) {
                current_state = STATE_DONE;
            }
            /* Stay in RUNNING state for next inference */
            break;

        case STATE_DONE:
            print_summary();
            while (1) { asm volatile("wfi"); }  /* Halt */
            break;
        }
    }
    
    return 0;
}

/*******************************************************************************
 * Helper: Load preprocessed image to CNN memory
 ******************************************************************************/
static void load_image_to_cnn(void)
{
    uint8_t *frame_buffer;
    uint32_t imgLen;
    uint32_t w, h;
    
    /* Get captured image from camera */
    camera_get_image(&frame_buffer, &imgLen, &w, &h);

    /*
     * BAYER demosaicing: Convert raw BAYER to RGB and load to CNN
     * CNN expects CHW format at specific memory addresses
     */
    uint8_t *raw = frame_buffer;
    
    static int8_t ch_r[IMAGE_SIZE_X * IMAGE_SIZE_Y];
    static int8_t ch_g[IMAGE_SIZE_X * IMAGE_SIZE_Y];
    static int8_t ch_b[IMAGE_SIZE_X * IMAGE_SIZE_Y];
    
    /* Center crop offsets */
    uint32_t start_x = ((w - IMAGE_SIZE_X) / 2) & ~1;
    uint32_t start_y = ((h - IMAGE_SIZE_Y) / 2) & ~1;
    
    int idx = 0;
    uint8_t r, g, b;
    
    for (uint32_t y = start_y; y < start_y + IMAGE_SIZE_Y; y++) {
        for (uint32_t x = start_x; x < start_x + IMAGE_SIZE_X; x++) {
            uint32_t pos = y * w + x;
            
            /* BAYER pattern (RGGB) demosaicing */
            if ((y & 1) == 0) {
                if ((x & 1) == 0) {
                    r = raw[pos];
                    g = (raw[pos + 1] + raw[pos + w]) / 2;
                    b = raw[pos + w + 1];
                } else {
                    r = raw[pos - 1];
                    g = raw[pos];
                    b = raw[pos + w];
                }
            } else {
                if ((x & 1) == 0) {
                    r = raw[pos - w];
                    g = raw[pos];
                    b = raw[pos + 1];
                } else {
                    r = raw[pos - w - 1];
                    g = (raw[pos - 1] + raw[pos - w]) / 2;
                    b = raw[pos];
                }
            }
            
            /* Normalize [0,255] -> [-128,127] */
            ch_r[idx] = (int8_t)(r - 128);
            ch_g[idx] = (int8_t)(g - 128);
            ch_b[idx] = (int8_t)(b - 128);
            idx++;
        }
    }

    /* Load to CNN: R->0x50400000, G->0x50800000, B->0x50C00000 */
    memcpy32((uint32_t *)0x50400000, (uint32_t *)ch_r, 1024);
    memcpy32((uint32_t *)0x50800000, (uint32_t *)ch_g, 1024);
    memcpy32((uint32_t *)0x50C00000, (uint32_t *)ch_b, 1024);
}

/*******************************************************************************
 * Helper: Get top-3 predictions from CNN output
 ******************************************************************************/
static void get_top3_predictions(int *idx, int *prob)
{
    /* Initialize top-3 with invalid values */
    int32_t top3_val[3] = {INT32_MIN, INT32_MIN, INT32_MIN};
    idx[0] = idx[1] = idx[2] = 0;
    
    /* Find top-3 by raw CNN output values */
    for (int i = 0; i < NUM_CLASSES; i++) {
        if (ml_data[i] > top3_val[0]) {
            /* Shift down */
            top3_val[2] = top3_val[1]; idx[2] = idx[1];
            top3_val[1] = top3_val[0]; idx[1] = idx[0];
            top3_val[0] = ml_data[i];  idx[0] = i;
        } else if (ml_data[i] > top3_val[1]) {
            top3_val[2] = top3_val[1]; idx[2] = idx[1];
            top3_val[1] = ml_data[i];  idx[1] = i;
        } else if (ml_data[i] > top3_val[2]) {
            top3_val[2] = ml_data[i];  idx[2] = i;
        }
    }
    
    /* Convert softmax to percentage */
    prob[0] = (100 * ml_softmax[idx[0]]) / 32768;
    prob[1] = (100 * ml_softmax[idx[1]]) / 32768;
    prob[2] = (100 * ml_softmax[idx[2]]) / 32768;
}

/*
  SUMMARY OF OPS
  Hardware: 19,893,632 ops (19,582,336 macc; 247,808 comp; 63,488 add; 0 mul; 0 bitwise)
    Layer 0 (conv1): 471,040 ops (442,368 macc; 28,672 comp; 0 add; 0 mul; 0 bitwise)
    Layer 1 (l1_0_conv): 4,751,360 ops (4,718,592 macc; 32,768 comp; 0 add; 0 mul; 0 bitwise)
    Layer 2 (l1_0_short): 557,056 ops (524,288 macc; 32,768 comp; 0 add; 0 mul; 0 bitwise)
    Layer 3 (l1_0_add): 32,768 ops (0 macc; 0 comp; 32,768 add; 0 mul; 0 bitwise)
    Layer 4 (l2_0_conv): 4,767,744 ops (4,718,592 macc; 49,152 comp; 0 add; 0 mul; 0 bitwise)
    Layer 5 (l2_0_short): 573,440 ops (524,288 macc; 49,152 comp; 0 add; 0 mul; 0 bitwise)
    Layer 6 (l2_0_add): 16,384 ops (0 macc; 0 comp; 16,384 add; 0 mul; 0 bitwise)
    Layer 7 (l3_0_conv): 2,379,776 ops (2,359,296 macc; 20,480 comp; 0 add; 0 mul; 0 bitwise)
    Layer 8 (l3_0_short): 278,528 ops (262,144 macc; 16,384 comp; 0 add; 0 mul; 0 bitwise)
    Layer 9 (l3_0_add): 4,096 ops (0 macc; 0 comp; 4,096 add; 0 mul; 0 bitwise)
    Layer 10 (l3_1_conv): 2,363,392 ops (2,359,296 macc; 4,096 comp; 0 add; 0 mul; 0 bitwise)
    Layer 11 (l3_0_add_gap): 0 ops (0 macc; 0 comp; 0 add; 0 mul; 0 bitwise)
    Layer 12 (l3_1_add): 4,096 ops (0 macc; 0 comp; 4,096 add; 0 mul; 0 bitwise)
    Layer 13 (l4_0_conv): 1,185,792 ops (1,179,648 macc; 6,144 comp; 0 add; 0 mul; 0 bitwise)
    Layer 14 (l4_0_short): 137,216 ops (131,072 macc; 6,144 comp; 0 add; 0 mul; 0 bitwise)
    Layer 15 (l4_0_add): 2,048 ops (0 macc; 0 comp; 2,048 add; 0 mul; 0 bitwise)
    Layer 16 (l4_1_conv): 2,361,344 ops (2,359,296 macc; 2,048 comp; 0 add; 0 mul; 0 bitwise)
    Layer 17 (l4_0_add_gap): 0 ops (0 macc; 0 comp; 0 add; 0 mul; 0 bitwise)
    Layer 18 (l4_1_add): 2,048 ops (0 macc; 0 comp; 2,048 add; 0 mul; 0 bitwise)
    Layer 19 (avg_pool): 2,048 ops (0 macc; 0 comp; 2,048 add; 0 mul; 0 bitwise)
    Layer 20 (fc): 3,456 ops (3,456 macc; 0 comp; 0 add; 0 mul; 0 bitwise)

  RESOURCE USAGE
  Weight memory: 336,688 bytes out of 442,368 bytes total (76.1%)
  Bias memory:   811 bytes out of 2,048 bytes total (39.6%)
*/


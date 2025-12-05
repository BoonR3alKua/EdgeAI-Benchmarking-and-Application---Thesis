/*******************************************************************************
* Copyright (C) 2019-2024 Maxim Integrated Products, Inc., All rights Reserved.
*
* ASL Recognition Demo - RISC-V Core
* 
* This application captures images from the camera, preprocesses them to 
* 64x64x3 CHW format, runs CNN inference for ASL (American Sign Language)
* recognition, and outputs results via UART.
*
* Target: MAX78000 FTHR_RevA
* CNN Model: ResNetV2-18 for ASL27 (A-Z + Unknown)
*******************************************************************************/

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
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
#define IMAGE_SIZE_X        64      // CNN input width
#define IMAGE_SIZE_Y        64      // CNN input height
#define CAMERA_SIZE_X       168     // Camera capture width (OV7692 minimum)
#define CAMERA_SIZE_Y       120     // Camera capture height
#define CAMERA_FREQ         10000000    // 10 MHz

#define NUM_CLASSES         CNN_NUM_OUTPUTS  // 27 classes (A-Z + Unknown)
#define INFERENCE_THRESHOLD 50               // Minimum confidence % to accept

/* **** Globals **** */
volatile uint32_t cnn_time;

/* CNN output buffers */
static int32_t ml_data[NUM_CLASSES];
static q15_t ml_softmax[NUM_CLASSES];

/* Statistics tracking */
static uint32_t total_inference_time = 0;
static uint32_t min_inference_time = 0xFFFFFFFF;
static uint32_t max_inference_time = 0;
static uint16_t inference_count = 0;
static uint16_t recognized_count = 0;
static uint16_t unknown_count = 0;

/* ASL Class Labels (A-Z + Unknown) */
__attribute__((section(".rvflash_section"))) 
static const char class_labels[NUM_CLASSES][12] = {
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z", "Unknown"
};

/* **** Function Prototypes **** */
static void fail(void);
static void capture_and_load_image(void);
static uint8_t check_inference(q15_t *ml_soft, int32_t *ml_data, 
                               int16_t *out_class, int *out_prob);
static void print_top_predictions(int32_t *data, q15_t *softmax, int top_n);

/* **** Main Function **** */
int main(void)
{
    int16_t out_class = -1;
    int probability = 0;
    uint32_t capture_time, load_time, inference_time, unload_time, softmax_time;
    uint8_t ret;
    int dma_channel;

    /* === RISC-V Initialization === */
    MXC_ICC_Enable(MXC_ICC1);  // Enable RISC-V cache

    /* Small delay to ensure ARM has finished all initialization */
    MXC_Delay(MXC_DELAY_MSEC(500));

    /* Print System Banner - UART already initialized by ARM */
    printf("\n");
    printf("========================================================\n");
    printf("     ASL Recognition Demo - RISC-V Core (MAX78000)\n");
    printf("========================================================\n");
    printf("System Configuration:\n");
    printf("  Core:           RISC-V\n");
    printf("  Image Size:     %dx%d (RGB)\n", IMAGE_SIZE_X, IMAGE_SIZE_Y);
    printf("  Input Format:   CHW (3x64x64)\n");
    printf("  Num Classes:    %d (A-Z + Unknown)\n", NUM_CLASSES);
    printf("  CNN Clock:      50 MHz (APB div 1)\n");
    printf("  Confidence:     %d%% threshold\n", INFERENCE_THRESHOLD);
    printf("========================================================\n");
    fflush(stdout);;

    /* === CNN Initialization === */
    printf("\nInitializing CNN accelerator...\n");
    cnn_enable(MXC_S_GCR_PCLKDIV_CNNCLKSEL_PCLK, MXC_S_GCR_PCLKDIV_CNNCLKDIV_DIV1);
    cnn_boost_enable(MXC_GPIO2, MXC_GPIO_PIN_5);
    cnn_init();
    cnn_load_weights();
    cnn_load_bias();
    printf("  Weights loaded: 336,688 bytes (76.1%% of capacity)\n");
    printf("  Bias loaded:    811 bytes (39.6%% of capacity)\n");
    printf("  Total ops:      19,893,632 (19.58M MACC)\n");

    /* === Camera Initialization === */
    printf("\nInitializing camera...\n");
    
    /* Initialize DMA for RISC-V (ARM already enabled clock, but RISC-V needs its own init) */
    MXC_DMA_Init();
    dma_channel = MXC_DMA_AcquireChannel();
    if (dma_channel < 0) {
        printf("ERROR: Failed to acquire DMA channel (%d)\n", dma_channel);
        fail();
    }
    printf("  DMA channel:    %d\n", dma_channel);

    /* Initialize camera (camera power already enabled by ARM) */
    camera_init(CAMERA_FREQ);
    
    int slaveAddress = camera_get_slave_address();
    printf("  I2C address:    0x%02X\n", slaveAddress);

    int camera_id = 0;
    camera_get_manufacture_id(&camera_id);
    printf("  Camera ID:      0x%04X\n", camera_id);

    /* Configure camera - use settings similar to cam01_facedetect_demo */
    camera_set_hmirror(0);
    camera_set_vflip(0);
    
    /* Add delay for camera to stabilize after init */
    MXC_Delay(MXC_DELAY_MSEC(100));
    
    int ret_cam = camera_setup(CAMERA_SIZE_X,     // width
                               CAMERA_SIZE_Y,     // height
                               PIXFORMAT_BAYER,   // BAYER format (like facedetect demo)
                               FIFO_FOUR_BYTE,    // FIFO mode
                               USE_DMA,           // Enable DMA
                               dma_channel);      // DMA channel

    if (ret_cam != E_NO_ERROR) {
        printf("ERROR: Camera setup failed! (%d)\n", ret_cam);
        fail();
    }
    printf("  Resolution:     %dx%d (capture) -> %dx%d (CNN)\n", 
           CAMERA_SIZE_X, CAMERA_SIZE_Y, IMAGE_SIZE_X, IMAGE_SIZE_Y);
    printf("  Pixel format:   BAYER\n");

    printf("\n========================================================\n");
    printf("*** READY - Running 20 inferences ***\n");
    printf("========================================================\n\n");

    /* === Main Processing Loop - 20 inferences === */
    while (inference_count < 20) {
        inference_count++;
        cnn_time = 0;

        printf("=========== Inference #%d ===========\n", inference_count);

        /* --- Re-initialize CNN for new inference --- */
        cnn_init();
        cnn_load_bias();
        cnn_configure();

        /* --- Capture Image --- */
        LED_On(LED1);
        MXC_TMR_SW_Start(MXC_TMR1);
        
        camera_start_capture_image();
        while (!camera_is_image_rcv()) {}
        
        capture_time = MXC_TMR_SW_Stop(MXC_TMR1);

        /* --- Load Image to CNN (with preprocessing) --- */
        MXC_TMR_SW_Start(MXC_TMR1);
        capture_and_load_image();
        load_time = MXC_TMR_SW_Stop(MXC_TMR1);

        /* --- Start CNN Inference --- */
        cnn_start();

        /* --- Wait for CNN Completion --- */
        while (cnn_time == 0) {
            asm volatile("wfi");
        }
        inference_time = cnn_time;
        LED_Off(LED1);

        /* --- Unload CNN Results --- */
        MXC_TMR_SW_Start(MXC_TMR1);
        cnn_unload((uint32_t *)ml_data);
        unload_time = MXC_TMR_SW_Stop(MXC_TMR1);

        /* --- Softmax Post-Processing --- */
        MXC_TMR_SW_Start(MXC_TMR1);
        softmax_q17p14_q15((const q31_t *)ml_data, NUM_CLASSES, ml_softmax);
        ret = check_inference(ml_softmax, ml_data, &out_class, &probability);
        softmax_time = MXC_TMR_SW_Stop(MXC_TMR1);

        /* --- Print Timing Breakdown --- */
        printf("Timing Breakdown:\n");
        printf("  Capture:        %u us\n", capture_time);
        printf("  Preprocess/Load:%u us\n", load_time);
        printf("  CNN Inference:  %u us\n", inference_time);
        printf("  Unload:         %u us\n", unload_time);
        printf("  Softmax:        %u us\n", softmax_time);
        
        uint32_t total_time = capture_time + load_time + inference_time + unload_time + softmax_time;
        printf("  Total:          %u us (%u.%u ms)\n", total_time, total_time / 1000, (total_time % 1000) / 10);

        /* --- Print Top-3 Predictions --- */
        printf("\nTop-3 Predictions:\n");
        print_top_predictions(ml_data, ml_softmax, 3);

        /* --- Print Final Result --- */
        printf("\n>>> RESULT: ");
        if (!ret) {
            printf("Low confidence (below %d%%)\n", INFERENCE_THRESHOLD);
            unknown_count++;
        } else if (out_class == NUM_CLASSES - 1) {
            printf("Unknown gesture\n");
            unknown_count++;
        } else {
            /* Print probability as integer percentage */
            int prob_int = (int)probability;
            printf("'%s' (%d%%)\n", class_labels[out_class], prob_int);
            recognized_count++;
            LED_On(LED2);
            MXC_Delay(MXC_DELAY_MSEC(100));
            LED_Off(LED2);
        }

        /* --- Update Statistics --- */
        total_inference_time += total_time;
        if (total_time < min_inference_time) min_inference_time = total_time;
        if (total_time > max_inference_time) max_inference_time = total_time;

        printf("\nSession Stats: %d recognized, %d unknown\n", 
               recognized_count, unknown_count);
        printf("Timing: min=%u us, max=%u us, avg=%u us\n",
               min_inference_time, max_inference_time, 
               total_inference_time / inference_count);
        printf("=====================================\n\n");
        fflush(stdout);

        /* Short delay between inferences */
        MXC_Delay(MXC_DELAY_MSEC(500));
    }

    /* === Print Session Summary === */
    printf("\n");
    printf("============ SESSION SUMMARY ============\n");
    printf("Total Inferences: %d\n", inference_count);
    if (inference_count > 0) {
        printf("  Recognized: %d (%d%%)\n", recognized_count,
               (100 * recognized_count) / inference_count);
        printf("  Unknown:    %d (%d%%)\n", unknown_count,
               (100 * unknown_count) / inference_count);
    }
    printf("Timing Statistics:\n");
    printf("  Min: %u us (%u.%u ms)\n", min_inference_time, 
           min_inference_time / 1000, (min_inference_time % 1000) / 10);
    printf("  Max: %u us (%u.%u ms)\n", max_inference_time,
           max_inference_time / 1000, (max_inference_time % 1000) / 10);
    if (inference_count > 0) {
        uint32_t avg_time = total_inference_time / inference_count;
        printf("  Avg: %u us (%u.%u ms)\n", avg_time, avg_time / 1000, (avg_time % 1000) / 10);
    }
    printf("=========================================\n");

    /* Disable CNN and signal ARM core */
    cnn_disable();
    MXC_SEMA->irq0 = MXC_F_SEMA_IRQ0_EN | MXC_F_SEMA_IRQ0_CM4_IRQ;

    while (1) {}
    return 0;
}

/* **** Capture and Load Image to CNN **** */
static void capture_and_load_image(void)
{
    uint8_t *frame_buffer;
    uint32_t imgLen;
    uint32_t w, h;
    
    /* Get captured image from camera */
    camera_get_image(&frame_buffer, &imgLen, &w, &h);

    /*
     * The CNN expects input in CHW format (Channel-Height-Width):
     *   - Channel 0 (R): stored at 0x50400000
     *   - Channel 1 (G): stored at 0x50800000  
     *   - Channel 2 (B): stored at 0x50C00000
     * 
     * Each channel is 64x64 = 4096 bytes = 1024 words
     * 
     * Camera outputs BAYER (raw) format - simple 2x2 pattern:
     *   R  G
     *   G  B
     * We use simple nearest-neighbor to extract RGB from 2x2 blocks
     * and crop center 64x64 from the larger image
     */

    uint8_t *raw = frame_buffer;
    
    /* Temporary buffers for each channel */
    static int8_t ch_r[IMAGE_SIZE_X * IMAGE_SIZE_Y];
    static int8_t ch_g[IMAGE_SIZE_X * IMAGE_SIZE_Y];
    static int8_t ch_b[IMAGE_SIZE_X * IMAGE_SIZE_Y];
    
    /* Calculate crop offsets to center the 64x64 region */
    uint32_t start_x = (w - IMAGE_SIZE_X) / 2;
    uint32_t start_y = (h - IMAGE_SIZE_Y) / 2;
    
    /* Ensure start positions are even for BAYER pattern alignment */
    start_x &= ~1;
    start_y &= ~1;
    
    int idx = 0;
    uint8_t r, g, b;
    
    /* Simple BAYER demosaicing with crop */
    for (uint32_t y = start_y; y < start_y + IMAGE_SIZE_Y; y++) {
        for (uint32_t x = start_x; x < start_x + IMAGE_SIZE_X; x++) {
            uint32_t pos = y * w + x;
            
            /* Simple nearest-neighbor demosaicing based on position in 2x2 block */
            /* BAYER pattern (RGGB): 
             *   Row even: R G R G ...
             *   Row odd:  G B G B ...
             */
            if ((y & 1) == 0) {
                /* Even row */
                if ((x & 1) == 0) {
                    /* R position - get R directly, interpolate G and B */
                    r = raw[pos];
                    g = (raw[pos + 1] + raw[pos + w]) / 2;
                    b = raw[pos + w + 1];
                } else {
                    /* G position (on R row) */
                    r = raw[pos - 1];
                    g = raw[pos];
                    b = raw[pos + w];
                }
            } else {
                /* Odd row */
                if ((x & 1) == 0) {
                    /* G position (on B row) */
                    r = raw[pos - w];
                    g = raw[pos];
                    b = raw[pos + 1];
                } else {
                    /* B position - get B directly, interpolate R and G */
                    r = raw[pos - w - 1];
                    g = (raw[pos - 1] + raw[pos - w]) / 2;
                    b = raw[pos];
                }
            }
            
            /* Normalize [0, 255] -> [-128, 127] */
            ch_r[idx] = (int8_t)(r - 128);
            ch_g[idx] = (int8_t)(g - 128);
            ch_b[idx] = (int8_t)(b - 128);
            
            idx++;
        }
    }

    /* Load to CNN data memory in CHW format */
    /* Channel 0 (R) -> 0x50400000 */
    memcpy32((uint32_t *)0x50400000, (uint32_t *)ch_r, 1024);
    
    /* Channel 1 (G) -> 0x50800000 */
    memcpy32((uint32_t *)0x50800000, (uint32_t *)ch_g, 1024);
    
    /* Channel 2 (B) -> 0x50C00000 */
    memcpy32((uint32_t *)0x50C00000, (uint32_t *)ch_b, 1024);
}

/* **** Check Inference Result **** */
static uint8_t check_inference(q15_t *ml_soft, int32_t *ml_data, 
                               int16_t *out_class, int *out_prob)
{
    q15_t max = 0;
    int32_t max_ml = 1 << 31;
    int16_t max_index = -1;

    /* Find class with highest probability */
    for (int i = 0; i < NUM_CLASSES; i++) {
        if (ml_data[i] > max_ml) {
            max_ml = ml_data[i];
            max = ml_soft[i];
            max_index = i;
        }
    }

    *out_class = max_index;
    *out_prob = (100 * max) / 32768;  /* Integer percentage */

    return (*out_prob >= INFERENCE_THRESHOLD) ? 1 : 0;
}

/* **** Print Top-N Predictions **** */
static void print_top_predictions(int32_t *data, q15_t *softmax, int top_n)
{
    int32_t temp[NUM_CLASSES];
    memcpy(temp, data, sizeof(int32_t) * NUM_CLASSES);

    for (int top = 0; top < top_n; top++) {
        int32_t max_val = 1 << 31;
        int max_idx = -1;

        /* Find maximum */
        for (int i = 0; i < NUM_CLASSES; i++) {
            if (temp[i] > max_val) {
                max_val = temp[i];
                max_idx = i;
            }
        }

        if (max_idx >= 0) {
            /* Calculate percentage using integer math: (softmax * 100) / 32768 */
            int prob_int = (100 * softmax[max_idx]) / 32768;
            printf("  %d. '%s': %d%% [raw: %d]\n", 
                   top + 1, class_labels[max_idx], prob_int, max_val);
            temp[max_idx] = 1 << 31;  // Mark as used
        }
    }
}

/* **** Fail Handler **** */
static void fail(void)
{
    printf("\n*** FAIL ***\n");
    LED_On(LED1);
    LED_On(LED2);
    while (1) {}
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


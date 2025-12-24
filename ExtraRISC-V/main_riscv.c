/*******************************************************************************
* Copyright (C) 2019-2024 Maxim Integrated Products, Inc., All rights Reserved.
*
* ASL Recognition Demo - RISC-V Core with FIFO Streaming
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
*   - CNN accelerator control with FIFO streaming
*   - Camera capture and image preprocessing
*   - Classification and result output
*   - Signals ARM when done
*
* Key Difference from DMA approach:
*   FIFO mode streams data directly to CNN while inference is running
*   Data writes to FIFO addresses AFTER cnn_start() is called
*
* Target: MAX78000 FTHR_RevA
* CNN Model: AlexNet-Small for ASL27 (A-Z + Unknown)
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
#include "cameraif.h"

/* **** Definitions **** */
#define IMAGE_SIZE_X        64
#define IMAGE_SIZE_Y        64
#define CAMERA_SIZE_X       168
#define CAMERA_SIZE_Y       120
#define CAMERA_FREQ         10000000

#define NUM_CLASSES         CNN_NUM_OUTPUTS
#define INFERENCE_THRESHOLD 50
#define MAX_INFERENCES      20
#define CAPTURE_TIMEOUT     1000000

/* Energy Measurement Markers */
#define ENERGY_MARKER_TOTAL_PORT    MXC_GPIO0
#define ENERGY_MARKER_TOTAL_PIN     MXC_GPIO_PIN_19  /* P0_19 (J4-9) - Total inference */
#define ENERGY_MARKER_CNN_PORT      MXC_GPIO1
#define ENERGY_MARKER_CNN_PIN       MXC_GPIO_PIN_6   /* P1_6 (J4-4) - CNN only */

/* FIFO Addresses for CNN data input (streaming mode) */
#define FIFO_STATUS_REG     ((volatile uint32_t *)0x50000004)
#define FIFO_0_WRITE        ((volatile uint32_t *)0x50000008)  /* R channel */
#define FIFO_1_WRITE        ((volatile uint32_t *)0x5000000c)  /* G channel */
#define FIFO_2_WRITE        ((volatile uint32_t *)0x50000010)  /* B channel */
#define FIFO_0_FULL         0x01
#define FIFO_1_FULL         0x02
#define FIFO_2_FULL         0x04

/* **** Simplified State Machine **** */
typedef enum {
    STATE_INIT,
    STATE_RUNNING,
    STATE_DONE
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

/* DMA channel (for camera driver compatibility) */
static int dma_channel = -1;

/* Preprocessed image buffers (CHW format as uint32_t for FIFO) */
/* Max size for 64x64: 64*64 = 4096 bytes / 4 = 1024 words */
static uint32_t ch_r[1024];
static uint32_t ch_g[1024];
static uint32_t ch_b[1024];

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
static void preprocess_camera_image(void);
static void load_input_fifo(void);
static void get_top3_predictions(int *idx, int *prob);
static void energy_measurement_init(void);

/* ========== ENERGY MEASUREMENT FUNCTIONS ========== */
static void energy_measurement_init(void)
{
    mxc_gpio_cfg_t gpio_cfg;
    
    /* Configure P0_19 (Total inference marker) */
    gpio_cfg.port = ENERGY_MARKER_TOTAL_PORT;
    gpio_cfg.mask = ENERGY_MARKER_TOTAL_PIN;
    gpio_cfg.pad = MXC_GPIO_PAD_NONE;
    gpio_cfg.func = MXC_GPIO_FUNC_OUT;
    gpio_cfg.vssel = MXC_GPIO_VSSEL_VDDIO;
    MXC_GPIO_Config(&gpio_cfg);
    MXC_GPIO_OutClr(ENERGY_MARKER_TOTAL_PORT, ENERGY_MARKER_TOTAL_PIN);
    
    /* Configure P1_6 (CNN-only marker) */
    gpio_cfg.port = ENERGY_MARKER_CNN_PORT;
    gpio_cfg.mask = ENERGY_MARKER_CNN_PIN;
    gpio_cfg.pad = MXC_GPIO_PAD_NONE;
    gpio_cfg.func = MXC_GPIO_FUNC_OUT;
    gpio_cfg.vssel = MXC_GPIO_VSSEL_VDDIO;
    MXC_GPIO_Config(&gpio_cfg);
    MXC_GPIO_OutClr(ENERGY_MARKER_CNN_PORT, ENERGY_MARKER_CNN_PIN);
    
    printf("Energy Measurement Configuration:\n");
    printf("  Total Inference Marker: P0_19 (J4-9)\n");
    printf("  CNN-Only Marker:        P1_6  (J4-4)\n");
    printf("  Connect oscilloscope or power analyzer to these pins\n");
    printf("  Measure power at VBUS (J4-3) with GND (J8-4 or J4-16)\n\n");
}

static inline void energy_marker_total_high(void) {
    MXC_GPIO_OutSet(ENERGY_MARKER_TOTAL_PORT, ENERGY_MARKER_TOTAL_PIN);
}

static inline void energy_marker_total_low(void) {
    MXC_GPIO_OutClr(ENERGY_MARKER_TOTAL_PORT, ENERGY_MARKER_TOTAL_PIN);
}

static inline void energy_marker_cnn_high(void) {
    MXC_GPIO_OutSet(ENERGY_MARKER_CNN_PORT, ENERGY_MARKER_CNN_PIN);
}

static inline void energy_marker_cnn_low(void) {
    MXC_GPIO_OutClr(ENERGY_MARKER_CNN_PORT, ENERGY_MARKER_CNN_PIN);
}
/* ================================================== */

/*******************************************************************************
 * STATE: INIT - One-time system initialization
 ******************************************************************************/
static int init_system(void)
{
    int ret;
    
    MXC_ICC_Enable(MXC_ICC1);
    MXC_Delay(MXC_DELAY_MSEC(500));

    printf("\n========================================================\n");
    printf("     ASL Recognition Demo - FIFO Streaming Mode\n");
    printf("========================================================\n");
    printf("System Configuration:\n");
    printf("  Core:           RISC-V\n");
    printf("  Image Size:     %dx%d (RGB)\n", IMAGE_SIZE_X, IMAGE_SIZE_Y);
    printf("  Input Format:   CHW via FIFO streaming\n");
    printf("  Num Classes:    %d (A-Z + Unknown)\n", NUM_CLASSES);
    printf("  CNN Model:      AlexNet-Small (27.04M ops)\n");
    printf("  CNN Clock:      50 MHz (APB div 1)\n");
    printf("  Confidence:     %d%% threshold\n", INFERENCE_THRESHOLD);
    printf("========================================================\n\n");

    /* --- Energy Measurement Setup --- */
    printf("Initializing energy measurement...\n");
    energy_measurement_init();

    /* --- CNN Initialization --- */
    printf("Initializing CNN accelerator...\n");
    cnn_enable(MXC_S_GCR_PCLKDIV_CNNCLKSEL_PCLK, MXC_S_GCR_PCLKDIV_CNNCLKDIV_DIV1);
    cnn_boost_enable(MXC_GPIO2, MXC_GPIO_PIN_5);
    cnn_init();
    cnn_load_weights();
    cnn_load_bias();
    printf("  Weights loaded: 182,496 bytes (41.3%% of capacity)\n");
    printf("  Bias loaded:    700 bytes (34.2%% of capacity)\n");
    printf("  Total ops:      27,041,536 (26.89M MACC)\n\n");

    /* --- Camera Initialization --- */
    printf("Initializing camera...\n");
    
    /* Initialize DMA - required for all camera modes */
    MXC_DMA_Init();
    dma_channel = MXC_DMA_AcquireChannel();
    if (dma_channel < 0) {
        printf("  ERROR: DMA channel acquisition failed!\n");
        return -1;
    }
    printf("  DMA channel:    %d\n", dma_channel);

    /* Initialize camera - following CameraIF example pattern */
    camera_init(CAMERA_FREQ);
    
    /* Verify camera is responding */
    int slave_addr = camera_get_slave_address();
    printf("  Camera I2C slave address: %02x\n", slave_addr);
    
    int camera_id;
    ret = camera_get_product_id(&camera_id);
    if (ret != STATUS_OK) {
        printf("  ERROR: Failed to read camera ID (%d)\n", ret);
        return -1;
    }
    printf("  Camera ID detected: %04x\n", camera_id);
    
    /* Setup camera for FIFO streaming (STREAMING_DMA mode) */
    ret = camera_setup(CAMERA_SIZE_X, CAMERA_SIZE_Y, PIXFORMAT_BAYER,
                       FIFO_FOUR_BYTE, STREAMING_DMA, dma_channel);
    if (ret != E_NO_ERROR) {
        printf("  ERROR: Camera setup failed (%d)!\n", ret);
        printf("  Possible causes:\n");
        printf("    -1 = E_NO_DEVICE (camera not detected)\n");
        printf("    -6 = E_INVALID (invalid parameters)\n");
        printf("    -10 = E_TIME_OUT (I2C timeout)\n");
        return -1;
    }
    
    printf("  Resolution:     %dx%d (capture) -> %dx%d (CNN)\n", 
           CAMERA_SIZE_X, CAMERA_SIZE_Y, IMAGE_SIZE_X, IMAGE_SIZE_Y);
    printf("  Mode:           FIFO STREAMING_DMA\n");
    printf("  Pixel format:   BAYER\n\n");
    printf("  Resolution:     %dx%d (capture) -> %dx%d (CNN)\n", 
           CAMERA_SIZE_X, CAMERA_SIZE_Y, IMAGE_SIZE_X, IMAGE_SIZE_Y);
    printf("  Pixel format:   BAYER\n\n");

    printf("========================================================\n");
    printf("*** READY - Running %d inferences with FIFO streaming ***\n", MAX_INFERENCES);
    printf("========================================================\n\n");
    return 0;
}

/*******************************************************************************
 * STATE: RUNNING - Single inference cycle with FIFO streaming
 * Capture -> Preprocess -> CNN_start -> FIFO_load -> Wait -> Result
 ******************************************************************************/
static int run_single_inference(void)
{
    uint32_t t_capture, t_preprocess, t_fifo_infer, t_unload, t_softmax, t_total;
    uint32_t timeout;
    int top3_idx[3], top3_prob[3];
    
    inference_count++;
    printf("=========== Inference #%d ===========\n", inference_count);

    /* ========== START TOTAL ENERGY MEASUREMENT ========== */
    energy_marker_total_high();
    /* ==================================================== */

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
        energy_marker_total_low();  /* Clean up on error */
        return -1;
    }

    /* --- Preprocess camera image into CHW buffers --- */
    MXC_TMR_SW_Start(MXC_TMR1);
    preprocess_camera_image();
    t_preprocess = MXC_TMR_SW_Stop(MXC_TMR1);

    /* --- Start CNN then stream data via FIFO --- */
    MXC_TMR_SW_Start(MXC_TMR1);
    
    /* ========== START CNN-ONLY MEASUREMENT ========== */
    energy_marker_cnn_high();
    /* ================================================= */
    
    cnn_start();           /* Start CNN FIRST */
    load_input_fifo();     /* Then stream data via FIFO */
    
    /* Wait for CNN inference to complete */
    while (cnn_time == 0) {
        asm volatile("wfi");
    }
    
    /* ========== STOP CNN-ONLY MEASUREMENT ========== */
    energy_marker_cnn_low();
    /* ================================================ */
    
    t_fifo_infer = MXC_TMR_SW_Stop(MXC_TMR1);
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

    /* ========== STOP TOTAL ENERGY MEASUREMENT ========== */
    energy_marker_total_low();
    /* ==================================================== */

    /* --- Calculate Total Time --- */
    t_total = t_capture + t_preprocess + t_fifo_infer + t_unload + t_softmax;

    /* --- Print Timing Breakdown --- */
    printf("Timing Breakdown:\n");
    printf("  Capture:         %lu us\n", t_capture);
    printf("  Preprocess:      %lu us\n", t_preprocess);
    printf("  FIFO+Inference:  %lu us (CNN timer: %lu us)\n", t_fifo_infer, cnn_time);
    printf("  Unload:          %lu us\n", t_unload);
    printf("  Softmax:         %lu us\n", t_softmax);
    printf("  Total:           %lu us (%.2f ms)\n\n", t_total, (double)t_total / 1000.0);

    /* --- Print Energy Measurement Info --- */
    printf("Energy Measurement Markers:\n");
    printf("  P0_19 (J4-9) was HIGH for: %lu us (Total inference)\n", t_total);
    printf("  P1_6  (J4-4) was HIGH for: %lu us (CNN only)\n", cnn_time);
    printf("  Other processing time:     %lu us (capture + preprocess + unload + softmax)\n\n",
           t_capture + t_preprocess + t_unload + t_softmax);

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
    
    cnn_disable();
    LED_Off(LED1);
    LED_Off(LED2);
    
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
                current_state = STATE_DONE;
            }
            break;

        case STATE_RUNNING:
            run_single_inference();
            MXC_Delay(MXC_DELAY_MSEC(500));
            
            if (inference_count >= MAX_INFERENCES) {
                current_state = STATE_DONE;
            }
            break;

        case STATE_DONE:
            print_summary();
            while (1) { asm volatile("wfi"); }
            break;
        }
    }
    
    return 0;
}

/*******************************************************************************
 * Helper: Preprocess camera BAYER image into CHW uint32_t buffers  
 * For STREAMING_DMA mode, we need to read line by line from streaming buffers
 ******************************************************************************/
static void preprocess_camera_image(void)
{
    /* Temporary byte buffers for preprocessing - fixed max size to avoid overflow */
    int8_t temp_r[64 * 64];  /* Max size: 4096 bytes */
    int8_t temp_g[64 * 64]; 
    int8_t temp_b[64 * 64];
    
    /* Center crop offsets - ensure even alignment for BAYER */
    uint32_t start_x = ((CAMERA_SIZE_X - IMAGE_SIZE_X) / 2) & ~1;
    uint32_t start_y = ((CAMERA_SIZE_Y - IMAGE_SIZE_Y) / 2) & ~1;
    
    /* Safety bounds check */
    if (start_x + IMAGE_SIZE_X >= CAMERA_SIZE_X || start_y + IMAGE_SIZE_Y >= CAMERA_SIZE_Y) {
        printf("ERROR: Image size too large for camera resolution\n");
        return;
    }
    
    int pixel_idx = 0;
    
    /* Get fresh camera frame data */
    uint8_t *frame_buffer;
    uint32_t imgLen;
    uint32_t w, h;
    
    camera_get_image(&frame_buffer, &imgLen, &w, &h);
    printf("  Image buffer: %dx%d, %d bytes, first bytes: %02x %02x %02x %02x\n", 
           w, h, imgLen, frame_buffer[0], frame_buffer[1], frame_buffer[2], frame_buffer[3]);

    uint8_t *raw = frame_buffer;
    
    /* Process center crop region with bounds checking */
    for (uint32_t y = start_y; y < start_y + IMAGE_SIZE_Y; y++) {
        for (uint32_t x = start_x; x < start_x + IMAGE_SIZE_X; x++) {
            uint32_t pos = y * w + x;
            uint8_t r, g, b;
            
            /* Simplified BAYER processing - just use the pixel value directly for now */
            if (pos < imgLen) {
                /* For debugging, treat as grayscale initially */
                uint8_t gray = raw[pos];
                r = gray;
                g = gray; 
                b = gray;
            } else {
                r = g = b = 128; /* Default middle value */
            }
            
            /* Normalize to [-128,+127] range */
            temp_r[pixel_idx] = (int8_t)(r - 128);
            temp_g[pixel_idx] = (int8_t)(g - 128);
            temp_b[pixel_idx] = (int8_t)(b - 128);
            pixel_idx++;
        }
    }
    
    /* Debug: Print some sample values from CURRENT frame */
    printf("  Sample pixels (R,G,B): (%d,%d,%d) (%d,%d,%d) (%d,%d,%d)\n",
           temp_r[0], temp_g[0], temp_b[0],
           temp_r[100], temp_g[100], temp_b[100],
           temp_r[1000], temp_g[1000], temp_b[1000]);

    /* Convert byte buffers to uint32_t for FIFO (4 pixels per word) */
    memcpy((void *)ch_r, (void *)temp_r, sizeof(temp_r));
    memcpy((void *)ch_g, (void *)temp_g, sizeof(temp_g));
    memcpy((void *)ch_b, (void *)temp_b, sizeof(temp_b));
}

/*******************************************************************************
 * Helper: Stream preprocessed image to CNN via FIFO
 * This function is called AFTER cnn_start() - data streams during inference
 ******************************************************************************/
static void load_input_fifo(void)
{
    int i;
    const uint32_t *in0 = ch_r;  /* R channel */
    const uint32_t *in1 = ch_g;  /* G channel */
    const uint32_t *in2 = ch_b;  /* B channel */

    printf("  Loading FIFO: first R,G,B words = 0x%08lx, 0x%08lx, 0x%08lx\n", 
           in0[0], in1[0], in2[0]);

    /* Calculate word count dynamically based on IMAGE_SIZE */
    /* 64x64 = 4096 bytes / 4 = 1024 words */
    /* 32x32 = 1024 bytes / 4 = 256 words */
    uint32_t word_count = (IMAGE_SIZE_X * IMAGE_SIZE_Y) / 4;
    
    printf("  Streaming %lu words per channel (image: %dx%d)\n", 
           word_count, IMAGE_SIZE_X, IMAGE_SIZE_Y);
    
    for (i = 0; i < word_count; i++) {
        /* Wait for FIFO 0 (R channel) to have space */
        while ((*FIFO_STATUS_REG & FIFO_0_FULL) != 0);
        *FIFO_0_WRITE = *in0++;
        
        /* Wait for FIFO 1 (G channel) to have space */
        while ((*FIFO_STATUS_REG & FIFO_1_FULL) != 0);
        *FIFO_1_WRITE = *in1++;
        
        /* Wait for FIFO 2 (B channel) to have space */
        while ((*FIFO_STATUS_REG & FIFO_2_FULL) != 0);
        *FIFO_2_WRITE = *in2++;
    }
    
    printf("  FIFO loading complete\n");
}

/*******************************************************************************
 * Helper: Get top-3 predictions from CNN output
 ******************************************************************************/
static void get_top3_predictions(int *idx, int *prob)
{
    int32_t top3_val[3] = {INT32_MIN, INT32_MIN, INT32_MIN};
    idx[0] = idx[1] = idx[2] = 0;
    
    for (int i = 0; i < NUM_CLASSES; i++) {
        if (ml_data[i] > top3_val[0]) {
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
    
    prob[0] = (100 * ml_softmax[idx[0]]) / 32768;
    prob[1] = (100 * ml_softmax[idx[1]]) / 32768;
    prob[2] = (100 * ml_softmax[idx[2]]) / 32768;
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
  Bias memory:   700 bytes out of 2,048 bytes total (34.2%)
*/


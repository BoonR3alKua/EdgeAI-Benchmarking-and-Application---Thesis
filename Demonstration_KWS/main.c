/**
 * @file    main.c
 * @brief   Keyword Spotting Demo for MAX78000 FTHR Board
 */

#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include "mxc_sys.h"
#include "icc.h"
#include "mxc_device.h"
#include "mxc_delay.h"
#include "nvic_table.h"
#include "i2s_regs.h"
#include "board.h"
#include "mxc.h"
#include "i2s.h"
#include "tmr.h"
#include "led.h"
#include "pb.h"
#include "cnn.h"

/* **** Definitions **** */
#define SAMPLE_RATE 16000
#define EXT_I2S_FREQ 12288000
#define CON_BAUD 115200

/* CNN/Audio Parameters - Do not change */
#define SAMPLE_SIZE 16384
#define CHUNK 128
#define TRANSPOSE_WIDTH 128
#define NUM_OUTPUTS CNN_NUM_OUTPUTS
#define I2S_RX_BUFFER_SIZE 64

/* Adjustable Parameters */
#define SAMPLE_SCALE_FACTOR 4
#define THRESHOLD_HIGH 350
#define THRESHOLD_LOW 100
#define SILENCE_COUNTER_THRESHOLD 20
#define PREAMBLE_SIZE (30 * CHUNK)
#define INFERENCE_THRESHOLD 91

/* Debug Printing */
#define PR_DEBUG(fmt, args...) printf(fmt, ##args)

/* **** Globals **** */
volatile uint32_t cnn_time;

static int8_t micBuff[SAMPLE_SIZE];
static int micBufIndex = 0;
static int utteranceIndex = 0;
static uint16_t utteranceAvg = 0;
static int zeroPad = 0;

static int32_t ml_data[NUM_OUTPUTS];
static q15_t ml_softmax[NUM_OUTPUTS];
static uint8_t soundBuffer[SAMPLE_SIZE];

static int16_t Max, Min;
static uint16_t thresholdHigh = THRESHOLD_HIGH;
static uint16_t thresholdLow = THRESHOLD_LOW;

static volatile uint8_t i2s_flag = 0;
static int32_t i2s_rx_buffer[I2S_RX_BUFFER_SIZE];

/* Statistics tracking for thesis */
static uint32_t total_inference_time = 0;
static uint32_t min_inference_time = 0xFFFFFFFF;
static uint32_t max_inference_time = 0;
static uint16_t recognized_count = 0;
static uint16_t unknown_count = 0;

/* Processing States */
typedef enum {
    STOP = 0,
    SILENCE = 1,
    KEYWORD = 2
} mic_processing_state;

/* Keyword Labels */
static const char *keywords[NUM_OUTPUTS] = {
    "UP", "DOWN", "LEFT", "RIGHT", "STOP", "GO", "YES", "NO", "ON", "OFF",
    "ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT", "NINE", "ZERO",
    "Unknown"
};

/* Function Prototypes */
static void fail(void);
static uint8_t cnn_load_data(uint8_t *inAud);
static uint8_t MicReadChunk(uint16_t *avg);
static uint8_t AddTranspose(uint8_t *inAud, uint8_t *outAud, uint16_t inSize, uint16_t outSize, uint16_t width);
static uint8_t check_inference(q15_t *ml_soft, int32_t *ml_data, int16_t *out_class, double *out_prob);
static void I2SInit(void);
static void HPF_init(void);
static int16_t HPF(int16_t input);

/* I2S Interrupt Handler */
void i2s_isr(void)
{
    i2s_flag = 1;
    MXC_I2S_ClearFlags(MXC_F_I2S_INTFL_RX_THD_CH0);
}

/* Main Function */
int main(void)
{
    uint32_t sampleCounter = 0;
    mxc_tmr_unit_t units;
    uint8_t pChunkBuff[CHUNK];
    uint16_t avg = 0;
    uint16_t nCounter = 0;
    uint16_t wordCounter = 0;
    uint16_t avgSilenceCounter = 0;
    mic_processing_state procState = STOP;

    /* Wait for PMIC 1.8V to become available (~180ms after power up) */
    MXC_Delay(200000);

    /* Enable cache */
    MXC_ICC_Enable(MXC_ICC0);

    /* Use IPO clock source */
    MXC_SYS_ClockSourceEnable(MXC_SYS_CLOCK_IPO);
    MXC_SYS_Clock_Select(MXC_SYS_CLOCK_IPO);
    SystemCoreClockUpdate();

    /* Enable microphone power */
    Microphone_Power(POWER_ON);

    /* Enable CNN: 50 MHz div 1 */
    cnn_enable(MXC_S_GCR_PCLKDIV_CNNCLKSEL_PCLK, MXC_S_GCR_PCLKDIV_CNNCLKDIV_DIV1);
    cnn_boost_enable(MXC_GPIO2, MXC_GPIO_PIN_5);

    printf("\n*** Keyword Spotting Demonstration ***\n");
    printf("========================================\n");
    printf("System Configuration:\n");
    printf("  CPU Clock:     %d MHz\n", SystemCoreClock / 1000000);
    printf("  CNN Clock:     50 MHz\n");
    printf("  Sample Rate:   %d Hz\n", SAMPLE_RATE);
    printf("  Sample Size:   %d bytes (%.2f sec)\n", SAMPLE_SIZE, (float)SAMPLE_SIZE / SAMPLE_RATE);
    printf("  Preamble:      %d bytes\n", PREAMBLE_SIZE);
    printf("  Num Classes:   %d\n", NUM_OUTPUTS);
    printf("  Threshold:     High=%d, Low=%d\n", THRESHOLD_HIGH, THRESHOLD_LOW);
    printf("  Confidence:    %d%%\n", INFERENCE_THRESHOLD);
    printf("========================================\n");
    
    memset(soundBuffer, 0, sizeof(soundBuffer));

    /* Initialize CNN */
    cnn_init();
    cnn_load_weights();
    cnn_configure();

    /* Disable CNN clock until needed */
    MXC_SYS_ClockDisable(MXC_SYS_PERIPH_CLOCK_CNN);

    /* Initialize I2S for microphone */
    I2SInit();

    /* Switch to silence detection state */
    procState = SILENCE;

    MXC_Delay(SEC(2));
    printf("*** READY ***\n");

    /* Main processing loop */
    while (1) {
        /* Read chunk from microphone */
        if (MicReadChunk(&avg) == 0) {
            continue;
        }

        sampleCounter += CHUNK;

        /* Wait for preamble samples */
        if (sampleCounter < PREAMBLE_SIZE)
            continue;

        /* Silence detection state */
        if (procState == SILENCE) {
            if (avg >= thresholdHigh) {
                procState = KEYWORD;
                utteranceAvg = avg;
                utteranceIndex = micBufIndex;
                nCounter += PREAMBLE_SIZE;
                continue;
            }
        }
        /* Keyword collection state */
        else if (procState == KEYWORD) {
            uint8_t ret = 0;
            nCounter += CHUNK;

            /* Detect silence after keyword */
            if ((avg < thresholdLow) && (nCounter >= SAMPLE_SIZE / 3)) {
                avgSilenceCounter++;
            } else {
                avgSilenceCounter = 0;
            }

            /* Zero-pad if silence detected */
            if (avgSilenceCounter > SILENCE_COUNTER_THRESHOLD) {
                memset(pChunkBuff, 0, CHUNK);
                zeroPad = SAMPLE_SIZE - nCounter;
                nCounter = SAMPLE_SIZE;
            }

            /* Run inference when enough samples collected */
            if (nCounter >= SAMPLE_SIZE) {
                int16_t out_class = -1;
                double probability = 0;
                int endIndex = (utteranceIndex + SAMPLE_SIZE - PREAMBLE_SIZE - zeroPad) % SAMPLE_SIZE;

                memset(pChunkBuff, 0, CHUNK);

                /* Reset transpose state before new word */
                AddTranspose(NULL, NULL, 0, 0, 0);

                /* Copy preamble with transpose */
                if (utteranceIndex - PREAMBLE_SIZE >= 0) {
                    if (AddTranspose((uint8_t *)&micBuff[utteranceIndex - PREAMBLE_SIZE],
                                 soundBuffer, PREAMBLE_SIZE, SAMPLE_SIZE, TRANSPOSE_WIDTH)) {
                        printf("ERROR: Transpose ended early (preamble)\n");
                    }
                } else {
                    /* Copy oldest samples to the beginning */
                    if (AddTranspose((uint8_t *)&micBuff[SAMPLE_SIZE - PREAMBLE_SIZE + utteranceIndex],
                                 soundBuffer, PREAMBLE_SIZE - utteranceIndex, SAMPLE_SIZE, TRANSPOSE_WIDTH)) {
                        printf("ERROR: Transpose ended early (preamble wrap 1)\n");
                    }
                    /* Copy latest samples afterwards */
                    if (AddTranspose((uint8_t *)&micBuff[0], soundBuffer, utteranceIndex,
                                 SAMPLE_SIZE, TRANSPOSE_WIDTH)) {
                        printf("ERROR: Transpose ended early (preamble wrap 2)\n");
                    }
                }

                /* Copy utterance with transpose */
                if (utteranceIndex < endIndex) {
                    if (AddTranspose((uint8_t *)&micBuff[utteranceIndex], soundBuffer,
                                 endIndex - utteranceIndex, SAMPLE_SIZE, TRANSPOSE_WIDTH)) {
                        printf("ERROR: Transpose ended early (utterance)\n");
                    }
                    /* Copy zero padding */
                    while (!ret) {
                        ret = AddTranspose(pChunkBuff, soundBuffer, CHUNK, SAMPLE_SIZE, TRANSPOSE_WIDTH);
                    }
                } else {
                    /* Copy from utterance to end */
                    if (AddTranspose((uint8_t *)&micBuff[utteranceIndex], soundBuffer,
                                 SAMPLE_SIZE - utteranceIndex, SAMPLE_SIZE, TRANSPOSE_WIDTH)) {
                        printf("ERROR: Transpose ended early (utterance wrap 1)\n");
                    }
                    /* Copy from beginning */
                    if (AddTranspose((uint8_t *)&micBuff[0], soundBuffer, endIndex, SAMPLE_SIZE, TRANSPOSE_WIDTH)) {
                        printf("ERROR: Transpose ended early (utterance wrap 2)\n");
                    }
                    /* Copy zero padding */
                    while (!ret) {
                        ret = AddTranspose(pChunkBuff, soundBuffer, CHUNK, SAMPLE_SIZE, TRANSPOSE_WIDTH);
                    }
                }

                /* Reset counters */
                nCounter = 0;
                avgSilenceCounter = 0;
                zeroPad = 0;
                wordCounter++;
                procState = SILENCE;

                if (ret != 1) {
                    fail();
                }

                /* Timing variables for detailed analysis */
                uint32_t load_time, inference_time, unload_time, softmax_time;

                /* Run CNN inference */
                MXC_SYS_ClockEnable(MXC_SYS_PERIPH_CLOCK_CNN);

                /* Measure data loading time */
                MXC_TMR_SW_Start(MXC_TMR1);
                if (!cnn_load_data(soundBuffer)) {
                    fail();
                }
                load_time = MXC_TMR_SW_Stop(MXC_TMR1);

                if (!cnn_start()) {
                    fail();
                }

                /* Wait for CNN to complete */
                while (cnn_time == 0) {
                    __WFI();
                }
                inference_time = cnn_time;

                /* Measure unload time */
                MXC_TMR_SW_Start(MXC_TMR1);
                cnn_unload((uint32_t *)ml_data);
                unload_time = MXC_TMR_SW_Stop(MXC_TMR1);
                
                cnn_stop();
                MXC_SYS_ClockDisable(MXC_SYS_PERIPH_CLOCK_CNN);

                /* Measure softmax time */
                MXC_TMR_SW_Start(MXC_TMR1);
                softmax_q17p14_q15((const q31_t *)ml_data, NUM_OUTPUTS, ml_softmax);
                softmax_time = MXC_TMR_SW_Stop(MXC_TMR1);
                
                ret = check_inference(ml_softmax, ml_data, &out_class, &probability);

                /* Print detailed results for thesis */
                printf("\n=========== Inference #%d ===========\n", wordCounter);
                printf("Timing Breakdown (clock cycles @ 100MHz):\n");
                printf("  Data Load:    %u cycles (%u us)\n", load_time, load_time / 100);
                printf("  CNN Inference:%u cycles (%u us)\n", inference_time, inference_time / 100);
                printf("  Data Unload:  %u cycles (%u us)\n", unload_time, unload_time / 100);
                printf("  Softmax:      %u cycles (%u us)\n", softmax_time, softmax_time / 100);
                printf("  Total:        %u us\n", (load_time + inference_time + unload_time + softmax_time) / 100);
                printf("\nAudio Stats - Min: %d, Max: %d\n", Min, Max);
                printf("Utterance Avg: %d (Threshold: %d)\n", utteranceAvg, thresholdHigh);
                printf("Zero Padding: %d samples\n", zeroPad);
                
                /* Print top-3 predictions */
                printf("\nTop Predictions:\n");
                int32_t temp_data[NUM_OUTPUTS];
                memcpy(temp_data, ml_data, sizeof(int32_t) * NUM_OUTPUTS);
                for (int top = 0; top < 3; top++) {
                    int32_t max_val = 1 << 31;
                    int max_idx = -1;
                    for (int i = 0; i < NUM_OUTPUTS; i++) {
                        if (temp_data[i] > max_val) {
                            max_val = temp_data[i];
                            max_idx = i;
                        }
                    }
                    if (max_idx >= 0) {
                        double prob = 100.0 * ml_softmax[max_idx] / 32768.0;
                        printf("  %d. %s: %.2f%%\n", top + 1, keywords[max_idx], prob);
                        temp_data[max_idx] = 1 << 31;  // Mark as used
                    }
                }
                
                /* Final detection result */
                printf("\n>>> RESULT: ");
                if (!ret || out_class == NUM_OUTPUTS - 1) {
                    printf("Unknown (below %.0f%% threshold)\n", (double)INFERENCE_THRESHOLD);
                    unknown_count++;
                } else {
                    printf("%s (%.1f%%)\n", keywords[out_class], probability);
                    recognized_count++;
                }
                
                /* Update statistics */
                uint32_t total_time = load_time + inference_time + unload_time + softmax_time;
                total_inference_time += total_time;
                if (total_time < min_inference_time) min_inference_time = total_time;
                if (total_time > max_inference_time) max_inference_time = total_time;
                
                printf("\nSession Stats: %d recognized, %d unknown, avg %u us/inference\n",
                       recognized_count, unknown_count, 
                       wordCounter > 0 ? (total_inference_time / wordCounter) / 100 : 0);
                printf("=====================================\n");

                Max = 0;
                Min = 0;

                /* Clear buffer for next utterance */
                memset(micBuff, 0, SAMPLE_SIZE);
                micBufIndex = 0;
            }
        }

        /* Stop if button pressed */
        if (PB_Get(0)) {
            printf("Stopped.\n");
            procState = STOP;
            break;
        }
    }

    LED_Off(LED2);
    printf("\n============ SESSION SUMMARY ============\n");
    printf("Total Samples: %d\n", sampleCounter);
    printf("Total Inferences: %d\n", wordCounter);
    printf("  Recognized: %d (%.1f%%)\n", recognized_count, 
           wordCounter > 0 ? 100.0 * recognized_count / wordCounter : 0);
    printf("  Unknown: %d (%.1f%%)\n", unknown_count,
           wordCounter > 0 ? 100.0 * unknown_count / wordCounter : 0);
    printf("Timing Statistics:\n");
    printf("  Min: %u us\n", min_inference_time / 100);
    printf("  Max: %u us\n", max_inference_time / 100);
    printf("  Avg: %u us\n", wordCounter > 0 ? (total_inference_time / wordCounter) / 100 : 0);
    printf("=========================================\n");

    while (1) {}
}

/* I2S Initialization */
static void I2SInit(void)
{
    mxc_i2s_req_t req;
    int32_t err;

    HPF_init();
    memset(i2s_rx_buffer, 0, sizeof(i2s_rx_buffer));

    req.wordSize = MXC_I2S_WSIZE_WORD;
    req.sampleSize = MXC_I2S_SAMPLESIZE_THIRTYTWO;
    req.bitsWord = 32;
    req.adjust = MXC_I2S_ADJUST_LEFT;
    req.justify = MXC_I2S_MSB_JUSTIFY;
    req.wsPolarity = MXC_I2S_POL_NORMAL;
    req.channelMode = MXC_I2S_INTERNAL_SCK_WS_0;
    req.stereoMode = MXC_I2S_MONO_LEFT_CH;
    req.bitOrder = MXC_I2S_MSB_FIRST;
    req.clkdiv = MXC_I2S_CalculateClockDiv(SAMPLE_RATE, req.wordSize, EXT_I2S_FREQ);
    req.rawData = NULL;
    req.txData = NULL;
    req.rxData = i2s_rx_buffer;
    req.length = I2S_RX_BUFFER_SIZE;

    if ((err = MXC_I2S_Init(&req)) != E_NO_ERROR) {
        printf("I2S Init Error: %d\n", err);
        while (1) {}
    }

    MXC_I2S_SetRXThreshold(4);
    
    /* Register I2S interrupt handler */
    MXC_NVIC_SetVector(I2S_IRQn, i2s_isr);
    NVIC_EnableIRQ(I2S_IRQn);
    
    /* Enable I2S RX threshold interrupt */
    MXC_I2S_EnableInt(MXC_F_I2S_INTEN_RX_THD_CH0);
    
    MXC_I2S_RXEnable();
    __enable_irq();
}

/* Check inference result */
static uint8_t check_inference(q15_t *ml_soft, int32_t *ml_data, int16_t *out_class, double *out_prob)
{
    int32_t temp[NUM_OUTPUTS];
    q15_t max = 0;
    int32_t max_ml = 1 << 31;
    int16_t max_index = -1;

    memcpy(temp, ml_data, sizeof(int32_t) * NUM_OUTPUTS);

    /* Find class with highest probability */
    for (int i = 0; i < NUM_OUTPUTS; i++) {
        if ((int32_t)temp[i] > max_ml) {
            max_ml = (int32_t)temp[i];
            max = ml_soft[i];
            max_index = i;
        }
    }

    *out_class = max_index;
    *out_prob = 100.0 * max / 32768.0;

    return (*out_prob > INFERENCE_THRESHOLD) ? 1 : 0;
}

/* Fail handler */
static void fail(void)
{
    printf("*** FAIL ***\n");
    while (1) {}
}
/* Load data to CNN memory */
static uint8_t cnn_load_data(uint8_t *inAud)
{
    uint32_t mem;
    uint16_t index = 0;

    /* inAud is 16KB, each 1KB belongs to a memory group */
    for (mem = 0x50400000; mem <= 0x50418000; mem += 0x8000) {
        memcpy((uint8_t *)mem, &inAud[index], 1024);
        index += 1024;
    }
    for (mem = 0x50800000; mem <= 0x50818000; mem += 0x8000) {
        memcpy((uint8_t *)mem, &inAud[index], 1024);
        index += 1024;
    }
    for (mem = 0x50C00000; mem <= 0x50C18000; mem += 0x8000) {
        memcpy((uint8_t *)mem, &inAud[index], 1024);
        index += 1024;
    }
    for (mem = 0x51000000; mem <= 0x51018000; mem += 0x8000) {
        memcpy((uint8_t *)mem, &inAud[index], 1024);
        index += 1024;
    }

    return CNN_OK;
}

/* Transpose audio data for CNN input format */
static uint8_t AddTranspose(uint8_t *inAud, uint8_t *outAud, uint16_t inSize, uint16_t outSize, uint16_t width)
{
    static uint16_t row = 0, col = 0, total = 0;
    uint16_t secondHalf, wordRow, byteInWord, group, index;

    /* Reset static variables if inAud is NULL (reset signal) */
    if (inAud == NULL) {
        row = 0;
        col = 0;
        total = 0;
        return 0;
    }

    for (int i = 0; i < inSize; i++) {
        secondHalf = (col >= (width >> 1)) ? 1 : 0;
        group = (col % (width >> 1)) / 4;
        wordRow = secondHalf + (row << 1);
        byteInWord = col % 4;
        index = 1024 * group + 4 * wordRow + byteInWord;

        outAud[index] = inAud[i];
        total++;
        col++;

        if (col >= width) {
            col = 0;
            row++;
        }
    }

    if (total >= outSize) {
        total = 0;
        row = 0;
        col = 0;
        return 1;
    }
    return 0;
}
/* Read chunk from microphone */
static uint8_t MicReadChunk(uint16_t *avg)
{
    static uint16_t chunkCount = 0;
    static uint16_t sum = 0;
    static uint32_t index = 0;
    int32_t sample;
    int16_t temp;
    uint32_t rx_size;

    if (!i2s_flag) {
        *avg = 0;
        return 0;
    }

    i2s_flag = 0;
    rx_size = MXC_I2S->dmach0 >> MXC_F_I2S_DMACH0_RX_LVL_POS;

    while ((rx_size--) && (chunkCount < CHUNK)) {
        sample = (int32_t)MXC_I2S->fifoch0;
        temp = sample >> 14;
        sample = HPF((int16_t)temp);

        /* Discard first 10k samples (mic warmup) */
        if (index++ < 10000) {
            continue;
        }

        LED_On(LED2);

        /* Absolute value for averaging */
        if (sample >= 0) {
            sum += sample;
        } else {
            sum -= sample;
        }

        /* Convert to 8-bit */
        micBuff[micBufIndex] = (sample) * SAMPLE_SCALE_FACTOR / 256;
        chunkCount++;

        temp = (int8_t)micBuff[micBufIndex];
        if (temp > Max) Max = temp;
        if (temp < Min) Min = temp;

        micBufIndex = (micBufIndex + 1) % SAMPLE_SIZE;
    }

    if (chunkCount < CHUNK) {
        *avg = 0;
        return 0;
    }

    *avg = (uint16_t)(sum / CHUNK);
    chunkCount = 0;
    sum = 0;
    return 1;
}

/* High-pass filter state */
static int16_t x_0, x_1, Coeff;
static int32_t y_0, y_1;

/* Initialize high-pass filter (100 Hz cutoff) */
static void HPF_init(void)
{
    Coeff = 32604;  // 0.995 * 2^15
    x_0 = 0;
    y_0 = 0;
    y_1 = 0;
    x_1 = 0;
}

/* High-pass filter: y(n) = x(n) - x(n-1) + A*y(n-1) */
static int16_t HPF(int16_t input)
{
    int32_t tmp;
    int16_t Acc;

    x_0 = input;
    tmp = (Coeff * y_1);
    Acc = (int16_t)((tmp + (1 << 14)) >> 15);
    y_0 = x_0 - x_1 + Acc;

    /* Clipping */
    if (y_0 > 32767) y_0 = 32767;
    if (y_0 < -32768) y_0 = -32768;

    y_1 = y_0;
    x_1 = x_0;

    return (int16_t)y_0;
}

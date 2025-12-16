/**
 * @file    energy_measurement_helper.h
 * @brief   Energy measurement helper functions for MAX78000FTHR
 * @details Use GPIO markers and timing for energy profiling
 */

#ifndef ENERGY_MEASUREMENT_HELPER_H
#define ENERGY_MEASUREMENT_HELPER_H

#include "mxc.h"
#include "gpio.h"

/* GPIO Marker Pins (compatible with Power Monitor on EV Kit) */
#define ENERGY_GPIO_PORT    MXC_GPIO0
#define ENERGY_MARKER_1     MXC_GPIO_PIN_2  /* P0.2 - LED1 (CNN start) */
#define ENERGY_MARKER_2     MXC_GPIO_PIN_3  /* P0.3 - LED2 (CNN complete) */

/* Measurement states */
typedef enum {
    MEASURE_IDLE = 0,
    MEASURE_WEIGHT_LOAD,
    MEASURE_INPUT_LOAD,
    MEASURE_CNN_INFERENCE,
    MEASURE_OUTPUT_UNLOAD,
    MEASURE_ARM_PROCESS,
    MEASURE_RISCV_PROCESS,
    MEASURE_TOTAL_CYCLE
} measurement_phase_t;

/* Timing structure */
typedef struct {
    uint32_t start_time;
    uint32_t end_time;
    uint32_t duration_us;
    measurement_phase_t phase;
} energy_measurement_t;

/* Global measurement data */
extern energy_measurement_t measurements[8];
extern uint8_t measurement_index;

/**
 * @brief Initialize GPIO pins for energy measurement markers
 */
static inline void energy_measurement_init(void) {
    mxc_gpio_cfg_t gpio_cfg;
    
    gpio_cfg.port = ENERGY_GPIO_PORT;
    gpio_cfg.mask = ENERGY_MARKER_1 | ENERGY_MARKER_2;
    gpio_cfg.pad = MXC_GPIO_PAD_NONE;
    gpio_cfg.func = MXC_GPIO_FUNC_OUT;
    gpio_cfg.vssel = MXC_GPIO_VSSEL_VDDIO;
    
    MXC_GPIO_Config(&gpio_cfg);
    MXC_GPIO_OutClr(ENERGY_GPIO_PORT, ENERGY_MARKER_1 | ENERGY_MARKER_2);
}

/**
 * @brief Start measurement window
 * @param phase Measurement phase identifier
 */
static inline void energy_start_measurement(measurement_phase_t phase) {
    MXC_GPIO_OutSet(ENERGY_GPIO_PORT, ENERGY_MARKER_1);
    
    if (measurement_index < 8) {
        measurements[measurement_index].phase = phase;
        measurements[measurement_index].start_time = MXC_TMR_GetCount(MXC_TMR0);
    }
}

/**
 * @brief Stop measurement window
 */
static inline void energy_stop_measurement(void) {
    if (measurement_index < 8) {
        measurements[measurement_index].end_time = MXC_TMR_GetCount(MXC_TMR0);
        measurements[measurement_index].duration_us = 
            (measurements[measurement_index].end_time - 
             measurements[measurement_index].start_time);
        measurement_index++;
    }
    
    MXC_GPIO_OutClr(ENERGY_GPIO_PORT, ENERGY_MARKER_1);
}

/**
 * @brief Mark CNN completion
 */
static inline void energy_mark_cnn_complete(void) {
    MXC_GPIO_OutSet(ENERGY_GPIO_PORT, ENERGY_MARKER_2);
    MXC_Delay(MXC_DELAY_USEC(10));  /* Brief pulse */
    MXC_GPIO_OutClr(ENERGY_GPIO_PORT, ENERGY_MARKER_2);
}

/**
 * @brief Print measurement results
 */
static inline void energy_print_results(void) {
    const char* phase_names[] = {
        "IDLE", "WEIGHT_LOAD", "INPUT_LOAD", "CNN_INFERENCE",
        "OUTPUT_UNLOAD", "ARM_PROCESS", "RISCV_PROCESS", "TOTAL_CYCLE"
    };
    
    printf("\n=== ENERGY MEASUREMENT RESULTS ===\n");
    for (uint8_t i = 0; i < measurement_index; i++) {
        printf("Phase: %-15s | Duration: %lu us (%.3f ms)\n",
               phase_names[measurements[i].phase],
               measurements[i].duration_us,
               measurements[i].duration_us / 1000.0f);
    }
    printf("===================================\n\n");
}

/**
 * @brief Reset measurement counters
 */
static inline void energy_reset_measurements(void) {
    measurement_index = 0;
    memset(measurements, 0, sizeof(measurements));
}

/**
 * @brief Configure timer for microsecond timing
 */
static inline void energy_timer_init(void) {
    mxc_tmr_cfg_t tmr_cfg;
    
    tmr_cfg.pres = TMR_PRES_1;
    tmr_cfg.mode = TMR_MODE_CONTINUOUS;
    tmr_cfg.bitMode = TMR_BIT_MODE_32;
    tmr_cfg.clock = MXC_TMR_APB_CLK;
    tmr_cfg.cmp_cnt = 0xFFFFFFFF;
    tmr_cfg.pol = 0;
    
    MXC_TMR_Init(MXC_TMR0, &tmr_cfg, true);
    MXC_TMR_Start(MXC_TMR0);
}

#endif /* ENERGY_MEASUREMENT_HELPER_H */

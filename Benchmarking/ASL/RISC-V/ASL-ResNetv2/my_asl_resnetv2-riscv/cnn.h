/**************************************************************************************************
* Copyright (C) 2019-2021 Maxim Integrated Products, Inc. All Rights Reserved.
*
* Maxim Integrated Products, Inc. Default Copyright Notice:
* https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
**************************************************************************************************/

/*
 * This header file was automatically @generated for the my_asl_resnetv2-riscv network from a template.
 * Please do not edit; instead, edit the template and regenerate.
 */

#ifndef __CNN_H__
#define __CNN_H__

#include <stdint.h>
typedef int32_t q31_t;
typedef int16_t q15_t;

/* Return codes */
#define CNN_FAIL 0
#define CNN_OK 1

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

/* Number of outputs for this network */
#define CNN_NUM_OUTPUTS 27

/* Use this timer to time the inference */
#define CNN_INFERENCE_TIMER MXC_TMR0

/* Port pin actions used to signal that processing is active */

#define CNN_START LED_On(1)
#define CNN_COMPLETE LED_Off(1)
#define SYS_START LED_On(0)
#define SYS_COMPLETE LED_Off(0)

/* Run software SoftMax on unloaded data */
void softmax_q17p14_q15(const q31_t * vec_in, const uint16_t dim_vec, q15_t * p_out);
/* Shift the input, then calculate SoftMax */
void softmax_shift_q17p14_q15(q31_t * vec_in, const uint16_t dim_vec, uint8_t in_shift, q15_t * p_out);

/* Stopwatch - holds the runtime when accelerator finishes */
extern volatile uint32_t cnn_time;

/* Custom memcopy routines used for weights and data */
void memcpy32(uint32_t *dst, const uint32_t *src, int n);
void memcpy32_const(uint32_t *dst, int n);

/* Enable clocks and power to accelerator, enable interrupt */
int cnn_enable(uint32_t clock_source, uint32_t clock_divider);

/* Disable clocks and power to accelerator */
int cnn_disable(void);

/* Perform minimum accelerator initialization so it can be configured */
int cnn_init(void);

/* Configure accelerator for the given network */
int cnn_configure(void);

/* Load accelerator weights */
int cnn_load_weights(void);

/* Verify accelerator weights (debug only) */
int cnn_verify_weights(void);

/* Load accelerator bias values (if needed) */
int cnn_load_bias(void);

/* Start accelerator processing */
int cnn_start(void);

/* Force stop accelerator */
int cnn_stop(void);

/* Continue accelerator after stop */
int cnn_continue(void);

/* Unload results from accelerator */
int cnn_unload(uint32_t *out_buf);

/* Turn on the boost circuit */
int cnn_boost_enable(mxc_gpio_regs_t *port, uint32_t pin);

/* Turn off the boost circuit */
int cnn_boost_disable(mxc_gpio_regs_t *port, uint32_t pin);

#endif // __CNN_H__

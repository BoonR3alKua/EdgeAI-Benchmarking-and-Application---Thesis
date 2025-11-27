/**************************************************************************************************
* Copyright (C) 2019-2021 Maxim Integrated Products, Inc. All Rights Reserved.
*
* Maxim Integrated Products, Inc. Default Copyright Notice:
* https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
**************************************************************************************************/

/*
 * This header file was automatically @generated for the my_cifar100_effnetv2 network from a template.
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

/* Number of outputs for this network */
#define CNN_NUM_OUTPUTS 100

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

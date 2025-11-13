/**************************************************************************************************
* Copyright (C) 2019-2021 Maxim Integrated Products, Inc. All Rights Reserved.
*
* Maxim Integrated Products, Inc. Default Copyright Notice:
* https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
**************************************************************************************************/

/*
 * This header file was automatically @generated for the my_cifar100_mobilenetv2 network from a template.
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
  Hardware: 4,312,320 ops (4,243,712 macc; 67,584 comp; 1,024 add; 0 mul; 0 bitwise)
    Layer 0: 344,064 ops (331,776 macc; 12,288 comp; 0 add; 0 mul; 0 bitwise)
    Layer 1: 1,339,392 ops (1,327,104 macc; 12,288 comp; 0 add; 0 mul; 0 bitwise)
    Layer 2: 12,288 ops (0 macc; 12,288 comp; 0 add; 0 mul; 0 bitwise)
    Layer 3: 79,872 ops (73,728 macc; 6,144 comp; 0 add; 0 mul; 0 bitwise)
    Layer 4: 1,333,248 ops (1,327,104 macc; 6,144 comp; 0 add; 0 mul; 0 bitwise)
    Layer 5: 102,400 ops (98,304 macc; 4,096 comp; 0 add; 0 mul; 0 bitwise)
    Layer 6: 4,096 ops (0 macc; 4,096 comp; 0 add; 0 mul; 0 bitwise)
    Layer 7: 34,816 ops (32,768 macc; 2,048 comp; 0 add; 0 mul; 0 bitwise)
    Layer 8: 591,872 ops (589,824 macc; 2,048 comp; 0 add; 0 mul; 0 bitwise)
    Layer 9: 50,688 ops (49,152 macc; 1,536 comp; 0 add; 0 mul; 0 bitwise)
    Layer 10: 1,536 ops (0 macc; 1,536 comp; 0 add; 0 mul; 0 bitwise)
    Layer 11: 19,200 ops (18,432 macc; 768 comp; 0 add; 0 mul; 0 bitwise)
    Layer 12: 332,544 ops (331,776 macc; 768 comp; 0 add; 0 mul; 0 bitwise)
    Layer 13: 25,088 ops (24,576 macc; 512 comp; 0 add; 0 mul; 0 bitwise)
    Layer 14: 33,792 ops (32,768 macc; 1,024 comp; 0 add; 0 mul; 0 bitwise)
    Layer 15: 1,024 ops (0 macc; 0 comp; 1,024 add; 0 mul; 0 bitwise)
    Layer 16: 6,400 ops (6,400 macc; 0 comp; 0 add; 0 mul; 0 bitwise)

  RESOURCE USAGE
  Weight memory: 49,844 bytes out of 442,368 bytes total (11.3%)
  Bias memory:   468 bytes out of 2,048 bytes total (22.9%)
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

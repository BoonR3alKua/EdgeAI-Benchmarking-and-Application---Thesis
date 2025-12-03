/**************************************************************************************************
* Copyright (C) 2019-2021 Maxim Integrated Products, Inc. All Rights Reserved.
*
* Maxim Integrated Products, Inc. Default Copyright Notice:
* https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
**************************************************************************************************/

/*
 * This header file was automatically @generated for the my_cifar100_mobilenetv2-riscv network from a template.
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

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

// ARM wrapper code
// ASL Recognition Demo using FIFO Streaming
// Modified for camera capture with FIFO-based CNN input
// Target: MAX78000 FTHR_RevA

#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include "mxc.h"
#include "mxc_delay.h"
#include "gcfr_regs.h"
#include "fcr_regs.h"
#include "sema_regs.h"
#include "board.h"
#include "uart.h"

/* UART Configuration */
#define CONSOLE_UART    MXC_UART0
#define CONSOLE_BAUD    115200

extern volatile void const *__FlashStart_; // Defined in linker file

void WakeISR(void)
{
    MXC_SEMA->irq0 = MXC_F_SEMA_IRQ0_EN & ~MXC_F_SEMA_IRQ0_CM4_IRQ;
}

int main(void)
{
    int ret;

    // Wait for PMIC 1.8V to become available (~180ms after power up)
    MXC_Delay(200000);

    MXC_ICC_Enable(MXC_ICC0); // Enable cache

    // Switch to 100 MHz clock
    MXC_SYS_Clock_Select(MXC_SYS_CLOCK_IPO);
    SystemCoreClockUpdate();

    // Initialize UART for console output
    ret = MXC_UART_Init(CONSOLE_UART, CONSOLE_BAUD, MXC_UART_APB_CLK);
    if (ret != E_NO_ERROR) {
        while(1) {} // UART init failed
    }

    // Enable camera power on FTHR_RevA board
    Camera_Power(POWER_ON);
    
    // Wait for camera power to stabilize
    MXC_Delay(MXC_DELAY_MSEC(200));

    // Enable DMA clock (required by camera driver)
    MXC_SYS_ClockEnable(MXC_SYS_PERIPH_CLOCK_DMA);
    
    // Enable PCIF (Parallel Camera Interface) clock
    MXC_SYS_ClockEnable(MXC_SYS_PERIPH_CLOCK_PCIF);

    printf("\n*** ARM Core: ASL FIFO Demo ***\n");
    printf("Camera power enabled, starting RISC-V core...\n");
    fflush(stdout);

    MXC_FCR->urvbootaddr = (uint32_t) &__FlashStart_; // Set RISC-V boot address
    MXC_SYS_ClockEnable(MXC_SYS_PERIPH_CLOCK_SMPHR); // Enable Semaphore clock
    NVIC_EnableIRQ(RISCV_IRQn);
    MXC_NVIC_SetVector(RISCV_IRQn, WakeISR); // Set wakeup ISR

    // Delay for debugger attachment
    for (int i = 0; i < (1 << 25); i++) {}

    // Enable RISC-V clock (direct register access)
    MXC_GCR->pclkdis1 &= ~MXC_F_GCR_PCLKDIS1_CPU1;

    printf("RISC-V core started. ARM entering sleep mode...\n\n");
    fflush(stdout);

    while (1) {
        __WFI(); // Wait For Interrupt - let RISC-V run
    }

    return 0;
}


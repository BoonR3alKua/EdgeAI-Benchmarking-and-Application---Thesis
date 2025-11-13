# Dual UART Setup Guide

## Overview

The MAX78000 has been configured to use **TWO UART ports**:

- **UART0 (COM5)**: Console output - printf messages, results, accuracy
- **UART1 (COM6/7)**: Data input - receives CIFAR-100 images

This allows you to:

1. Monitor real-time results in PuTTY on COM5
2. Send images via Python on COM6/7
3. Both running simultaneously!

## Hardware Setup

### MAX78000 FTHR Board Pinout:

- **UART0 (Console)**:
  - TX: P0.0 (USB Console - COM5)
  - RX: P0.1
- **UART1 (Data)**:
  - TX: P0.10
  - RX: P0.11
  - **⚠️ Needs USB-Serial adapter or second MAX78000 connection**

## Problem: Only One USB-Serial Detected

If you only see COM5, you have two options:

### Option 1: Use USB-UART Adapter (Hardware Solution)

1. Get a USB-to-UART adapter (FTDI, CP2102, etc.)
2. Connect:
   - Adapter TX → MAX78000 P0.11 (UART1 RX)
   - Adapter GND → MAX78000 GND
3. Adapter will show as COM6 or COM7
4. Update sender.py to use the new COM port

### Option 2: Time-Multiplexed Single UART (Software Solution)

Since you only have one USB-Serial port, use it in two phases:

#### Phase 1: Send Data (No PuTTY)

```bash
python sender.py
# Script sends all 1000 images
# Board logs results internally
```

#### Phase 2: View Results (PuTTY)

```bash
# Open PuTTY on COM5 after sender completes
# Board displays all accumulated results
```

## Implementation: Time-Multiplexed Approach

Since you only have COM5 available, let me modify the code to:

1. Buffer results in memory
2. Display all results after test completes
3. You can connect PuTTY after sending finishes

This is the **SIMPLEST** solution without additional hardware!

## Current Status

✅ Code modified to use UART1 for data
✅ UART0 free for console
❌ Need to configure UART1 hardware connection OR
✅ Use time-multiplexed approach (see below)

## Next Steps

Choose ONE approach:

### A. Hardware Approach (Best for real-time monitoring)

1. Get USB-UART adapter
2. Connect to UART1 pins
3. Update sender.py SERIAL_PORT to new COM port
4. Run PuTTY on COM5 while sender uses COM6

### B. Software Approach (No extra hardware needed)

1. Revert to single UART (UART0)
2. Modify code to:
   - Run test with sender
   - Store results in memory
   - Display results after completion
3. Connect PuTTY AFTER sender finishes

**Recommendation**: Use Software Approach (B) since you only have one USB-Serial port!

# MAX78000FTHR Energy Measurement Guide

## Table of Contents

1. [Overview](#overview)
2. [Hardware Setup](#hardware-setup)
3. [Software Instrumentation](#software-instrumentation)
4. [Measurement Procedures](#measurement-procedures)
5. [Data Processing](#data-processing)
6. [Expected Results](#expected-results)

---

## Overview

The MAX78000FTHR board requires **external measurement equipment** since it doesn't have the built-in MAX34417 power monitor found on the EV Kit. This guide covers multiple methods to measure energy consumption for:

- **CNN Accelerator Core**
- **ARM Cortex-M4 Core**
- **RISC-V Coprocessor**
- **Total System Power**

---

## Hardware Setup

### Method 1: USB Power Measurement (Simplest)

#### Equipment Needed:

- USB Power Meter (e.g., UM25C, UM34C) - **$20-40**
- MAX78000FTHR board
- USB cable

#### Setup:

```
Computer USB → [USB Power Meter] → MAX78000FTHR
```

#### Pros:

- Very simple
- No board modification
- Real-time display

#### Cons:

- Measures total system power only
- Cannot separate CNN/ARM/RISC-V
- Limited time resolution

---

### Method 2: Precision Shunt Resistor (Recommended)

#### Equipment Needed:

- 0.1Ω precision shunt resistor (1%, low inductance)
- Oscilloscope or differential probe
- Soldering equipment (for permanent installation)

#### Setup:

**Option A: Total System Measurement**

```
VBUS (USB 5V) → [Shunt 0.1Ω] → MAX20303 PMIC Input
                        ↓
                  Oscilloscope Ch1+
                        ↓
                  Oscilloscope Ch1-
```

**Option B: Individual Rail Measurement (Advanced)**
Requires accessing test points or modifying the board:

- VCORE_A: ARM Cortex-M4 power rail
- VCORE_B: CNN accelerator power rail

#### Measurement:

1. Measure voltage drop across shunt: `V_shunt`
2. Calculate current: `I = V_shunt / R_shunt`
3. Calculate power: `P = I × V_supply`
4. Integrate for energy: `E = ∫ P dt`

#### Oscilloscope Settings:

- **Bandwidth:** 20 MHz minimum
- **Sample rate:** 100 kS/s minimum (1 MS/s preferred)
- **Coupling:** DC
- **Vertical scale:** 50 mV/div (for 0.1Ω shunt)
- **Time scale:** Adjust based on event duration

---

### Method 3: Dedicated Power Analyzer (Best Accuracy)

#### Recommended Equipment:

| Device                    | Price  | Dynamic Range | Accuracy | Features                    |
| ------------------------- | ------ | ------------- | -------- | --------------------------- |
| **Nordic PPK2**           | $100   | 0.2µA - 1A    | ±0.2%    | USB, real-time graph        |
| **Otii Arc**              | $500   | 0µA - 5A      | ±0.1%    | Battery simulation, logging |
| **Monsoon Power Monitor** | $500   | 0µA - 4.5A    | ±0.1%    | High precision, USB         |
| **Keysight N6705**        | $5000+ | nA - A range  | <0.05%   | Lab-grade, 4 channels       |

#### Recommended: Nordic PPK2

**Setup:**

```
PPK2 VOUT → MAX78000FTHR VBUS (or battery input)
PPK2 GND  → MAX78000FTHR GND
PPK2 USB  → Computer (nRF Connect Power Profiler software)
```

**Advantages:**

- Measures current directly
- Software records time-series data
- Can export CSV for analysis
- Shows real-time power graph
- Affordable

---

## Software Instrumentation

### 1. GPIO Timing Markers

The MAX78000 supports GPIO toggles to mark measurement windows. These are compatible with the EV Kit's Power Monitor approach.

#### Pin Assignments:

- **P0.2** (LED1): Start/stop marker (compatible with EV Kit CNN start)
- **P0.3** (LED2): Event marker (compatible with EV Kit CNN complete)
- **P2.5**: CNN boost enable signal

#### Code Integration:

Add to your main.c:

```c
#include "energy_measurement_helper.h"

int main(void) {
    // Initialize measurement system
    energy_measurement_init();
    energy_timer_init();

    // Your existing initialization...

    while (1) {
        // Reset measurements for this cycle
        energy_reset_measurements();

        // Measure weight loading
        energy_start_measurement(MEASURE_WEIGHT_LOAD);
        cnn_init();
        cnn_load_weights();
        cnn_load_bias();
        energy_stop_measurement();

        // Measure input loading
        energy_start_measurement(MEASURE_INPUT_LOAD);
        cnn_load_data(input_data);
        energy_stop_measurement();

        // Measure CNN inference
        energy_start_measurement(MEASURE_CNN_INFERENCE);
        cnn_start();
        while (cnn_time == 0) __WFI();
        energy_mark_cnn_complete();
        energy_stop_measurement();

        // Measure output unload
        energy_start_measurement(MEASURE_OUTPUT_UNLOAD);
        cnn_unload(output);
        energy_stop_measurement();

        // Measure ARM processing (softmax)
        energy_start_measurement(MEASURE_ARM_PROCESS);
        softmax_q17p14_q15((const q31_t *)ml_data, NUM_OUTPUTS, ml_softmax);
        energy_stop_measurement();

        // Print timing results
        energy_print_results();

        MXC_Delay(MXC_DELAY_SEC(1));
    }
}
```

---

### 2. Clock Configuration for Measurement

Different clock settings affect power consumption:

```c
// Maximum performance (highest power)
void configure_max_performance(void) {
    MXC_SYS_Clock_Select(MXC_SYS_CLOCK_IPO);  // 100 MHz
    SystemCoreClockUpdate();
    MXC_SYS_ClockEnable(MXC_SYS_PERIPH_CLOCK_CNN);
}

// Balanced (medium power)
void configure_balanced(void) {
    MXC_SYS_Clock_Select(MXC_SYS_CLOCK_ISO);  // 60 MHz
    SystemCoreClockUpdate();
    MXC_SYS_ClockEnable(MXC_SYS_PERIPH_CLOCK_CNN);
}

// Low power (slowest)
void configure_low_power(void) {
    MXC_SYS_Clock_Select(MXC_SYS_CLOCK_IBRO);  // 7.3728 MHz
    SystemCoreClockUpdate();
}
```

---

### 3. Power Mode Control

```c
// Enter low power mode while waiting for CNN
void wait_for_cnn_low_power(void) {
    // Configure wakeup source
    NVIC_EnableIRQ(CNN_IRQn);

    // Enter sleep mode
    MXC_LP_EnterSleepMode();

    // Wakes up when CNN completes
}

// Deep sleep (LPM) - requires RISC-V or timer wakeup
void enter_lpm(void) {
    MXC_LP_ClearWakeStatus();
    SCB->SCR |= SCB_SCR_SLEEPDEEP_Msk;
    MXC_GCR->pm &= ~MXC_F_GCR_PM_MODE;
    MXC_GCR->pm |= MXC_S_GCR_PM_MODE_LPM;
    __WFI();
}
```

---

## Measurement Procedures

### Procedure 1: Baseline Measurements

#### Step 1: Measure Idle Power

```c
int main(void) {
    // Minimal initialization
    MXC_SYS_Clock_Select(MXC_SYS_CLOCK_IPO);
    SystemCoreClockUpdate();

    while (1) {
        __WFI();  // Sleep, no activity
    }
}
```

**Expected:** 6-8 mW (SLEEP mode)

---

#### Step 2: Measure ARM Active (No CNN)

```c
int main(void) {
    MXC_SYS_Clock_Select(MXC_SYS_CLOCK_IPO);  // 100 MHz
    SystemCoreClockUpdate();

    volatile uint32_t dummy = 0;
    while (1) {
        for (int i = 0; i < 1000; i++) {
            dummy += i * i;  // Keep ARM busy
        }
    }
}
```

**Expected:** 20-25 mW

---

#### Step 3: Measure CNN Active (No ARM Processing)

```c
int main(void) {
    MXC_SYS_Clock_Select(MXC_SYS_CLOCK_IPO);
    SystemCoreClockUpdate();

    cnn_init();
    cnn_load_weights();
    cnn_load_bias();

    while (1) {
        energy_start_measurement(MEASURE_CNN_INFERENCE);

        cnn_load_data(sample_data);
        cnn_start();

        // ARM sleeps during inference
        __WFI();

        cnn_unload(output);
        energy_stop_measurement();

        MXC_Delay(MXC_DELAY_SEC(1));
    }
}
```

**Expected:** 35-40 mW during inference, 0.14-0.6 mJ per inference

---

### Procedure 2: Separate Core Measurements

#### Measuring CNN Core Only:

1. Configure ARM to sleep during CNN operation
2. Use GPIO markers around cnn_start() / cnn_completion
3. Measure power spike during CNN active period
4. Subtract baseline idle power

#### Measuring ARM Core Only:

1. Disable CNN clock: `MXC_SYS_ClockDisable(MXC_SYS_PERIPH_CLOCK_CNN)`
2. Run preprocessing/postprocessing code
3. Measure power during ARM computation

#### Measuring RISC-V Core:

1. Configure RISC-V to drive CNN
2. Put ARM in deep sleep (LPM)
3. Measure total power
4. Subtract CNN contribution

---

### Procedure 3: Complete Inference Cycle

#### Full Measurement Sequence:

```c
// Typical inference cycle
void measure_complete_inference(void) {
    uint32_t total_time_start, total_time_end;

    total_time_start = MXC_TMR_GetCount(MXC_TMR0);
    energy_start_measurement(MEASURE_TOTAL_CYCLE);

    // 1. Load weights (one-time per boot)
    energy_start_measurement(MEASURE_WEIGHT_LOAD);
    cnn_init();
    cnn_load_weights();
    cnn_load_bias();
    energy_stop_measurement();

    // 2. Load input data
    energy_start_measurement(MEASURE_INPUT_LOAD);
    load_input(input_data);
    energy_stop_measurement();

    // 3. CNN inference
    energy_start_measurement(MEASURE_CNN_INFERENCE);
    cnn_start();
    while (cnn_time == 0) __WFI();
    energy_stop_measurement();

    // 4. Unload output
    energy_start_measurement(MEASURE_OUTPUT_UNLOAD);
    cnn_unload(ml_data);
    energy_stop_measurement();

    // 5. Post-processing (softmax)
    energy_start_measurement(MEASURE_ARM_PROCESS);
    softmax_q17p14_q15((const q31_t *)ml_data, NUM_OUTPUTS, ml_softmax);
    int16_t result_class;
    double result_prob;
    check_inference(ml_softmax, ml_data, &result_class, &result_prob);
    energy_stop_measurement();

    energy_stop_measurement();  // Total cycle
    total_time_end = MXC_TMR_GetCount(MXC_TMR0);

    energy_print_results();
}
```

---

## Data Processing

### Using the Python Script

1. **Manual Entry (Multimeter Measurements):**

```bash
python process_power_measurements.py
```

This will show expected values and examples.

2. **From Oscilloscope CSV:**

```python
from process_power_measurements import EnergyAnalyzer

analyzer = EnergyAnalyzer(supply_voltage=3.3, shunt_resistance=0.1)

# Load oscilloscope data
analyzer.load_csv('cnn_inference.csv', 'CNN_Inference')
analyzer.load_csv('arm_processing.csv', 'ARM_Processing')

# Analyze
cnn_results = analyzer.analyze_phase('CNN_Inference')
arm_results = analyzer.analyze_phase('ARM_Processing')

print(f"CNN Energy: {cnn_results['energy_mj']:.3f} mJ")
print(f"ARM Energy: {arm_results['energy_mj']:.3f} mJ")

# Generate plots
analyzer.plot_power_profile('CNN_Inference', 'cnn_power_profile.png')

# Generate report
analyzer.generate_report('energy_measurement_report.txt')
```

3. **From Nordic PPK2:**

- Export data from nRF Connect Power Profiler as CSV
- Load into analyzer:

```python
analyzer.load_csv('ppk2_export.csv', 'Total_Inference')
```

---

## Expected Results

### Typical Power Consumption Values

Based on Analog Devices Application Note AN7417:

#### CNN Inference (ARM @ 100MHz, CNN @ 50MHz):

| Phase             | Duration | Avg Power | Energy       |
| ----------------- | -------- | --------- | ------------ |
| **Weight Load**   | 11.6 ms  | 19.8 mW   | 0.146 mJ     |
| **Input Load**    | 1.1 ms   | 19.3 mW   | 0.013 mJ     |
| **CNN Inference** | 2.0 ms   | 37.5 mW   | 0.140 mJ     |
| **Output Unload** | 0.15 ms  | 10.7 mW   | 0.002 mJ     |
| **ARM Softmax**   | 0.5 ms   | 9.8 mW    | 0.005 mJ     |
| **TOTAL**         | 15.35 ms | -         | **0.306 mJ** |

#### Power Modes (idle):

| Mode                  | Power        |
| --------------------- | ------------ |
| **ACTIVE**            | 20 mW        |
| **SLEEP**             | 7-8 mW       |
| **LPM (Low Power)**   | 3-6 mW       |
| **UPM (Micro Power)** | 0.3-0.4 mW   |
| **STANDBY**           | 0.03-0.05 mW |
| **BACKUP**            | <0.01 mW     |

#### Core Contributions:

From the measurements, you can expect:

- **CNN Core (during inference):** 15-20 mW
- **ARM Core (100 MHz active):** 20-25 mW
- **ARM Core (SLEEP):** 7-8 mW
- **RISC-V Core (60 MHz active):** 12-15 mW
- **RISC-V Core (SLEEP):** <1 mW

---

### Your Specific Models

Based on your workspace logs:

#### ASL (Sign Language):

- **Model:** ResNetv2
- **Expected inference time:** 2-4 ms
- **Expected energy:** 0.15-0.4 mJ per inference

#### CIFAR-10:

- **Model:** AlexNet
- **Expected inference time:** 2-3 ms
- **Expected energy:** 0.14-0.3 mJ per inference

#### KWS20 (Keyword Spotting):

- **Model:** Custom CNN
- **Expected inference time:** 2.0 ms (from your logs: 2597 µs)
- **Expected energy:** 0.14 mJ per inference

---

## Validation Checklist

✅ **Before measuring:**

- [ ] Shunt resistor properly installed
- [ ] Oscilloscope/power analyzer calibrated
- [ ] GPIO markers configured and tested
- [ ] Timer initialized for timing measurements
- [ ] Baseline idle power measured

✅ **During measurement:**

- [ ] Consistent supply voltage (3.3V ± 0.1V)
- [ ] Stable ambient temperature
- [ ] No USB communication during measurement
- [ ] Multiple runs for averaging (10+ samples)

✅ **After measurement:**

- [ ] Data exported and saved
- [ ] Timing markers visible in data
- [ ] Results within expected ranges
- [ ] Calculations verified

---

## Troubleshooting

### Issue: Measured power much higher than expected

- Check for enabled peripherals (UART, I2C, LEDs)
- Verify clock configuration
- Check if CNN clock is disabled when not in use
- Look for busy-wait loops

### Issue: Cannot isolate CNN power

- CNN and ARM share power rail on FTHR board
- Use timing markers + power measurement correlation
- Subtract baseline ARM power during CNN operation

### Issue: Noisy measurements

- Use proper grounding
- Shorten measurement leads
- Add capacitors near measurement point (be careful not to filter too much)
- Increase oscilloscope averaging

---

## References

1. [MAX78000 Datasheet](https://www.analog.com/en/products/max78000.html)
2. [MAX78000FTHR Evaluation Kit](https://www.analog.com/en/resources/evaluation-hardware-and-software/evaluation-boards-kits/max78000fthr.html)
3. [AN7417: Developing Power-Optimized Applications on the MAX78000](https://www.analog.com/en/resources/app-notes/developing-power-optimized-apps-on-the-max78000.html)
4. [MAX78000 User Guide](https://www.analog.com/media/en/technical-documentation/user-guides/max78000-user-guide.pdf)
5. [Nordic PPK2 Power Profiler Kit](https://www.nordicsemi.com/Products/Development-hardware/Power-Profiler-Kit-2)

---

## Summary

**Recommended Approach for Your Thesis:**

1. **Hardware:** Nordic PPK2 (~$100) - best balance of cost and accuracy
2. **Software:** Use the provided `energy_measurement_helper.h` for timing
3. **Data Analysis:** Use the provided Python script
4. **Measurements:**
   - Baseline idle power
   - CNN inference only (ARM sleeping)
   - ARM preprocessing/postprocessing
   - Total system per-inference energy
5. **Report:** Energy per inference (mJ), power breakdown, battery life estimates

This approach gives you:

- Publishable accuracy
- Reproducible results
- Core-specific energy breakdown
- Professional measurement methodology

# EdgeAI Benchmarking and Application

Comprehensive benchmarking and implementation study of edge AI inference on the MAX78000 microcontroller, featuring both ARM Cortex-M4 and RISC-V cores with hardware CNN accelerator.

## Overview

This project evaluates edge AI performance across multiple neural network architectures and real-world applications on the Analog Devices MAX78000 microcontroller. It includes training pipelines, hardware deployment, performance benchmarking, and practical demonstrations of keyword spotting (KWS) and American Sign Language (ASL) recognition.

## Hardware Platform

**MAX78000 Microcontroller**
- Dual-core: ARM Cortex-M4F (100 MHz) + RISC-V (60 MHz)
- Hardware CNN accelerator (50 MHz)
- 512 KB flash, 128 KB SRAM
- Integrated camera interface
- Hardware support: Conv2D, pooling, activation functions

## Project Structure

```
├── Models/                          # Neural network architectures
│   ├── ai85net-simplenet.py        # SimpleNet (381K params)
│   ├── ai85net-mobilenetv2.py      # MobileNetV2 (121K params)
│   ├── ai85net-effnetv2.py         # EfficientNetV2 (280K params)
│   ├── ai85net-alexnet.py          # AlexNet variant
│   ├── ai85net-resnetv2.py         # ResNetV2
│   └── ai85net-kws20.py            # Keyword Spotting (20 words)
│
├── Benchmarking/                    # Performance evaluation
│   ├── CIFAR10/                    # 32x32 and 64x64 benchmarks
│   ├── CIFAR100/                   # 100-class classification
│   ├── ASL/                        # Sign language recognition
│   └── Training Logs.txt           # Training metrics
│
├── Demonstration_KWS/              # Keyword spotting demo
│   ├── main.c                      # Audio processing pipeline
│   ├── cnn.c                       # CNN accelerator driver
│   ├── weights.h                   # Model weights
│   └── Makefile                    # Build configuration
│
├── Demonstration_ASL/              # ASL recognition demo
│   ├── main.c                      # ARM core controller
│   ├── main_riscv.c                # RISC-V core processing
│   ├── cnn.c                       # CNN implementation
│   └── Makefile.ARM / .RISCV       # Dual-core builds
│
├── ExtraARM/                       # ARM-only experiments
├── ExtraRISC-V/                    # RISC-V-only experiments
└── *.txt                           # Execution logs
```

## Neural Network Architectures

### CIFAR-100 Training Results

| Model | Top-1 Acc | Top-5 Acc | Parameters | Inference Time |
|-------|-----------|-----------|------------|----------------|
| **SimpleNet** | 58.2% | 85.1% | 381,792 | 2.60 ms (ARM) |
| **MobileNetV2** | 46.6% | 77.5% | 121,136 | 1.52 ms (ARM) |
| **EfficientNetV2** | 52.5% | 80.7% | 280,136 | — |
| **AlexNet** | — | — | — | 13.39 ms (RISC-V) |
| **ResNetV2** | — | — | — | — |

### Keyword Spotting (KWS20)

- **Dataset**: 20 command words + unknown class
- **Input**: 128-channel MFCC, 16 kHz audio
- **Latency**: **25 µs** per inference
- **Throughput**: 40,000 inferences/second
- **Confidence threshold**: 91%

## Performance Benchmarks

### ARM Cortex-M4 (100 MHz)

#### CIFAR-100 (32×32 RGB)

| Model | Throughput | Core Latency | System Latency | Accuracy |
|-------|------------|--------------|----------------|----------|
| SimpleNet | 385 FPS | 2.60 ms | 3.02 ms | 13.3% (30 samples) |
| MobileNetV2 | 660 FPS | 1.52 ms | 2.02 ms | 6.7% (30 samples) |

**TOPS Analysis**
- SimpleNet: 0.0072 TOPS effective, 12.4% NPU utilization
- MobileNetV2: Higher throughput, lower parameter count

### RISC-V Core (60 MHz)

#### ASL Recognition (64×64 RGB)

- **Model**: ResNetV2
- **Capture + Preprocess**: 77–91 ms
- **CNN Inference**: 3.25 ms
- **Total Latency**: 83–99 ms per frame
- **Recognition Rate**: 65% (13/20 inferences)
- **Confidence Threshold**: 50%

**FIFO Streaming Mode**
- Continuous image streaming via DMA
- AlexNet-Small: 27.04M ops, 13.50 ms inference
- Total pipeline: 81–108 ms (capture + inference)

## Applications

### 1. Keyword Spotting (KWS)

Real-time voice command recognition running on ARM Cortex-M4.

**Features**
- 20 command vocabulary: digits (0-9), directions (up/down/left/right), actions (on/off/stop/go/yes/no)
- Ultra-low latency: **25 µs** per inference
- High confidence filtering (91% threshold)
- 16 kHz audio sampling, 128 MFCC channels
- Zero-padding detection for silence frames

**Performance Metrics** (from KWS Log.txt)
```
Recognition Rate: 50% (5/10 test samples)
Min/Max/Avg Latency: 25 µs constant
Timing Breakdown:
  - Data Load: 10 µs
  - CNN Inference: 14 µs
  - Softmax: <1 µs
```

**Usage**
```bash
cd Demonstration_KWS
make
# Flash to MAX78000 development board
```

### 2. American Sign Language (ASL) Recognition

Real-time camera-based sign language recognition using dual-core architecture.

**Architecture**
- **ARM Core**: Power management, camera control, result display
- **RISC-V Core**: Image preprocessing, CNN inference, classification
- **Input**: 168×120 camera capture → 64×64 CNN input
- **Output**: 27 classes (A-Z letters + Unknown)

**Performance Metrics** (from ASL Log.txt)
```
Recognition Rate: 65% (13/20 inferences)
Total Latency: 83-99 ms per frame
  - Camera Capture: 76-91 ms
  - Preprocessing: 4.7 ms
  - CNN Inference: 3.2 ms
  - Softmax: 56 µs
```

**Dual-Core Operation**
1. ARM enables camera and wakes RISC-V core
2. RISC-V captures frame via DMA channel 0
3. RISC-V performs Bayer → RGB conversion and resizing
4. RISC-V runs CNN inference at 50 MHz
5. Results passed back to ARM for display

**Usage**
```bash
cd Demonstration_ASL
make -f Makefile.ARM    # Build ARM firmware
make -f Makefile.RISCV  # Build RISC-V firmware
# Flash both binaries to MAX78000
```

## Key Findings

### Latency Analysis

1. **Keyword Spotting**: Hardware optimized for 1D → 2D convolutions
   - Consistent 25 µs latency regardless of input
   - Dominated by CNN accelerator (14 µs = 56%)

2. **Image Classification**: Camera I/O dominates total latency
   - Camera capture: 76-91 ms (85-92% of total time)
   - CNN inference: 3.2 ms (3-4% of total time)
   - Preprocessing overhead: 4.7 ms (5%)

3. **Cross-Architecture Comparison**
   - ARM (100 MHz): Better for I/O and control tasks
   - RISC-V (60 MHz): Dedicated to compute-intensive CNN
   - Dual-core reduces main CPU load during inference

### Model Selection Trade-offs

| Criteria | Best Model | Rationale |
|----------|------------|-----------|
| **Accuracy** | SimpleNet | 58.2% top-1 on CIFAR-100 |
| **Speed** | MobileNetV2 | 660 FPS, 1.52 ms latency |
| **Memory** | MobileNetV2 | 121K params (68% smaller) |
| **Balanced** | SimpleNet | Good accuracy + reasonable speed |

### NPU Utilization

- Measured utilization: 12-20% across models
- Bottlenecks: Memory bandwidth, fixed-point quantization
- Overhead breakdown shows initialization costs dwarf inference time for small models

## Technical Implementation

### CNN Accelerator Programming

The MAX78000 CNN accelerator requires careful configuration:

**Weight Loading**
- Quantized to 8-bit fixed point
- Loaded during initialization (8-26 ms depending on model)
- Stored in dedicated accelerator memory (not SRAM)

**Data Flow**
```
Input → CNN SRAM → Conv Layers → Pooling → Output SRAM → Softmax (CPU)
```

**Timing Optimization**
- Use FIFO streaming for continuous inference
- Overlap capture and inference when possible
- Minimize CNN init/config overhead for repeated runs

### Building from Source

**Prerequisites**
- Maxim Micros SDK (MSDK)
- ARM GCC toolchain
- OpenOCD for flashing

**Model Training**
```bash
# Train on CIFAR-100 with quantization-aware training (QAT)
python train.py --model ai85simplenet --dataset CIFAR100 \
                --epochs 250 --qat-policy schedule.yaml
```

**Deployment**
```bash
# Generate CNN configuration and weights
ai8xize.py --model ai85simplenet --checkpoint best.pth.tar \
           --output-dir deployment/

# Build firmware
cd deployment/
make
make flash
```

## Benchmarking Methodology

### Dataset Splits

- **Training**: CIFAR-10/100 full training sets
- **Validation**: Standard validation splits
- **On-Device Testing**: 30-image subsets for latency measurement

### Metrics Collected

1. **Latency**: Per-layer and end-to-end timing
2. **Throughput**: FPS for batch processing
3. **Energy**: Measured via power profiler (not in logs)
4. **Accuracy**: Top-1 and Top-5 on test sets
5. **TOPS**: Theoretical vs. effective computation rate

### Measurement Tools

- Hardware timers (microsecond precision)
- GPIO toggles for oscilloscope verification
- Serial output for result logging

## Logs and Results

Detailed execution logs are available:

- `KWS Log.txt`: Keyword spotting inference results
- `ASL Log.txt`: Sign language recognition session
- `FIFO Log.txt`: FIFO streaming mode performance
- `Benchmarking/*.txt`: Cross-model comparisons
- `ExtraARM/log.txt`, `ExtraRISC-V/log.txt`: Core-specific tests

## Limitations and Future Work

### Current Limitations

1. **Accuracy Gap**: On-device accuracy lower than training accuracy
   - Likely due to quantization effects
   - Input preprocessing differences

2. **Camera Latency**: Dominates ASL inference time
   - 76-91 ms capture time limits real-time use
   - Hardware limitation of sensor, not CNN

3. **Small Test Sets**: 30-image benchmarks
   - Sufficient for latency/throughput
   - Too small for robust accuracy evaluation

### Future Improvements

1. **Model Optimization**
   - Quantization-aware fine-tuning
   - Pruning + knowledge distillation
   - Architecture search for MAX78000 constraints

2. **System Optimization**
   - Parallel camera capture + inference
   - Lower-latency camera sensor
   - Power profiling and optimization

3. **Application Expansion**
   - Object detection (YOLO-tiny)
   - Gesture recognition
   - Continuous streaming inference

## References

- Analog Devices MAX78000 User Guide
- Maxim Micros SDK Documentation
- AI8X Model Training Framework

## License

Code from Maxim Integrated / Analog Devices: Apache License 2.0

---

**Author**: Thesis Project  
**Target**: EdgeAI Benchmarking and Application Study


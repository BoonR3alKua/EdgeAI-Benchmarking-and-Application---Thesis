# CIFAR-100 UART Classification System - Code Review & Fixes

## ✅ FIXES APPLIED

### 1. **CNN Timer Reset (CRITICAL)**

- **Issue**: `cnn_time` was not reset before `cnn_start()`, causing the wait loop to exit immediately after first inference
- **Fixed**: Added `cnn_time = 0;` before `cnn_start()` in `classify_image()`
- **Impact**: CNN inference now works correctly for all images, not just the first one

### 2. **Buffer Overflow Protection (CRITICAL)**

- **Issue**: Buffer bounds check was done AFTER writing to buffer
- **Fixed**: Added bounds check BEFORE writing in `receive_image_uart()`
- **Impact**: Prevents memory corruption from malformed packets

### 3. **Softmax Function Declaration**

- **Added**: External function prototype for `softmax_q17p14_q15()`
- **Impact**: Eliminates potential compilation warnings

### 4. **Documentation & Comments**

- **Added**: Comprehensive header comments explaining critical aspects
- **Added**: TODO note to verify CNN input memory address
- **Added**: Alternative memory loading strategy (commented out)

---

## ⚠️ REMAINING ISSUES TO VERIFY

### 1. **CNN Input Memory Address (HIGH PRIORITY)** ⚠️

**Current implementation**:

```c
memcpy32((uint32_t *)0x51018000, input_data, BYTES_PER_IMAGE / 4);
```

**Concern**: This loads ALL 3072 bytes to a single address. The MAX78000 CNN architecture may require data distribution across multiple memory banks.

**How to verify**:

1. Look for the original `ai8xize.py` generated code - it should have a `load_input()` function
2. Check if there's a reference implementation in the SDK
3. Run a test with known input and compare output with expected values
4. If results are wrong, try the multi-bank approach (uncommented in code)

**Alternative approach (if single address fails)**:

```c
// Split data across 4 memory banks
int words_per_bank = (BYTES_PER_IMAGE / 4) / 4;
memcpy32((uint32_t *)0x50400000, &input_data[0 * words_per_bank], words_per_bank);
memcpy32((uint32_t *)0x50408000, &input_data[1 * words_per_bank], words_per_bank);
memcpy32((uint32_t *)0x50410000, &input_data[2 * words_per_bank], words_per_bank);
memcpy32((uint32_t *)0x50418000, &input_data[3 * words_per_bank], words_per_bank);
```

### 2. **CNN Reinitialization Between Inferences**

**Current**:

```c
cnn_init();
cnn_configure();
```

**Question**: Are weights/bias persistence guaranteed across inferences?

**To verify**: Check if accuracy drops after first image. If yes, add:

```c
cnn_init();
cnn_load_weights();  // Add if needed
cnn_load_bias();     // Add if needed
cnn_configure();
```

### 3. **UART Timeout Mechanism**

**Issue**: `receive_image_uart()` can block forever if sender disconnects

**Recommendation**: Add timeout:

```c
uint32_t timeout_counter = 0;
#define UART_TIMEOUT 1000000  // Adjust as needed

while (1) {
    if (MXC_UART_ReadCharacter(RX_UART, &byte) != E_NO_ERROR) {
        if (++timeout_counter > UART_TIMEOUT) {
            printf("UART timeout\n");
            return E_TIME_OUT;
        }
        continue;
    }
    timeout_counter = 0;  // Reset on successful read
    // ... rest of code
}
```

### 4. **Data Format Verification**

**Verify these match between sender.py and main.c**:

- ✅ Image size: 32x32x3 = 3072 bytes
- ✅ Protocol: START (0xFF 0xAA) + IDX (4 bytes) + LABEL (1 byte) + IMAGE (3072) + END (0xBB 0xFF)
- ✅ Endianness: Little-endian (`<I` in Python struct.pack)
- ✅ Data type: uint8 (0-255)
- ✅ Format: HWC (Height, Width, Channel)
- ⚠️ **Channel order**: Verify if CNN expects RGB or BGR

---

## ✅ VERIFIED CORRECT

1. **Protocol Synchronization**: Sender and receiver protocols match exactly
2. **UART Configuration**: Both use 115200 baud, 8N1
3. **Buffer Sizes**: Adequate for protocol overhead
4. **State Machine**: Robust packet reception logic
5. **Accuracy Calculation**: Correctly tracks predictions vs ground truth
6. **Classification Logic**: Properly finds max class from CNN output
7. **LED Indicators**: Good visual feedback for debugging

---

## 🧪 TESTING CHECKLIST

### Before Running:

- [ ] Compile successfully without errors
- [ ] Flash to MAX78000 board
- [ ] Verify UART0 connected to PC COM5
- [ ] Run sender.py with test images

### During Testing:

- [ ] First image processes correctly
- [ ] Subsequent images also process correctly (verifies timer reset fix)
- [ ] Accuracy makes sense (>10% for 100 classes is reasonable, >50% is good)
- [ ] No crashes or hangs
- [ ] LED 0 stays on (ready)
- [ ] LED 1 blinks during inference (processing)

### If Results Are Wrong:

1. Check CNN input memory address (try alternative multi-bank approach)
2. Verify data format matches CNN training (RGB vs BGR, normalization)
3. Check if weights loaded correctly
4. Verify CNN configuration matches model architecture

### If System Hangs:

1. Check UART connection
2. Verify sender.py is running
3. Check for buffer overflow errors in console
4. Add timeout mechanism if needed

---

## 📊 EXPECTED PERFORMANCE

**CIFAR-100 Classification (100 classes)**:

- **Random guess**: 1% accuracy
- **Poor model**: 5-20% accuracy
- **Decent model**: 30-50% accuracy
- **Good model**: 60-75% accuracy
- **State-of-art**: 80%+ accuracy

**Inference Speed**:

- Based on CNN ops: ~9.5M operations
- Expected: < 50ms per image at 50MHz CNN clock

---

## 🔧 QUICK DEBUG COMMANDS

### Check if code compiles:

```bash
make clean
make
```

### Monitor UART output:

- Use serial terminal (PuTTY, TeraTerm, etc.)
- Port: Check device manager
- Baud: 115200, 8N1

### Verify sender works:

```bash
python sender.py
```

---

## 📝 ADDITIONAL NOTES

1. **Input Memory**: The uncertainty about memory address is the biggest risk. Monitor first few inferences closely.

2. **Softmax**: The function is called but not strictly necessary for classification (argmax of logits = argmax of softmax). However, it's useful for confidence scores.

3. **Performance**: With 1000 images at ~50ms each, full test should take ~50 seconds.

4. **COM Port**: Sender uses COM5 - verify this is correct for your system.

---

## 🎯 NEXT STEPS

1. **Build and flash** the code
2. **Run sender.py** with small batch first (10-20 images)
3. **Observe results** - if accuracy is terrible (<5%), suspect memory address issue
4. **If memory issue suspected**, uncomment multi-bank approach and retest
5. **Full run** with 1000 images once working

---

## 📚 REFERENCES

- MAX78000 CNN Configuration: See `cnn.c` auto-generated comments
- Input Format: HWC (Height, Width, Channel), 3x32x32
- Memory Map: Check MAX78000 datasheet section on CNN memory
- Protocol: Custom framing with start/end markers

---

**Generated**: 2025-11-10  
**Status**: CRITICAL FIXES APPLIED - READY FOR TESTING ⚠️ VERIFY CNN MEMORY ADDRESS

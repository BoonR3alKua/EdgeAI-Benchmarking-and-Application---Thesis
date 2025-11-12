# Quick Start Guide - CIFAR-100 UART Testing

## ⚡ Fast Setup (5 minutes)

### 1. Build & Flash (1 min)

```bash
cd d:\Thesis\Models\CIFAR100-effnetv2\my_cifar100_effnetv2
make clean
make
# Flash to board using your normal method
```

### 2. Connect Hardware (1 min)

- Connect MAX78000 to PC via USB
- Verify COM port (Device Manager on Windows)
- Update `sender.py` line 10 if not COM5

### 3. Run Test (3 min)

```bash
python sender.py
```

## 🔍 What to Look For

### ✅ Good Signs:

```
Predicted Class: 5
Confidence: 45.2%
Image 0: Predicted=5, Actual=5, CORRECT
Accuracy: 1/1 (100.00%)
```

### ❌ Bad Signs (Memory Issue):

```
Predicted Class: 67
Confidence: 2.1%
Image 0: Predicted=67, Actual=5, WRONG
Accuracy: 0/1 (0.00%)
Image 1: Predicted=23, Actual=8, WRONG
Accuracy: 0/2 (0.00%)
```

**If accuracy < 5% after 20 images** → CNN input memory address is likely wrong!

## 🔧 Quick Fix for Memory Issue

Edit `main.c`, find `load_input_from_buffer()` around line 130:

**Comment out the current line:**

```c
// memcpy32((uint32_t *)0x51018000, input_data, BYTES_PER_IMAGE / 4);
```

**Uncomment the multi-bank version:**

```c
int words_per_bank = (BYTES_PER_IMAGE / 4) / 4;
memcpy32((uint32_t *)0x50400000, &input_data[0 * words_per_bank], words_per_bank);
memcpy32((uint32_t *)0x50408000, &input_data[1 * words_per_bank], words_per_bank);
memcpy32((uint32_t *)0x50410000, &input_data[2 * words_per_bank], words_per_bank);
memcpy32((uint32_t *)0x50418000, &input_data[3 * words_per_bank], words_per_bank);
```

Rebuild and test again.

## 📊 Expected Results

| Metric                     | Value        |
| -------------------------- | ------------ |
| Inference time             | 10-50 ms     |
| Accuracy (good model)      | 40-70%       |
| Accuracy (poor model)      | 10-30%       |
| Transfer time per image    | ~100 ms      |
| Total time for 1000 images | ~2-3 minutes |

## 🐛 Troubleshooting

### No output from board:

- Check UART connection
- Verify baud rate (115200)
- Check serial terminal settings

### Python error "No module named 'serial'":

Already fixed! (pyserial installed)

### Board hangs:

- Reset board
- Check sender.py is running
- Verify COM port correct

### Low accuracy (<5%):

- **Most likely**: Wrong CNN input memory address
- Try multi-bank approach (see above)
- Verify data format (RGB vs BGR)

### Crashes after first image:

- Already fixed! (cnn_time reset added)

## 💡 Pro Tips

1. **Start small**: Test with 10 images first
2. **Watch the pattern**: Random results = memory issue, systematic errors = model issue
3. **LED feedback**: LED 1 should blink during inference
4. **Serial monitor**: Keep it open to see real-time results

---

**Need help?** Check `REVIEW_AND_FIXES.md` for detailed analysis.

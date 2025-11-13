# 🚀 OPTION A: See Live Results in Python Terminal

## Quick Start

### 1. Make sure PuTTY is CLOSED ❌

- Close any serial monitors
- Close PuTTY if open
- Close any terminal connected to COM5

### 2. Run the sender script

```bash
C:/Users/Asus/AppData/Local/Programs/Python/Python311/python.exe d:\Thesis\Models\CIFAR100-effnetv2\my_cifar100_effnetv2\sender.py
```

### 3. Watch the Python terminal! 👀

You will see:

```
Sending images:   0%|          | 4/1000 [00:01<04:23,  3.63it/s]

[Device]: Waiting for image...
[Device]: Received image 0, label 19
[Device]: Image 0: Predicted=19, Actual=19, CORRECT
[Device]: Accuracy: 1/1 (100.00%)

[Device]: Waiting for image...
[Device]: Received image 1, label 29
[Device]: Image 1: Predicted=29, Actual=29, CORRECT
[Device]: Accuracy: 2/2 (100.00%)

... (continues for all 1000 images)
```

## What You'll See

### Real-time Output:

- Progress bar showing images being sent
- Device responses showing:
  - Image received confirmation
  - Prediction vs actual label
  - CORRECT or WRONG status
  - Running accuracy percentage

### Expected Results:

- **Good model**: 30-70% accuracy
- **Poor model**: 10-30% accuracy
- **Broken/wrong memory**: < 5% accuracy

## Troubleshooting

### If you don't see device output:

1. Check if board is flashed and running
2. Increase delay in sender.py line 116: `time.sleep(0.1)` instead of `0.05`
3. Make sure board hasn't crashed (check LEDs)

### If accuracy is very low (< 5%):

❌ **CNN input memory address is wrong!**

Edit `main.c` line 145-155 and uncomment the multi-bank approach:

```c
// Comment out line 148:
// memcpy32((uint32_t *)0x51018000, input_data, BYTES_PER_IMAGE / 4);

// Uncomment lines 154-157:
int words_per_bank = (BYTES_PER_IMAGE / 4) / 4;
memcpy32((uint32_t *)0x50400000, &input_data[0 * words_per_bank], words_per_bank);
memcpy32((uint32_t *)0x50408000, &input_data[1 * words_per_bank], words_per_bank);
memcpy32((uint32_t *)0x50410000, &input_data[2 * words_per_bank], words_per_bank);
memcpy32((uint32_t *)0x50418000, &input_data[3 * words_per_bank], words_per_bank);
```

Then rebuild and reflash.

### If transmission is too slow:

Reduce delay in sender.py line 220: `time.sleep(0.001)` # 1ms instead of 10ms

## After Test Completes

You'll see final statistics:

```
============================================================
Transmission Complete
============================================================
Total images sent: 1000/1000
Failed transmissions: 0
Total time: 180.23 seconds
Average time per image: 180.23 ms
Throughput: 5.55 images/second
Data rate: 16.67 KB/s
```

**AND** the final accuracy from processing all 1000 images! 🎯

---

Ready? Run the command above and watch the magic happen! ✨

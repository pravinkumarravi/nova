# Nova Troubleshooting Guide

This guide addresses common setup, execution, and environment issues on Windows.

---

## 1. PowerShell Script Execution Disabled

### Symptom:
```text
.\.venv\Scripts\Activate.ps1 : File cannot be loaded because running scripts is disabled on this system.
```

### Cause:
Windows restricts script execution by default under the `Restricted` ExecutionPolicy.

### Solution:
Run this command in your PowerShell terminal to permit script execution for the current session:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

---

## 2. ModuleNotFoundError: No module named 'nova'

### Symptom:
```text
ModuleNotFoundError: No module named 'nova'
```

### Cause:
Python does not know that packages reside inside the `src/` folder unless `PYTHONPATH` is set.

### Solution:
Set `PYTHONPATH` in PowerShell:
```powershell
$env:PYTHONPATH = "src"
```

---

## 3. ZoneInfoNotFoundError on Windows

### Symptom:
```text
zoneinfo._common.ZoneInfoNotFoundError: 'No time zone found with key Asia/Kolkata'
```

### Cause:
Unlike Linux, Windows does not ship with standard IANA tzdata files in its system directories.

### Solution:
Nova's [`src/nova/tools/date_time.py`](file:///d:/AI/hello_ai/src/nova/tools/date_time.py) includes an automatic fallback that catches `ZoneInfoNotFoundError` and initializes a standard `datetime.timezone(timedelta(hours=5, minutes=30), name="IST")`. You can also install the IANA database via `pip install tzdata`.

---

## 4. PyTorch CUDA / CPU Device Issues

### Symptom:
```text
AssertionError: Torch not compiled with CUDA enabled
```

### Cause:
Attempting to run `--device cuda` on a machine without an NVIDIA GPU or with CPU-only PyTorch wheels.

### Solution:
Nova automatically defaults to CPU execution. You can explicitly force CPU mode on any command:
```powershell
python -m nova.cli train-intent --cpu
python -m nova.cli chat --cpu
```

---

## 5. Intent Classifier Falls Back to UNKNOWN

### Symptom:
The model responds: `"I have not been trained to answer general knowledge questions yet."` for a supported question.

### Cause:
The query phrasing was either:
1. Out of domain (expected behavior).
2. Worded with unfamiliar words, causing the neural softmax confidence to drop below the threshold (default: 0.65).

### Diagnostic:
Launch chat in debug mode to see the exact confidence score:
```powershell
python -m nova.cli chat --debug
```
You can adjust the threshold dynamically using `--threshold`:
```powershell
python -m nova.cli chat --threshold 0.50
```

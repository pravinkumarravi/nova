# Windows 10/11 Setup and PowerShell Execution Guide

This guide walks you through setting up and running Nova on a standard Windows 10 or 11 system using PowerShell.

---

## 1. Prerequisites

- **Operating System:** Windows 10 (64-bit) or Windows 11
- **Python Version:** Python 3.10 or higher (Python 3.11, 3.12, 3.13, or 3.14)
- **Shell:** Windows PowerShell 5.1 or PowerShell 7 (pwsh)
- **Hardware:** Standard CPU with at least 4 GB RAM. (No dedicated GPU required).

---

## 2. Step-by-Step PowerShell Setup

Open PowerShell in the project directory (`d:\AI\hello_ai`):

### Step A: Configure PowerShell Execution Policy
Windows PowerShell may block running script files like `Activate.ps1` by default. Enable script execution for the current shell session:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### Step B: Create a Dedicated Virtual Environment
Always isolate project dependencies inside `.venv`:
```powershell
python -m venv .venv
```

### Step C: Activate the Virtual Environment
```powershell
.\.venv\Scripts\Activate.ps1
```
*(You will see `(.venv)` prepended to your command prompt).*

### Step D: Upgrade Pip and Install Dependencies
```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Step E: Set PYTHONPATH
Tell Python to find the `nova` package in the `src/` directory:
```powershell
$env:PYTHONPATH = "src"
```

---

## 3. Verifying the Installation

Execute the automated test suite:
```powershell
pytest -v tests
```
You should see all unit and integration tests passing.

---

## 4. End-to-End Execution Workflow

Run these commands in order:

```powershell
# 1. Generate synthetic datasets
python -m nova.cli generate-data

# 2. Audit datasets for zero data leakage
python -m nova.cli audit-data

# 3. Train the Decoder Language Model
python -m nova.cli train-lm

# 4. Train the Neural Intent Classifier
python -m nova.cli train-intent

# 5. Evaluate classifier against baseline
python -m nova.cli evaluate

# 6. Start the interactive agent chat
python -m nova.cli chat

# 7. (Optional) Run in debug mode to see all internal pipeline steps
python -m nova.cli chat --debug

# 8. (Optional) Run direct language model text generation
python -m nova.cli generate-lm --prompt "User: What is your name?`nNova:"
```

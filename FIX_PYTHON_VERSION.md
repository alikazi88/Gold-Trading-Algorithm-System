# Fix Python Version for TensorFlow

## Problem

You're using **Python 3.13**, but TensorFlow only supports **Python 3.9 - 3.12**.

## Solution

Recreate your virtual environment with Python 3.12:

### Step 1: Remove Current Virtual Environment

```bash
cd /Users/ali/Developer/trader/gold_scalping_system
rm -rf venv
```

### Step 2: Install Python 3.12 (if not installed)

**Option A: Using Homebrew**
```bash
brew install python@3.12
```

**Option B: Download from python.org**
- Visit https://www.python.org/downloads/
- Download Python 3.12.x for macOS

### Step 3: Create New Virtual Environment with Python 3.12

```bash
# Find Python 3.12 path
which python3.12

# Create venv with Python 3.12
python3.12 -m venv venv

# Activate it
source venv/bin/activate

# Verify version
python --version  # Should show Python 3.12.x
```

### Step 4: Install Requirements

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Expected output:
```
Successfully installed tensorflow-macos-2.15.0
Successfully installed tensorflow-metal-1.1.0
...
```

### Step 5: Verify TensorFlow

```bash
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}, GPU: {len(tf.config.list_physical_devices(\"GPU\"))}')"
```

Expected output:
```
TensorFlow: 2.15.0, GPU: 1
```

## Quick Fix (One Command)

```bash
# Remove old venv, create new with Python 3.12, install requirements
rm -rf venv && python3.12 -m venv venv && source venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt
```

## Alternative: Use System Python

If you don't want to install Python 3.12:

```bash
# Check your system Python versions
ls /usr/local/bin/python3.*

# Use any Python 3.9-3.12
python3.11 -m venv venv  # or python3.10, python3.9
source venv/bin/activate
pip install -r requirements.txt
```

## After Fix

Once you have Python 3.12 with TensorFlow installed:

```bash
# Train GRU model
python train_gru_model.py

# Or quick test
python train_gru_model.py config/config_gru_test.yaml
```

## Why Python 3.13 Doesn't Work

- TensorFlow 2.15 (latest) only supports Python 3.9-3.12
- TensorFlow 2.16+ with Python 3.13 support is not released yet
- Expected release: Q2 2025

## Recommended Setup

- **Python Version**: 3.12.x (latest stable with TensorFlow support)
- **TensorFlow**: 2.15.0 (latest for macOS)
- **Metal Plugin**: 1.1.0 (GPU acceleration)

---

**After fixing, your GRU model will train 4-5x faster with Metal GPU!** 🚀

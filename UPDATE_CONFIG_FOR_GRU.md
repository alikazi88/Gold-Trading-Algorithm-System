# How to Update config.yaml for GRU Model

## Quick Update

Open your `config/config.yaml` and add these sections:

### 1. Change Model Type

Find the `model:` section and change:
```yaml
model:
  type: "gru"  # Change from "random_forest" to "gru"
```

### 2. Add GRU Parameters

Add these parameters under the `model:` section:

```yaml
model:
  type: "gru"
  
  # GRU parameters
  gru_sequence_length: 20        # Number of time steps to look back
  gru_units: [128, 64, 32]       # GRU layer sizes [layer1, layer2, layer3]
  gru_dropout: 0.3               # Dropout rate (0.2-0.5)
  gru_learning_rate: 0.001       # Adam optimizer learning rate
  gru_batch_size: 32             # Batch size for training
  gru_epochs: 50                 # Maximum epochs (early stopping enabled)
  
  # Common parameters
  min_accuracy: 0.60
  min_precision: 0.65
  retrain_interval_days: 14
  model_path: "models/saved/gru_scalping_model_latest.h5"
```

## Complete Model Section

Here's the complete `model:` section with both Random Forest and GRU parameters:

```yaml
model:
  # Model type: 'random_forest' or 'gru'
  type: "gru"
  
  # Random Forest parameters (if type='random_forest')
  n_estimators: 100
  max_depth: 20
  min_samples_split: 5
  min_samples_leaf: 1
  max_features: "sqrt"
  random_state: 42
  
  # GRU parameters (if type='gru')
  gru_sequence_length: 20        # Number of time steps to look back
  gru_units: [128, 64, 32]       # GRU layer sizes
  gru_dropout: 0.3               # Dropout rate
  gru_learning_rate: 0.001       # Learning rate
  gru_batch_size: 32             # Batch size for training
  gru_epochs: 50                 # Maximum epochs (early stopping enabled)
  
  # Common parameters
  min_accuracy: 0.60
  min_precision: 0.65
  retrain_interval_days: 14
  model_path: "models/saved/gru_scalping_model_latest.h5"
```

## Configuration for Different Macs

### For M1/M2/M3 Mac (8GB RAM)
```yaml
model:
  type: "gru"
  gru_sequence_length: 20
  gru_units: [128, 64, 32]
  gru_dropout: 0.3
  gru_learning_rate: 0.001
  gru_batch_size: 32             # Good for 8GB
  gru_epochs: 50
```

### For M1/M2/M3 Mac (16GB+ RAM)
```yaml
model:
  type: "gru"
  gru_sequence_length: 30
  gru_units: [256, 128, 64]      # Larger model
  gru_dropout: 0.3
  gru_learning_rate: 0.001
  gru_batch_size: 64             # Larger batches
  gru_epochs: 50
```

### For Intel Mac (CPU only)
```yaml
model:
  type: "gru"
  gru_sequence_length: 15        # Shorter for speed
  gru_units: [64, 32]            # Smaller model
  gru_dropout: 0.2
  gru_learning_rate: 0.001
  gru_batch_size: 16             # Smaller batches
  gru_epochs: 30                 # Fewer epochs
```

### For Quick Testing
```yaml
model:
  type: "gru"
  gru_sequence_length: 10        # Very short
  gru_units: [32, 16]            # Tiny model
  gru_dropout: 0.2
  gru_learning_rate: 0.001
  gru_batch_size: 64
  gru_epochs: 10                 # Fast training
  min_accuracy: 0.50             # Lower threshold
  min_precision: 0.50
```

## Parameter Explanation

| Parameter | Description | Recommended Range |
|-----------|-------------|-------------------|
| `gru_sequence_length` | How many past candles to look at | 15-30 |
| `gru_units` | Size of each GRU layer | [64-256, 32-128, 16-64] |
| `gru_dropout` | Prevents overfitting | 0.2-0.5 |
| `gru_learning_rate` | How fast model learns | 0.0001-0.01 |
| `gru_batch_size` | Samples per training step | 16-128 |
| `gru_epochs` | Maximum training iterations | 30-100 |

## Effects on Performance

### Sequence Length
- **Shorter (10-15)**: Faster training, reacts quickly, more noise
- **Medium (20-30)**: Balanced, recommended
- **Longer (40-60)**: Slower training, smoother predictions

### Model Size (gru_units)
- **Small [64, 32]**: Fast, less overfitting, lower accuracy
- **Medium [128, 64, 32]**: Balanced, recommended
- **Large [256, 128, 64]**: Slow, high accuracy, needs more data

### Batch Size
- **Small (16-32)**: More stable, slower training
- **Medium (32-64)**: Balanced
- **Large (64-128)**: Faster training, needs GPU

## After Updating config.yaml

### 1. Install TensorFlow (if not done)
```bash
# For M1/M2/M3 Mac
pip install tensorflow-macos tensorflow-metal

# For Intel Mac
pip install tensorflow
```

### 2. Train the Model
```bash
python train_gru_model.py
```

### 3. Run Live Trading
```bash
python main.py
```

The system will automatically use the GRU model based on `type: "gru"` in config.yaml!

## Switching Between Models

### To use Random Forest:
```yaml
model:
  type: "random_forest"
```

### To use GRU:
```yaml
model:
  type: "gru"
```

You can keep both sets of parameters in config.yaml. The system will use the one specified by `type`.

## Verification

After updating, verify your config:

```bash
python -c "from utils.helpers import load_config; c = load_config('config/config.yaml'); print(f'Model type: {c[\"model\"][\"type\"]}')"
```

Should output:
```
Model type: gru
```

---

**Your config.yaml is now ready for GRU model training!** 🚀

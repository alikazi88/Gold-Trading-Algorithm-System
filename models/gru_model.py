"""
GRU (Gated Recurrent Unit) model for gold scalping predictions.
Uses deep learning for time-series forecasting with sequential data.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import json
from datetime import datetime
from pathlib import Path

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, callbacks
    from tensorflow.keras.optimizers import Adam
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("Warning: TensorFlow not installed. Install with: pip install tensorflow")

from utils.logger import TradingLogger


class GRUScalpingModel:
    """
    GRU-based model for gold scalping signal prediction.
    Uses sequential price data to predict BUY/SELL/NO_TRADE signals.
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[TradingLogger] = None):
        """
        Initialize GRU model.
        
        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for GRU model. Install with: pip install tensorflow")
        
        self.config = config
        self.logger = logger or TradingLogger.get_logger(__name__)
        
        # Enable GPU memory growth for Apple Silicon (M1/M2/M3)
        self._configure_gpu_memory()
        
        # Model parameters
        self.sequence_length = config.get('model', {}).get('gru_sequence_length', 20)
        self.gru_units = config.get('model', {}).get('gru_units', [128, 64, 32])
        self.dropout_rate = config.get('model', {}).get('gru_dropout', 0.3)
        self.learning_rate = config.get('model', {}).get('gru_learning_rate', 0.001)
        self.batch_size = config.get('model', {}).get('gru_batch_size', 32)
        self.epochs = config.get('model', {}).get('gru_epochs', 50)
        
        # Model components
        self.model: Optional[keras.Model] = None
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self.training_metrics: Dict[str, Any] = {}
        self.model_path: Optional[str] = None
        
        # Class mapping
        self.class_mapping = {0: 'SELL', 1: 'NO_TRADE', 2: 'BUY'}
        self.reverse_mapping = {'SELL': 0, 'NO_TRADE': 1, 'BUY': 2}
        
        self.logger.info("GRU Model initialized")
    
    def _configure_gpu_memory(self) -> None:
        """
        Configure GPU memory growth for Apple Silicon Macs.
        Prevents TensorFlow from allocating all GPU memory at once.
        """
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                self.logger.info(f"GPU memory growth enabled for {len(gpus)} GPU(s) (Apple Metal)")
            else:
                self.logger.info("No GPU detected, using CPU")
        except Exception as e:
            self.logger.warning(f"Could not configure GPU memory: {e}")
    
    def _build_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """
        Build GRU neural network architecture.
        
        Args:
            input_shape: (sequence_length, n_features)
            
        Returns:
            Compiled Keras model
        """
        self.logger.info(f"Building GRU model with input shape: {input_shape}")
        
        model = models.Sequential(name='GRU_Scalping_Model')
        
        # Input layer
        model.add(layers.Input(shape=input_shape))
        
        # GRU layers with dropout
        for i, units in enumerate(self.gru_units):
            return_sequences = (i < len(self.gru_units) - 1)
            
            model.add(layers.GRU(
                units=units,
                return_sequences=return_sequences,
                dropout=self.dropout_rate,
                recurrent_dropout=self.dropout_rate * 0.5,
                name=f'gru_{i+1}'
            ))
            
            # Batch normalization
            model.add(layers.BatchNormalization(name=f'bn_{i+1}'))
        
        # Dense layers
        model.add(layers.Dense(64, activation='relu', name='dense_1'))
        model.add(layers.Dropout(self.dropout_rate, name='dropout_1'))
        
        model.add(layers.Dense(32, activation='relu', name='dense_2'))
        model.add(layers.Dropout(self.dropout_rate * 0.5, name='dropout_2'))
        
        # Output layer (3 classes: SELL, NO_TRADE, BUY)
        model.add(layers.Dense(3, activation='softmax', name='output'))
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.logger.info(f"Model built with {model.count_params():,} parameters")
        return model
    
    def _create_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for GRU training.
        
        Args:
            X: Feature array (n_samples, n_features)
            y: Target array (n_samples,)
            
        Returns:
            Tuple of (X_sequences, y_sequences)
        """
        X_seq, y_seq = [], []
        
        for i in range(len(X) - self.sequence_length):
            X_seq.append(X[i:i + self.sequence_length])
            y_seq.append(y[i + self.sequence_length])
        
        return np.array(X_seq), np.array(y_seq)
    
    def train(self, X: pd.DataFrame, y: pd.Series, 
              validation_split: float = 0.2) -> Dict[str, Any]:
        """
        Train the GRU model.
        
        Args:
            X: Feature DataFrame
            y: Target Series (SELL=-1, NO_TRADE=0, BUY=1)
            validation_split: Validation data split ratio
            
        Returns:
            Training metrics dictionary
        """
        self.logger.info("Starting GRU model training...")
        self.logger.info(f"Training samples: {len(X)}, Features: {len(X.columns)}")
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        # Convert target to 0, 1, 2 for categorical
        y_categorical = y.map({-1: 0, 0: 1, 1: 2}).values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X.values)
        
        # Create sequences
        self.logger.info(f"Creating sequences with length {self.sequence_length}...")
        X_seq, y_seq = self._create_sequences(X_scaled, y_categorical)
        
        self.logger.info(f"Sequence shape: {X_seq.shape}, Target shape: {y_seq.shape}")
        
        # Split data (time-series aware - no shuffle)
        split_idx = int(len(X_seq) * (1 - validation_split))
        X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]
        
        self.logger.info(f"Train: {len(X_train)}, Validation: {len(X_val)}")
        
        # Build model
        input_shape = (self.sequence_length, len(self.feature_names))
        self.model = self._build_model(input_shape)
        
        # Callbacks
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
        
        # Train model
        self.logger.info("Training GRU model...")
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        
        # Evaluate on validation set
        val_loss, val_acc = self.model.evaluate(X_val, y_val, verbose=0)
        
        # Calculate precision and recall manually
        y_pred = self.model.predict(X_val, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        from sklearn.metrics import precision_score, recall_score
        val_prec = precision_score(y_val, y_pred_classes, average='weighted', zero_division=0)
        val_rec = recall_score(y_val, y_pred_classes, average='weighted', zero_division=0)
        
        # Store metrics
        self.training_metrics = {
            'final_train_loss': float(history.history['loss'][-1]),
            'final_train_accuracy': float(history.history['accuracy'][-1]),
            'val_loss': float(val_loss),
            'val_accuracy': float(val_acc),
            'val_precision': float(val_prec),
            'val_recall': float(val_rec),
            'epochs_trained': len(history.history['loss']),
            'total_samples': len(X_seq),
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'sequence_length': self.sequence_length,
            'n_features': len(self.feature_names)
        }
        
        self.logger.info(f"Training completed - Val Accuracy: {val_acc:.4f}, Val Precision: {val_prec:.4f}")
        
        return self.training_metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict signals for new data.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Array of predictions (SELL=-1, NO_TRADE=0, BUY=1)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Scale features
        X_scaled = self.scaler.transform(X.values)
        
        # Create sequences
        if len(X_scaled) < self.sequence_length:
            raise ValueError(f"Need at least {self.sequence_length} samples for prediction")
        
        X_seq = []
        for i in range(len(X_scaled) - self.sequence_length + 1):
            X_seq.append(X_scaled[i:i + self.sequence_length])
        
        X_seq = np.array(X_seq)
        
        # Predict
        predictions = self.model.predict(X_seq, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Convert back to -1, 0, 1
        signal_mapping = {0: -1, 1: 0, 2: 1}
        signals = np.array([signal_mapping[c] for c in predicted_classes])
        
        return signals
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Array of probabilities (n_samples, 3) for [SELL, NO_TRADE, BUY]
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Scale features
        X_scaled = self.scaler.transform(X.values)
        
        # Create sequences
        if len(X_scaled) < self.sequence_length:
            raise ValueError(f"Need at least {self.sequence_length} samples for prediction")
        
        X_seq = []
        for i in range(len(X_scaled) - self.sequence_length + 1):
            X_seq.append(X_scaled[i:i + self.sequence_length])
        
        X_seq = np.array(X_seq)
        
        # Predict probabilities
        probabilities = self.model.predict(X_seq, verbose=0)
        
        return probabilities
    
    def save_model(self, save_dir: str, version: Optional[str] = None) -> str:
        """
        Save the trained model.
        
        Args:
            save_dir: Directory to save model
            version: Version string (default: timestamp)
            
        Returns:
            Path to saved model
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save Keras model
        model_file = save_path / f"gru_scalping_model_{version}.h5"
        self.model.save(str(model_file))
        
        # Save scaler and metadata
        metadata = {
            'version': version,
            'feature_names': self.feature_names,
            'sequence_length': self.sequence_length,
            'training_metrics': self.training_metrics,
            'scaler_mean': self.scaler.mean_.tolist(),
            'scaler_scale': self.scaler.scale_.tolist(),
            'class_mapping': self.class_mapping,
            'model_config': {
                'gru_units': self.gru_units,
                'dropout_rate': self.dropout_rate,
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'epochs': self.epochs
            }
        }
        
        metadata_file = save_path / f"gru_model_metadata_{version}.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.model_path = str(model_file)
        self.logger.info(f"Model saved to {model_file}")
        self.logger.info(f"Metadata saved to {metadata_file}")
        
        return str(model_file)
    
    def load_model(self, model_path: str) -> None:
        """
        Load a trained model.
        
        Args:
            model_path: Path to saved model file
        """
        self.logger.info(f"Loading model from {model_path}")
        
        # Load Keras model
        self.model = keras.models.load_model(model_path)
        
        # Load metadata
        metadata_path = model_path.replace('.h5', '_metadata.json').replace('gru_scalping_model', 'gru_model_metadata')
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        self.feature_names = metadata['feature_names']
        self.sequence_length = metadata['sequence_length']
        self.training_metrics = metadata['training_metrics']
        self.class_mapping = {int(k): v for k, v in metadata['class_mapping'].items()}
        
        # Restore scaler
        self.scaler.mean_ = np.array(metadata['scaler_mean'])
        self.scaler.scale_ = np.array(metadata['scaler_scale'])
        
        self.model_path = model_path
        self.logger.info("Model loaded successfully")
    
    def get_model_summary(self) -> str:
        """
        Get model architecture summary.
        
        Returns:
            Model summary string
        """
        if self.model is None:
            return "Model not built yet"
        
        summary_list = []
        self.model.summary(print_fn=lambda x: summary_list.append(x))
        return '\n'.join(summary_list)

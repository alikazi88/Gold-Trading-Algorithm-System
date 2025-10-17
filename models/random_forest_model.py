"""
Random Forest model for gold scalping predictions.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib
import json
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
from utils.logger import TradingLogger
import os


class GoldScalpingModel:
    """Random Forest model for predicting trade signals."""
    
    def __init__(self, config: Dict[str, Any], logger: Optional[TradingLogger] = None):
        """
        Initialize model.
        
        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        self.config = config
        self.logger = logger or TradingLogger.get_logger(__name__)
        
        model_config = config.get('model', {})
        
        self.test_size = model_config.get('test_size', 0.2)
        self.random_state = model_config.get('random_state', 42)
        self.hyperparameters = model_config.get('hyperparameters', {})
        self.min_accuracy = model_config.get('min_accuracy', 0.60)
        self.min_precision = model_config.get('min_precision', 0.65)
        
        self.model = None
        self.feature_names = None
        self.feature_importance = None
        self.training_metrics = {}
    
    def prepare_data(self, df: pd.DataFrame, feature_names: list) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for training.
        
        Args:
            df: DataFrame with features and labels
            feature_names: List of feature column names
            
        Returns:
            Tuple of (X, y)
        """
        # Remove rows with no label or missing features
        df_clean = df.dropna(subset=['label'])
        
        # Get features
        X = df_clean[feature_names].copy()
        y = df_clean['label'].copy()
        
        # Handle missing values in features
        X = X.fillna(0)
        X = X.replace([np.inf, -np.inf], 0)
        
        self.logger.info(f"Prepared data: {len(X)} samples, {len(feature_names)} features")
        
        return X, y
    
    def split_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into train and test sets (time-based split).
        
        Args:
            X: Features
            y: Labels
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        split_index = int(len(X) * (1 - self.test_size))
        
        X_train = X.iloc[:split_index]
        X_test = X.iloc[split_index:]
        y_train = y.iloc[:split_index]
        y_test = y.iloc[split_index:]
        
        self.logger.info(f"Train set: {len(X_train)} samples")
        self.logger.info(f"Test set: {len(X_test)} samples")
        
        return X_train, X_test, y_train, y_test
    
    def handle_class_imbalance(self, y_train: pd.Series) -> Dict[int, float]:
        """
        Calculate class weights to handle imbalance.
        
        Args:
            y_train: Training labels
            
        Returns:
            Dictionary of class weights
        """
        from sklearn.utils.class_weight import compute_class_weight
        
        classes = np.unique(y_train)
        weights = compute_class_weight('balanced', classes=classes, y=y_train)
        
        class_weights = dict(zip(classes, weights))
        self.logger.info(f"Class weights: {class_weights}")
        
        return class_weights
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
             tune_hyperparameters: bool = True) -> None:
        """
        Train the Random Forest model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            tune_hyperparameters: Whether to perform hyperparameter tuning
        """
        self.logger.info("Starting model training")
        
        # Handle class imbalance
        class_weights = self.handle_class_imbalance(y_train)
        
        if tune_hyperparameters and self.hyperparameters:
            self.logger.info("Performing hyperparameter tuning")
            
            # Use TimeSeriesSplit for cross-validation
            tscv = TimeSeriesSplit(n_splits=3)
            
            # Create base model
            base_model = RandomForestClassifier(
                random_state=self.random_state,
                class_weight=class_weights,
                n_jobs=-1
            )
            
            # Grid search
            grid_search = GridSearchCV(
                base_model,
                self.hyperparameters,
                cv=tscv,
                scoring='f1_weighted',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            self.model = grid_search.best_estimator_
            self.logger.info(f"Best parameters: {grid_search.best_params_}")
            self.training_metrics['best_params'] = grid_search.best_params_
            
        else:
            self.logger.info("Training with default parameters")
            
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=self.random_state,
                class_weight=class_weights,
                n_jobs=-1
            )
            
            self.model.fit(X_train, y_train)
        
        # Store feature names
        self.feature_names = X_train.columns.tolist()
        
        # Calculate feature importance
        self.feature_importance = dict(zip(
            self.feature_names,
            self.model.feature_importances_
        ))
        
        # Sort by importance
        self.feature_importance = dict(
            sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
        )
        
        self.logger.info("Model training completed")
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Evaluate model performance.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of metrics
        """
        self.logger.info("Evaluating model")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Per-class metrics
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': report
        }
        
        self.training_metrics.update(metrics)
        
        self.logger.info(f"Accuracy: {accuracy:.4f}")
        self.logger.info(f"Precision: {precision:.4f}")
        self.logger.info(f"Recall: {recall:.4f}")
        self.logger.info(f"F1 Score: {f1:.4f}")
        
        # Check if model meets minimum requirements
        if accuracy < self.min_accuracy:
            self.logger.warning(f"Accuracy {accuracy:.4f} below minimum {self.min_accuracy}")
        
        if precision < self.min_precision:
            self.logger.warning(f"Precision {precision:.4f} below minimum {self.min_precision}")
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions.
        
        Args:
            X: Features
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        return predictions, probabilities
    
    def predict_single(self, features: Dict[str, Any]) -> Tuple[int, float]:
        """
        Predict for a single sample.
        
        Args:
            features: Dictionary of features
            
        Returns:
            Tuple of (prediction, confidence)
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Create DataFrame with single row
        X = pd.DataFrame([features])
        
        # Ensure all feature columns exist
        for feature in self.feature_names:
            if feature not in X.columns:
                X[feature] = 0
        
        # Select features in correct order
        X = X[self.feature_names]
        
        # Handle missing values
        X = X.fillna(0)
        X = X.replace([np.inf, -np.inf], 0)
        
        # Predict
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]
        
        # Get confidence (probability of predicted class)
        class_index = list(self.model.classes_).index(prediction)
        confidence = probabilities[class_index]
        
        return int(prediction), float(confidence)
    
    def get_top_features(self, n: int = 15) -> Dict[str, float]:
        """
        Get top N most important features.
        
        Args:
            n: Number of features to return
            
        Returns:
            Dictionary of top features and their importance
        """
        if self.feature_importance is None:
            return {}
        
        return dict(list(self.feature_importance.items())[:n])
    
    def save_model(self, path: str, version: str = None) -> str:
        """
        Save model to disk.
        
        Args:
            path: Directory path to save model
            version: Model version string
            
        Returns:
            Full path to saved model
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        os.makedirs(path, exist_ok=True)
        
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        model_filename = f"gold_scalping_model_{version}.pkl"
        model_path = os.path.join(path, model_filename)
        
        # Save model
        joblib.dump(self.model, model_path)
        
        # Save metadata
        metadata = {
            'version': version,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance,
            'training_metrics': self.training_metrics,
            'config': self.config.get('model', {}),
            'saved_at': datetime.now().isoformat()
        }
        
        metadata_path = os.path.join(path, f"model_metadata_{version}.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Model saved to {model_path}")
        
        return model_path
    
    def load_model(self, model_path: str, metadata_path: str = None) -> None:
        """
        Load model from disk.
        
        Args:
            model_path: Path to model file
            metadata_path: Path to metadata file (optional)
        """
        self.model = joblib.load(model_path)
        self.logger.info(f"Model loaded from {model_path}")
        
        if metadata_path and os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            self.feature_names = metadata.get('feature_names', [])
            self.feature_importance = metadata.get('feature_importance', {})
            self.training_metrics = metadata.get('training_metrics', {})
            
            self.logger.info("Model metadata loaded")

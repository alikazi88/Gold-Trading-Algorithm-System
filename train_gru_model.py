#!/usr/bin/env python3
"""
Train the GRU (Gated Recurrent Unit) model for gold scalping.
Deep learning approach for time-series prediction.
"""
import sys
import os
from datetime import datetime, timezone
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.helpers import load_config
from utils.logger import TradingLogger
from data.database import TradingDatabase
from data.preprocessor import DataPreprocessor
from models.feature_engineering import FeatureEngineer
from models.labeling import TradeLabeler
from models.gru_model import GRUScalpingModel
from models.model_evaluation import ModelEvaluator


def main():
    """Main training function."""
    # Load configuration (use test config for quick testing)
    import sys
    config_file = sys.argv[1] if len(sys.argv) > 1 else 'config/config.yaml'
    config = load_config(config_file)
    
    # Setup logger
    logger = TradingLogger.setup_from_config(config, __name__)
    logger.info("=" * 60)
    logger.info("GRU MODEL TRAINING FOR GOLD SCALPING")
    logger.info("=" * 60)
    
    try:
        # Initialize components
        database = TradingDatabase(config['database']['path'])
        preprocessor = DataPreprocessor(logger)
        feature_engineer = FeatureEngineer(config, logger)
        labeler = TradeLabeler(config, logger)
        
        # Step 1: Load data
        logger.info("\n[1/7] Loading historical data from database...")
        df = database.get_candles()
        
        if len(df) < 1000:
            logger.error(f"Insufficient data: {len(df)} candles. Need at least 1000.")
            logger.info("Run: python main.py --fetch-data --days 365")
            logger.info("Or: python generate_mock_data.py --days 365")
            return
        
        logger.info(f"Loaded {len(df)} candles")
        
        # Step 2: Preprocess
        logger.info("\n[2/7] Preprocessing data...")
        df = preprocessor.clean_candle_data(df)
        logger.info(f"Cleaned data: {len(df)} candles remaining")
        
        # Step 3: Feature engineering
        logger.info("\n[3/7] Engineering features...")
        df = feature_engineer.calculate_all_features(df)
        logger.info(f"Feature calculation completed")
        
        # Step 4: Label data
        logger.info("\n[4/7] Labeling data for training...")
        df = labeler.label_dataset(df)
        
        # Remove unlabeled rows
        df_labeled = df[df['label'].notna()].copy()
        logger.info(f"Labeled samples: {len(df_labeled)}")
        
        if len(df_labeled) < 500:
            logger.error(f"Insufficient labeled data: {len(df_labeled)}. Need at least 500.")
            return
        
        # Prepare features and target
        feature_cols = feature_engineer.get_feature_names()
        X = df_labeled[feature_cols].copy()
        y = df_labeled['label'].copy()
        
        # Check for missing values
        if X.isnull().any().any():
            logger.warning("Missing values detected, filling with forward fill...")
            X = X.fillna(method='ffill').fillna(0)
        
        logger.info(f"Features: {len(feature_cols)}, Samples: {len(X)}")
        logger.info(f"Class distribution: SELL={sum(y==-1)}, NO_TRADE={sum(y==0)}, BUY={sum(y==1)}")
        
        # Step 5: Initialize and train GRU model
        logger.info("\n[5/7] Initializing GRU model...")
        model = GRUScalpingModel(config, logger)
        
        logger.info("\n[6/7] Training GRU model...")
        logger.info("This may take several minutes...")
        
        metrics = model.train(X, y, validation_split=0.2)
        
        logger.info("\nTraining Metrics:")
        logger.info(f"  Validation Accuracy: {metrics['val_accuracy']:.4f}")
        logger.info(f"  Validation Precision: {metrics['val_precision']:.4f}")
        logger.info(f"  Validation Recall: {metrics['val_recall']:.4f}")
        logger.info(f"  Epochs Trained: {metrics['epochs_trained']}")
        logger.info(f"  Sequence Length: {metrics['sequence_length']}")
        
        # Step 7: Evaluate model
        logger.info("\n[7/7] Evaluating model performance...")
        
        # Make predictions on full dataset for evaluation
        # Need to account for sequence length
        seq_len = model.sequence_length
        X_eval = X.iloc[seq_len-1:].copy()
        y_eval = y.iloc[seq_len-1:].values
        
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        
        # Get confidence scores (max probability)
        confidences = np.max(probabilities, axis=1)
        
        # Evaluate with sklearn metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(y_eval, predictions)
        precision = precision_score(y_eval, predictions, average='weighted', zero_division=0)
        recall = recall_score(y_eval, predictions, average='weighted', zero_division=0)
        f1 = f1_score(y_eval, predictions, average='weighted', zero_division=0)
        
        evaluation = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")
        
        # Generate report
        evaluator = ModelEvaluator(logger)
        report = evaluator.generate_evaluation_report(
            y_true=y_eval,
            y_pred=predictions,
            y_pred_proba=confidences,
            feature_importance=None  # GRU doesn't have feature importance like RF
        )
        
        evaluator.print_evaluation_summary(report)
        
        # Check thresholds
        min_accuracy = config['model']['min_accuracy']
        min_precision = config['model']['min_precision']
        
        if accuracy < min_accuracy:
            logger.warning(f"Accuracy {accuracy:.4f} below minimum {min_accuracy}")
        if precision < min_precision:
            logger.warning(f"Precision {precision:.4f} below minimum {min_precision}")
        
        # Save model
        logger.info("\nSaving model...")
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = model.save_model("models/saved", version=version)
        logger.info(f"Model saved to {model_path}")
        
        # Save performance to database
        performance_data = {
            'model_version': version,
            'training_date': int(datetime.now(timezone.utc).timestamp()),
            'accuracy': evaluation['accuracy'],
            'precision': evaluation['precision'],
            'recall': evaluation['recall'],
            'f1_score': evaluation['f1_score'],
            'train_samples': metrics['train_samples'],
            'test_samples': metrics['val_samples'],
            'feature_importance': 'N/A (GRU model)',
            'hyperparameters': str(model.training_metrics.get('model_config', {}))
        }
        
        database.insert_model_performance(performance_data)
        logger.info("Performance metrics saved to database")
        
        # Save evaluation report
        report_path = f"models/saved/gru_evaluation_report_{version}.json"
        evaluator.save_report(report, report_path)
        logger.info(f"Evaluation report saved to {report_path}")
        
        # Print model summary
        logger.info("\nModel Architecture:")
        logger.info(model.get_model_summary())
        
        logger.info("\n" + "=" * 60)
        logger.info("GRU MODEL TRAINING COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info(f"\nModel file: {model_path}")
        logger.info(f"Validation Accuracy: {metrics['val_accuracy']:.2%}")
        logger.info(f"Validation Precision: {metrics['val_precision']:.2%}")
        logger.info("\nTo use this model, update config.yaml:")
        logger.info("  model:")
        logger.info("    type: 'gru'")
        logger.info(f"    model_path: '{model_path}'")
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

"""
Train and retrain the Random Forest model.
"""
import sys
import os
from datetime import datetime, timezone, timedelta
import pandas as pd

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.helpers import load_config
from utils.logger import TradingLogger
from data.database import TradingDatabase
from data.fetcher import AllTickDataFetcher
from data.preprocessor import DataPreprocessor
from models.feature_engineering import FeatureEngineer
from models.labeling import TradeLabeler
from models.random_forest_model import GoldScalpingModel
from models.model_evaluation import ModelEvaluator


def train_model(config_path: str = "config/config.yaml", retrain: bool = False):
    """
    Train or retrain the ML model.
    
    Args:
        config_path: Path to configuration file
        retrain: Whether this is a retraining session
    """
    # Load configuration
    config = load_config(config_path)
    
    # Setup logger
    logger = TradingLogger.setup_from_config(config, __name__)
    logger.info("="*60)
    logger.info("GOLD SCALPING MODEL TRAINING")
    logger.info("="*60)
    
    try:
        # Initialize components
        database = TradingDatabase(config['database']['path'])
        data_fetcher = AllTickDataFetcher(config, logger)
        preprocessor = DataPreprocessor(logger)
        feature_engineer = FeatureEngineer(config, logger)
        labeler = TradeLabeler(config, logger)
        model = GoldScalpingModel(config, logger)
        evaluator = ModelEvaluator(logger)
        
        # Step 1: Load historical data
        logger.info("\n[1/7] Loading historical data from database...")
        
        if retrain:
            # For retraining, use last 6 months
            months = config['model'].get('training_data_months', 6)
            start_time = int((datetime.utcnow() - timedelta(days=months*30)).timestamp())
            df = database.get_candles(start_time=start_time)
        else:
            # For initial training, use all available data
            df = database.get_candles()
        
        if len(df) == 0:
            logger.error("No data found in database. Please run data fetching first.")
            return False
        
        logger.info(f"Loaded {len(df)} candles")
        
        # Step 2: Preprocess data
        logger.info("\n[2/7] Preprocessing data...")
        df = preprocessor.clean_candle_data(df)
        df = preprocessor.add_datetime_features(df)
        
        # Step 3: Engineer features
        logger.info("\n[3/7] Engineering features...")
        df = feature_engineer.calculate_all_features(df)
        
        # Step 4: Label data
        logger.info("\n[4/7] Labeling data...")
        df = labeler.label_dataset(df)
        
        # Get label statistics
        label_stats = labeler.get_label_statistics(df)
        logger.info(f"Label statistics: {label_stats}")
        
        # Step 5: Prepare data for training
        logger.info("\n[5/7] Preparing data for model training...")
        feature_names = feature_engineer.get_feature_names()
        X, y = model.prepare_data(df, feature_names)
        
        # Split data
        X_train, X_test, y_train, y_test = model.split_data(X, y)
        
        # Step 6: Train model
        logger.info("\n[6/7] Training Random Forest model...")
        logger.info("This may take several minutes...")
        
        tune_hyperparameters = not retrain  # Only tune on initial training
        model.train(X_train, y_train, tune_hyperparameters=tune_hyperparameters)
        
        # Step 7: Evaluate model
        logger.info("\n[7/7] Evaluating model performance...")
        metrics = model.evaluate(X_test, y_test)
        
        # Generate detailed evaluation report
        y_pred, y_pred_proba = model.predict(X_test)
        feature_importance = model.get_top_features(15)
        
        report = evaluator.generate_evaluation_report(
            y_test.values, y_pred, y_pred_proba, feature_importance
        )
        
        evaluator.print_evaluation_summary(report)
        
        # Check if model meets minimum requirements
        if metrics['accuracy'] < config['model']['min_accuracy']:
            logger.warning(f"Model accuracy {metrics['accuracy']:.4f} below minimum threshold")
        
        if metrics['precision'] < config['model']['min_precision']:
            logger.warning(f"Model precision {metrics['precision']:.4f} below minimum threshold")
        
        # Save model
        logger.info("\nSaving model...")
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = model.save_model("models/saved", version=version)
        logger.info(f"Model saved to {model_path}")
        
        # Save performance to database
        performance_data = {
            'model_version': version,
            'training_date': int(datetime.now(timezone.utc).timestamp()),
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score'],
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'feature_importance': str(feature_importance),
            'hyperparameters': str(model.training_metrics.get('best_params', {}))
        }
        
        database.insert_model_performance(performance_data)
        logger.info("Performance metrics saved to database")
        
        # Save evaluation report
        report_path = f"models/saved/evaluation_report_{version}.json"
        evaluator.save_report(report, report_path)
        logger.info(f"Evaluation report saved to {report_path}")
        
        logger.info("\n" + "="*60)
        logger.info("MODEL TRAINING COMPLETED SUCCESSFULLY")
        logger.info("="*60)
        
        return True
        
    except Exception as e:
        logger.error(f"Error during model training: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Gold Scalping ML Model")
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--retrain', action='store_true',
                       help='Retrain existing model with recent data')
    
    args = parser.parse_args()
    
    success = train_model(args.config, args.retrain)
    sys.exit(0 if success else 1)

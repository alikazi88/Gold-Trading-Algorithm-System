"""
Model evaluation and performance analysis.
"""
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from typing import Dict, Any, Optional
from utils.logger import TradingLogger
import json


class ModelEvaluator:
    """Evaluate and analyze model performance."""
    
    def __init__(self, logger: Optional[TradingLogger] = None):
        """
        Initialize model evaluator.
        
        Args:
            logger: Logger instance
        """
        self.logger = logger or TradingLogger.get_logger(__name__)
    
    def calculate_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """
        Calculate confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary with confusion matrix data
        """
        cm = confusion_matrix(y_true, y_pred)
        
        # Get unique classes
        classes = np.unique(np.concatenate([y_true, y_pred]))
        
        return {
            'matrix': cm.tolist(),
            'classes': classes.tolist()
        }
    
    def calculate_per_class_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Calculate metrics for each class.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary with per-class metrics
        """
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        
        # Extract per-class metrics
        class_metrics = {}
        for label in ['-1', '0', '1']:
            if label in report:
                class_name = {'-1': 'SELL', '0': 'NO_TRADE', '1': 'BUY'}[label]
                class_metrics[class_name] = {
                    'precision': report[label]['precision'],
                    'recall': report[label]['recall'],
                    'f1_score': report[label]['f1-score'],
                    'support': report[label]['support']
                }
        
        return class_metrics
    
    def calculate_trading_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                  y_pred_proba: np.ndarray = None) -> Dict[str, Any]:
        """
        Calculate trading-specific metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Prediction probabilities
            
        Returns:
            Dictionary with trading metrics
        """
        # Count predictions
        buy_signals = np.sum(y_pred == 1)
        sell_signals = np.sum(y_pred == -1)
        no_trade_signals = np.sum(y_pred == 0)
        
        # Correct predictions
        correct_buys = np.sum((y_pred == 1) & (y_true == 1))
        correct_sells = np.sum((y_pred == -1) & (y_true == -1))
        
        # Win rates
        buy_win_rate = (correct_buys / buy_signals * 100) if buy_signals > 0 else 0
        sell_win_rate = (correct_sells / sell_signals * 100) if sell_signals > 0 else 0
        
        # Overall trade win rate (excluding no_trade)
        total_trades = buy_signals + sell_signals
        correct_trades = correct_buys + correct_sells
        trade_win_rate = (correct_trades / total_trades * 100) if total_trades > 0 else 0
        
        metrics = {
            'total_predictions': len(y_pred),
            'buy_signals': int(buy_signals),
            'sell_signals': int(sell_signals),
            'no_trade_signals': int(no_trade_signals),
            'buy_win_rate': buy_win_rate,
            'sell_win_rate': sell_win_rate,
            'overall_trade_win_rate': trade_win_rate,
            'signal_rate': ((buy_signals + sell_signals) / len(y_pred) * 100)
        }
        
        # Average confidence if probabilities provided
        if y_pred_proba is not None:
            avg_confidence = np.mean(np.max(y_pred_proba, axis=1)) * 100
            metrics['average_confidence'] = avg_confidence
        
        return metrics
    
    def analyze_feature_importance(self, feature_importance: Dict[str, float],
                                   top_n: int = 15) -> Dict[str, Any]:
        """
        Analyze feature importance.
        
        Args:
            feature_importance: Dictionary of feature importances
            top_n: Number of top features to analyze
            
        Returns:
            Dictionary with analysis
        """
        # Sort by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        top_features = dict(sorted_features[:top_n])
        
        # Calculate cumulative importance
        total_importance = sum(feature_importance.values())
        cumulative_importance = 0
        
        for i, (feature, importance) in enumerate(sorted_features):
            cumulative_importance += importance
            if cumulative_importance >= 0.8 * total_importance:
                features_for_80_pct = i + 1
                break
        else:
            features_for_80_pct = len(sorted_features)
        
        return {
            'top_features': top_features,
            'features_for_80_percent': features_for_80_pct,
            'total_features': len(feature_importance)
        }
    
    def generate_evaluation_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                                   y_pred_proba: np.ndarray = None,
                                   feature_importance: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Prediction probabilities
            feature_importance: Feature importance dictionary
            
        Returns:
            Complete evaluation report
        """
        self.logger.info("Generating evaluation report")
        
        report = {
            'confusion_matrix': self.calculate_confusion_matrix(y_true, y_pred),
            'per_class_metrics': self.calculate_per_class_metrics(y_true, y_pred),
            'trading_metrics': self.calculate_trading_metrics(y_true, y_pred, y_pred_proba)
        }
        
        if feature_importance:
            report['feature_analysis'] = self.analyze_feature_importance(feature_importance)
        
        return report
    
    def print_evaluation_summary(self, report: Dict[str, Any]) -> None:
        """
        Print evaluation summary to console.
        
        Args:
            report: Evaluation report dictionary
        """
        print("\n" + "="*60)
        print("MODEL EVALUATION SUMMARY")
        print("="*60)
        
        # Trading metrics
        if 'trading_metrics' in report:
            tm = report['trading_metrics']
            print("\nTrading Metrics:")
            print(f"  Total Predictions: {tm['total_predictions']}")
            print(f"  Buy Signals: {tm['buy_signals']} (Win Rate: {tm['buy_win_rate']:.2f}%)")
            print(f"  Sell Signals: {tm['sell_signals']} (Win Rate: {tm['sell_win_rate']:.2f}%)")
            print(f"  No Trade Signals: {tm['no_trade_signals']}")
            print(f"  Overall Trade Win Rate: {tm['overall_trade_win_rate']:.2f}%")
            print(f"  Signal Rate: {tm['signal_rate']:.2f}%")
            if 'average_confidence' in tm:
                print(f"  Average Confidence: {tm['average_confidence']:.2f}%")
        
        # Per-class metrics
        if 'per_class_metrics' in report:
            print("\nPer-Class Metrics:")
            for class_name, metrics in report['per_class_metrics'].items():
                print(f"\n  {class_name}:")
                print(f"    Precision: {metrics['precision']:.4f}")
                print(f"    Recall: {metrics['recall']:.4f}")
                print(f"    F1-Score: {metrics['f1_score']:.4f}")
                print(f"    Support: {metrics['support']}")
        
        # Feature importance
        if 'feature_analysis' in report:
            fa = report['feature_analysis']
            print(f"\nFeature Analysis:")
            print(f"  Total Features: {fa['total_features']}")
            print(f"  Features for 80% Importance: {fa['features_for_80_percent']}")
            print(f"\n  Top 10 Features:")
            for i, (feature, importance) in enumerate(list(fa['top_features'].items())[:10], 1):
                print(f"    {i}. {feature}: {importance:.4f}")
        
        print("\n" + "="*60 + "\n")
    
    def save_report(self, report: Dict[str, Any], filepath: str) -> None:
        """
        Save evaluation report to file.
        
        Args:
            report: Evaluation report
            filepath: Path to save file
        """
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Evaluation report saved to {filepath}")

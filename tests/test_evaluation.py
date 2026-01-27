"""
Unit Tests for Evaluation Module
Tests metric calculation, visualization, and model comparison
"""

import pytest
import numpy as np
import torch
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.evaluation import ModelEvaluator, calculate_metrics
from client.model import HeartDiseaseMLP


class TestEvaluationMetrics:
    """
    Test suite for evaluation metrics
    """
    
    def test_calculate_metrics_perfect_prediction(self):
        """Test metrics calculation with perfect predictions"""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1, 0, 1])  # Perfect match
        
        metrics = calculate_metrics(y_true, y_pred)
        
        assert metrics['accuracy'] == 1.0
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0
        assert metrics['f1_score'] == 1.0
    
    def test_calculate_metrics_all_wrong(self):
        """Test metrics calculation with all wrong predictions"""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([1, 0, 1, 0])  # All wrong
        
        metrics = calculate_metrics(y_true, y_pred)
        
        assert metrics['accuracy'] == 0.0
        assert metrics['precision'] == 0.0
        assert metrics['recall'] == 0.0
        assert metrics['f1_score'] == 0.0
    
    def test_calculate_metrics_partial_accuracy(self):
        """Test metrics with partial accuracy"""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 0, 1])  # 4 out of 6 correct
        
        metrics = calculate_metrics(y_true, y_pred)
        
        assert metrics['accuracy'] == 4/6
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['f1_score'] <= 1
    
    def test_calculate_metrics_with_probabilities(self):
        """Test metrics calculation with probabilities"""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1, 0, 1])
        y_pred_proba = np.array([0.1, 0.9, 0.2, 0.8, 0.15, 0.85])
        
        metrics = calculate_metrics(y_true, y_pred, y_pred_proba)
        
        assert metrics['accuracy'] == 1.0
        assert metrics['auc'] is not None
        assert 0 <= metrics['auc'] <= 1
    
    def test_confusion_matrix_calculation(self):
        """Test confusion matrix calculation"""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 0, 1])
        
        metrics = calculate_metrics(y_true, y_pred)
        cm = metrics['confusion_matrix']
        
        # Check confusion matrix shape
        assert cm.shape == (2, 2)
        # Check all values are non-negative
        assert np.all(cm >= 0)


class TestModelEvaluator:
    """
    Test suite for ModelEvaluator class
    """
    
    def test_evaluator_initialization(self):
        """Test evaluator initialization"""
        evaluator = ModelEvaluator(model_name="Test Model")
        assert evaluator.model_name == "Test Model"
    
    def test_evaluate_model(self):
        """Test model evaluation"""
        # Create a simple model and test data
        model = HeartDiseaseMLP(input_size=13)
        X_test = torch.randn(20, 13)
        y_test = torch.randint(0, 2, (20, 1)).float()
        test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10)
        
        evaluator = ModelEvaluator(model_name="Test")
        metrics = evaluator.evaluate_model(model, test_loader, device="cpu")
        
        # Check all required metrics exist
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 'confusion_matrix' in metrics
        assert 'loss' in metrics
        
        # Check metric ranges
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['f1_score'] <= 1


class TestEvaluationEdgeCases:
    """
    Test edge cases for evaluation
    """
    
    def test_metrics_with_all_zeros(self):
        """Test metrics when all predictions are 0"""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 0, 0, 0])  # All predicted as 0
        
        metrics = calculate_metrics(y_true, y_pred)
        
        # Should handle gracefully
        assert metrics['accuracy'] == 0.5  # 2 out of 4 correct
        assert metrics['precision'] == 0.0  # No positive predictions
        assert metrics['recall'] == 0.0  # No true positives
    
    def test_metrics_with_all_ones(self):
        """Test metrics when all predictions are 1"""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([1, 1, 1, 1])  # All predicted as 1
        
        metrics = calculate_metrics(y_true, y_pred)
        
        assert metrics['accuracy'] == 0.5
        assert metrics['recall'] == 1.0  # All positives caught
        assert metrics['precision'] == 0.5  # Half are false positives
    
    def test_metrics_with_single_class(self):
        """Test metrics with single class in true labels"""
        y_true = np.array([1, 1, 1, 1])
        y_pred = np.array([1, 1, 1, 1])
        
        metrics = calculate_metrics(y_true, y_pred)
        
        # Should handle single class
        assert metrics['accuracy'] == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


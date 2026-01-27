"""
Local Model Evaluation - Phase 2
Evaluates trained model on local test data

This module computes comprehensive evaluation metrics:
- Accuracy
- Precision
- Recall
- F1-Score
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from client.model import HeartDiseaseMLP
from client.data_preprocessing import DataPreprocessor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


class LocalEvaluator:
    """
    Local model evaluator for client-side model evaluation
    
    This class handles:
    1. Loading trained model
    2. Loading preprocessed test data
    3. Computing evaluation metrics
    4. Displaying results
    """
    
    def __init__(self, client_id: int, device: str = "cpu"):
        """
        Initialize the evaluator
        
        Args:
            client_id: Client identifier
            device: Device to use ('cpu' or 'cuda')
        """
        self.client_id = client_id
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    def load_model(
        self,
        model_path: str,
        num_features: int
    ) -> HeartDiseaseMLP:
        """
        Load trained model from file
        
        Args:
            model_path: Path to saved model weights (.pth file)
            num_features: Number of input features (must match training)
            
        Returns:
            Loaded model
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"[Client {self.client_id}] Model not found: {model_path}")
        
        print(f"[Client {self.client_id}] Loading model from: {model_path}")
        
        # Create model with same architecture
        model = HeartDiseaseMLP(input_size=num_features)
        
        # Load weights
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()  # Set to evaluation mode
        
        print(f"[Client {self.client_id}] ✓ Model loaded successfully")
        
        return model
    
    def evaluate_model(
        self,
        model: HeartDiseaseMLP,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict:
        """
        Evaluate model on test data
        
        Args:
            model: Trained PyTorch model
            X_test: Test features (numpy array)
            y_test: Test labels (numpy array)
            
        Returns:
            Dictionary containing evaluation metrics
        """
        print(f"\n[Client {self.client_id}] Evaluating model on test data...")
        
        # Convert to PyTorch tensors
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = model(X_test_tensor)
            probabilities = outputs.cpu().numpy()
        
        # Convert probabilities to binary predictions (threshold = 0.5)
        predictions = (probabilities > 0.5).astype(int).flatten()
        y_test_flat = y_test.flatten()
        
        # Calculate metrics
        accuracy = accuracy_score(y_test_flat, predictions)
        precision = precision_score(y_test_flat, predictions, zero_division=0)
        recall = recall_score(y_test_flat, predictions, zero_division=0)
        f1 = f1_score(y_test_flat, predictions, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_test_flat, predictions)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'num_samples': len(y_test_flat)
        }
        
        return metrics
    
    def print_results(self, metrics: Dict):
        """
        Print evaluation results in a clear format
        
        Args:
            metrics: Dictionary containing evaluation metrics
        """
        print(f"\n[Client {self.client_id}] " + "=" * 60)
        print(f"[Client {self.client_id}] EVALUATION RESULTS")
        print(f"[Client {self.client_id}] " + "=" * 60)
        
        print(f"\n[Client {self.client_id}] Test Samples: {metrics['num_samples']}")
        
        print(f"\n[Client {self.client_id}] Classification Metrics:")
        print(f"[Client {self.client_id}]   Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"[Client {self.client_id}]   Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
        print(f"[Client {self.client_id}]   Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
        print(f"[Client {self.client_id}]   F1-Score:  {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)")
        
        # Confusion Matrix
        cm = metrics['confusion_matrix']
        print(f"\n[Client {self.client_id}] Confusion Matrix:")
        print(f"[Client {self.client_id}]                  Predicted")
        print(f"[Client {self.client_id}]                  No Disease  Disease")
        print(f"[Client {self.client_id}]   Actual No Disease    {cm[0][0]:6d}    {cm[0][1]:6d}")
        print(f"[Client {self.client_id}]   Actual Disease      {cm[1][0]:6d}    {cm[1][1]:6d}")
        
        # Interpretation
        print(f"\n[Client {self.client_id}] Interpretation:")
        print(f"[Client {self.client_id}]   True Negatives (TN):  {cm[0][0]} - Correctly predicted no disease")
        print(f"[Client {self.client_id}]   False Positives (FP): {cm[0][1]} - Incorrectly predicted disease")
        print(f"[Client {self.client_id}]   False Negatives (FN): {cm[1][0]} - Missed disease cases")
        print(f"[Client {self.client_id}]   True Positives (TP):  {cm[1][1]} - Correctly predicted disease")
        
        print(f"\n[Client {self.client_id}] " + "=" * 60)


def evaluate_client_model(
    client_id: int,
    model_path: str,
    client_data_path: str
) -> Dict:
    """
    Complete evaluation workflow for a single client
    
    Args:
        client_id: Client identifier
        model_path: Path to saved model weights
        client_data_path: Path to client's dataset CSV file
        
    Returns:
        Dictionary containing evaluation metrics
    """
    print(f"\n[Client {client_id}] " + "=" * 60)
    print(f"[Client {client_id}] EVALUATING CLIENT MODEL")
    print(f"[Client {client_id}] " + "=" * 60)
    
    # Step 1: Preprocess data to get test set
    print(f"\n[Client {client_id}] Step 1: Loading and preprocessing data...")
    preprocessor = DataPreprocessor(client_id=client_id)
    preprocessed_data = preprocessor.preprocess(
        file_path=client_data_path,
        test_size=0.2,
        random_state=42,
        normalize=True
    )
    
    # Step 2: Load model
    print(f"\n[Client {client_id}] Step 2: Loading trained model...")
    evaluator = LocalEvaluator(client_id=client_id)
    model = evaluator.load_model(
        model_path=model_path,
        num_features=preprocessed_data['num_features']
    )
    
    # Step 3: Evaluate
    print(f"\n[Client {client_id}] Step 3: Evaluating model...")
    metrics = evaluator.evaluate_model(
        model=model,
        X_test=preprocessed_data['X_test'],
        y_test=preprocessed_data['y_test']
    )
    
    # Step 4: Print results
    evaluator.print_results(metrics)
    
    return metrics


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("LOCAL MODEL EVALUATION - PHASE 2")
    print("=" * 60)
    
    # Example: Evaluate client 1
    client_id = 1
    model_path = f"models/client_{client_id}_model.pth"
    client_data_path = "dataset/processed/client_1.csv"
    
    if os.path.exists(model_path) and os.path.exists(client_data_path):
        metrics = evaluate_client_model(
            client_id=client_id,
            model_path=model_path,
            client_data_path=client_data_path
        )
        print(f"\n✓ Client {client_id} evaluation complete!")
    else:
        if not os.path.exists(model_path):
            print(f"Error: Model not found at {model_path}")
            print("Please train the model first using client/train.py")
        if not os.path.exists(client_data_path):
            print(f"Error: Client data not found at {client_data_path}")
            print("Please run Phase 1 first to create client datasets")


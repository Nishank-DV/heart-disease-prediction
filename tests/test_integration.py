"""
Integration Tests
Tests end-to-end workflows and component interactions
"""

import pytest
import pandas as pd
import numpy as np
import torch
import sys
import os
import tempfile

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from client.model import HeartDiseaseMLP
from client.data_preprocessing import DataPreprocessor
from client.train import LocalTrainer
from utils.dataset_splitter import split_dataset_for_clients
from utils.evaluation import ModelEvaluator


class TestEndToEndWorkflow:
    """
    Test complete workflow from data to model
    """
    
    def create_sample_dataset(self, num_samples=100):
        """Create a sample dataset"""
        np.random.seed(42)
        data = {
            'age': np.random.randint(29, 80, num_samples),
            'sex': np.random.randint(0, 2, num_samples),
            'cp': np.random.randint(0, 4, num_samples),
            'trestbps': np.random.randint(94, 200, num_samples),
            'chol': np.random.randint(126, 564, num_samples),
            'fbs': np.random.randint(0, 2, num_samples),
            'restecg': np.random.randint(0, 3, num_samples),
            'thalach': np.random.randint(71, 202, num_samples),
            'exang': np.random.randint(0, 2, num_samples),
            'oldpeak': np.random.uniform(0, 6.2, num_samples),
            'slope': np.random.randint(0, 3, num_samples),
            'ca': np.random.randint(0, 4, num_samples),
            'thal': np.random.randint(0, 4, num_samples),
            'target': np.random.randint(0, 2, num_samples)
        }
        return pd.DataFrame(data)
    
    def test_complete_preprocessing_to_training(self):
        """Test complete workflow: preprocessing -> training"""
        # Create dataset file
        df = self.create_sample_dataset(100)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            dataset_path = f.name
        
        try:
            # Step 1: Preprocess
            preprocessor = DataPreprocessor(client_id=1)
            preprocessed = preprocessor.preprocess(
                file_path=dataset_path,
                test_size=0.2,
                random_state=42,
                normalize=True
            )
            
            # Step 2: Create model
            model = HeartDiseaseMLP(input_size=preprocessed['num_features'])
            
            # Step 3: Train
            trainer = LocalTrainer(client_id=1, learning_rate=0.001, batch_size=16)
            train_loader, test_loader = trainer.load_preprocessed_data(preprocessed)
            
            # Quick training test
            criterion = torch.nn.BCELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
            # Train for 1 epoch
            model.train()
            for features, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                break  # Just one batch for test
            
            # Step 4: Evaluate
            evaluator = ModelEvaluator()
            metrics = evaluator.evaluate_model(model, test_loader, device="cpu")
            
            # Should have valid metrics
            assert 'accuracy' in metrics
            assert 0 <= metrics['accuracy'] <= 1
            
        finally:
            os.unlink(dataset_path)
    
    def test_dataset_splitting_workflow(self):
        """Test dataset splitting creates valid client datasets"""
        df = self.create_sample_dataset(100)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            dataset_path = f.name
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Split dataset
                client_paths = split_dataset_for_clients(
                    dataset_path=dataset_path,
                    num_clients=3,
                    output_dir=temp_dir,
                    random_state=42,
                    stratify=True
                )
                
                # Check all client files exist
                assert len(client_paths) == 3
                for path in client_paths:
                    assert os.path.exists(path)
                    client_df = pd.read_csv(path)
                    assert client_df.shape[0] > 0
                    assert 'target' in client_df.columns
        finally:
            os.unlink(dataset_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


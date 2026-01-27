"""
Integration Tests for Federated Learning
Tests federated learning workflow, client-server communication, and aggregation
"""

import pytest
import torch
import numpy as np
import sys
import os
import tempfile

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from client.model import HeartDiseaseMLP
from client.fl_client import FlowerClient, create_flower_client
from client.data_preprocessing import DataPreprocessor


class TestFederatedLearning:
    """
    Test suite for federated learning components
    """
    
    def create_sample_dataset_file(self, num_samples=100):
        """Create a sample dataset CSV file"""
        import pandas as pd
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
        df = pd.DataFrame(data)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            return f.name
    
    def test_flower_client_initialization(self):
        """Test Flower client initialization"""
        dataset_path = self.create_sample_dataset_file(50)
        try:
            client = create_flower_client(
                client_id=1,
                client_data_path=dataset_path,
                num_features=13,
                local_epochs=2,
                learning_rate=0.001
            )
            assert client.client_id == 1
            assert client.num_features == 13
            assert client.local_epochs == 2
        finally:
            os.unlink(dataset_path)
    
    def test_flower_client_get_parameters(self):
        """Test getting model parameters from client"""
        dataset_path = self.create_sample_dataset_file(50)
        try:
            client = create_flower_client(
                client_id=1,
                client_data_path=dataset_path,
                num_features=13
            )
            params = client.get_parameters({})
            
            # Should return list of numpy arrays
            assert isinstance(params, list)
            assert len(params) > 0
            assert all(isinstance(p, np.ndarray) for p in params)
        finally:
            os.unlink(dataset_path)
    
    def test_flower_client_set_parameters(self):
        """Test setting model parameters on client"""
        dataset_path = self.create_sample_dataset_file(50)
        try:
            client = create_flower_client(
                client_id=1,
                client_data_path=dataset_path,
                num_features=13
            )
            
            # Get initial parameters
            initial_params = client.get_parameters({})
            
            # Create new parameters (simulate server aggregation)
            new_params = [p + 0.1 for p in initial_params]
            
            # Set new parameters
            client.set_parameters(new_params)
            
            # Get parameters again
            updated_params = client.get_parameters({})
            
            # Should be different from initial
            assert not np.array_equal(initial_params[0], updated_params[0])
        finally:
            os.unlink(dataset_path)
    
    def test_flower_client_fit(self):
        """Test client fit method (local training)"""
        dataset_path = self.create_sample_dataset_file(100)
        try:
            client = create_flower_client(
                client_id=1,
                client_data_path=dataset_path,
                num_features=13,
                local_epochs=2
            )
            
            # Get initial parameters
            initial_params = client.get_parameters({})
            
            # Perform local training
            updated_params, num_samples, metrics = client.fit(
                parameters=initial_params,
                config={"local_epochs": 2}
            )
            
            # Check return values
            assert isinstance(updated_params, list)
            assert num_samples > 0
            assert isinstance(metrics, dict)
            assert "loss" in metrics
            assert "client_id" in metrics
            
            # Parameters should be updated (different from initial)
            assert not np.array_equal(initial_params[0], updated_params[0])
        finally:
            os.unlink(dataset_path)
    
    def test_flower_client_evaluate(self):
        """Test client evaluate method"""
        dataset_path = self.create_sample_dataset_file(100)
        try:
            client = create_flower_client(
                client_id=1,
                client_data_path=dataset_path,
                num_features=13
            )
            
            # Get model parameters
            params = client.get_parameters({})
            
            # Evaluate model
            loss, num_samples, metrics = client.evaluate(
                parameters=params,
                config={}
            )
            
            # Check return values
            assert isinstance(loss, float)
            assert num_samples > 0
            assert isinstance(metrics, dict)
            assert "accuracy" in metrics
            assert "loss" in metrics
            assert "client_id" in metrics
            
            # Loss should be non-negative
            assert loss >= 0
            # Accuracy should be between 0 and 1
            assert 0 <= metrics["accuracy"] <= 1
        finally:
            os.unlink(dataset_path)


class TestFederatedAveraging:
    """
    Test federated averaging logic
    """
    
    def test_weighted_average_calculation(self):
        """Test weighted average calculation (simulating FedAvg)"""
        # Simulate 3 clients with different sample counts
        client1_weights = [np.array([1.0, 2.0, 3.0])]
        client2_weights = [np.array([2.0, 3.0, 4.0])]
        client3_weights = [np.array([3.0, 4.0, 5.0])]
        
        client1_samples = 100
        client2_samples = 200
        client3_samples = 300
        total_samples = client1_samples + client2_samples + client3_samples
        
        # Calculate weighted average (FedAvg)
        aggregated = [
            (client1_weights[0] * client1_samples +
             client2_weights[0] * client2_samples +
             client3_weights[0] * client3_samples) / total_samples
        ]
        
        # Expected: weighted average
        expected = (client1_weights[0] * 100 + 
                   client2_weights[0] * 200 + 
                   client3_weights[0] * 300) / 600
        
        assert np.allclose(aggregated[0], expected)
    
    def test_parameter_aggregation(self):
        """Test aggregating parameters from multiple clients"""
        # Create 3 models with different initializations
        model1 = HeartDiseaseMLP(input_size=13)
        model2 = HeartDiseaseMLP(input_size=13)
        model3 = HeartDiseaseMLP(input_size=13)
        
        # Get parameters from each
        params1 = model1.get_model_parameters()
        params2 = model2.get_model_parameters()
        params3 = model3.get_model_parameters()
        
        # Simulate weighted average (equal weights for simplicity)
        n1, n2, n3 = 100, 200, 300
        total = n1 + n2 + n3
        
        aggregated = []
        for p1, p2, p3 in zip(params1, params2, params3):
            avg = (p1 * n1 + p2 * n2 + p3 * n3) / total
            aggregated.append(avg)
        
        # Should have same structure as input
        assert len(aggregated) == len(params1)
        assert all(a.shape == p.shape for a, p in zip(aggregated, params1))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


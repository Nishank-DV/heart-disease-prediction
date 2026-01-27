"""
Unit Tests for MLP Model
Tests model creation, forward pass, and parameter handling
"""

import pytest
import torch
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from client.model import HeartDiseaseMLP, create_model


class TestModel:
    """Test cases for MLP model"""
    
    def test_model_creation(self):
        """Test 1: Model can be created with different input sizes"""
        # Test with 13 features (typical heart disease dataset)
        model = HeartDiseaseMLP(input_size=13)
        assert model is not None
        assert model.input_size == 13
        
        # Test with different input size
        model2 = HeartDiseaseMLP(input_size=20)
        assert model2.input_size == 20
    
    def test_model_architecture(self):
        """Test 2: Model has correct architecture"""
        model = HeartDiseaseMLP(input_size=13)
        
        # Check layer sizes
        assert model.hidden1.in_features == 13
        assert model.hidden1.out_features == 64
        assert model.hidden2.in_features == 64
        assert model.hidden2.out_features == 32
        assert model.output.in_features == 32
        assert model.output.out_features == 1
    
    def test_forward_pass(self):
        """Test 3: Forward pass produces correct output shape"""
        model = HeartDiseaseMLP(input_size=13)
        model.eval()
        
        # Create dummy input
        batch_size = 10
        dummy_input = torch.randn(batch_size, 13)
        
        # Forward pass
        output = model(dummy_input)
        
        # Check output shape
        assert output.shape == (batch_size, 1)
        
        # Check output range (should be 0-1 due to sigmoid)
        assert torch.all(output >= 0.0)
        assert torch.all(output <= 1.0)
    
    def test_model_parameters(self):
        """Test 4: Model parameters can be extracted and set"""
        model = HeartDiseaseMLP(input_size=13)
        
        # Get parameters
        params = model.get_model_parameters()
        assert len(params) > 0
        assert all(isinstance(p, np.ndarray) for p in params)
        
        # Set parameters (should not raise error)
        model.set_model_parameters(params)
    
    def test_model_prediction_range(self):
        """Test 5: Model predictions are in valid range [0, 1]"""
        model = HeartDiseaseMLP(input_size=13)
        model.eval()
        
        # Test with various inputs
        test_inputs = [
            torch.randn(1, 13),  # Random input
            torch.zeros(1, 13),  # Zero input
            torch.ones(1, 13) * 10,  # Large values
            torch.ones(1, 13) * -10,  # Negative values
        ]
        
        for inp in test_inputs:
            output = model(inp)
            assert 0.0 <= output.item() <= 1.0, f"Output {output.item()} not in [0, 1]"
    
    def test_model_with_medical_data_range(self):
        """Test 6: Model handles realistic medical data ranges"""
        model = HeartDiseaseMLP(input_size=13)
        model.eval()
        
        # Create realistic medical data
        # Age: 30-80, normalized values: -1.5 to 1.5
        realistic_input = torch.tensor([[
            0.5,   # age (normalized)
            1.0,   # sex (male)
            2.0,   # cp (chest pain type)
            0.3,   # trestbps (normalized)
            0.8,   # chol (normalized)
            0.0,   # fbs
            1.0,   # restecg
            -0.5,  # thalach (normalized)
            0.0,   # exang
            0.2,   # oldpeak (normalized)
            1.0,   # slope
            1.0,   # ca
            2.0    # thal
        ]])
        
        output = model(realistic_input)
        assert 0.0 <= output.item() <= 1.0
        assert not torch.isnan(output)
        assert not torch.isinf(output)
    
    def test_model_factory_function(self):
        """Test 7: Factory function creates valid model"""
        model = create_model(input_size=13)
        assert isinstance(model, HeartDiseaseMLP)
        assert model.input_size == 13
    
    def test_model_gradient_flow(self):
        """Test 8: Model can compute gradients (for training)"""
        model = HeartDiseaseMLP(input_size=13)
        model.train()
        
        dummy_input = torch.randn(5, 13, requires_grad=True)
        output = model(dummy_input)
        
        # Create dummy loss
        target = torch.ones(5, 1)
        loss = torch.nn.functional.binary_cross_entropy(output, target)
        
        # Backward pass
        loss.backward()
        
        # Check gradients exist
        assert dummy_input.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

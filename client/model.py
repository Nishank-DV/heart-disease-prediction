"""
Deep Learning Model Definition - Phase 2
Multilayer Perceptron (MLP) for Heart Disease Prediction

This module implements a neural network model suitable for binary classification
of heart disease using tabular medical data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HeartDiseaseMLP(nn.Module):
    """
    Multilayer Perceptron (MLP) for Binary Heart Disease Classification
    
    Architecture:
    - Input Layer: Dynamic size (based on number of features)
    - Hidden Layer 1: 64 neurons with ReLU activation
    - Hidden Layer 2: 32 neurons with ReLU activation
    - Output Layer: 1 neuron with Sigmoid activation
    
    Why MLP for Medical Tabular Data?
    ===================================
    MLPs are well-suited for tabular medical data because:
    1. They can learn complex non-linear relationships between features
    2. They handle both numerical and encoded categorical features effectively
    3. They are interpretable compared to more complex models
    4. They work well with moderate-sized datasets (100-1000s of samples)
    5. They can capture interactions between medical features (e.g., age + cholesterol)
    
    For heart disease prediction, MLPs can learn patterns like:
    - "High cholesterol + high blood pressure + age > 60" → higher risk
    - "Normal ECG + low ST depression" → lower risk
    """
    
    def __init__(self, input_size: int):
        """
        Initialize the MLP model
        
        Args:
            input_size: Number of input features (must match preprocessed data)
                       This is dynamic and determined from the dataset
        """
        super(HeartDiseaseMLP, self).__init__()
        
        self.input_size = input_size
        
        # Hidden Layer 1: 64 neurons
        # This layer learns the first level of feature interactions
        self.hidden1 = nn.Linear(input_size, 64)
        
        # Hidden Layer 2: 32 neurons
        # This layer learns higher-level patterns from Layer 1's outputs
        self.hidden2 = nn.Linear(64, 32)
        
        # Output Layer: 1 neuron
        # Single neuron for binary classification (0 = no disease, 1 = disease)
        self.output = nn.Linear(32, 1)
        
        # Initialize weights using Xavier/Glorot initialization
        # This helps with training stability and convergence
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Initialize model weights using Xavier/Glorot uniform initialization
        
        Why Xavier Initialization?
        - Prevents vanishing/exploding gradients
        - Ensures weights start in a good range for training
        - Particularly important for deep networks
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            
        Returns:
            Output tensor of shape (batch_size, 1) with probabilities between 0 and 1
        """
        # Hidden Layer 1: Linear transformation + ReLU activation
        # ReLU (Rectified Linear Unit) introduces non-linearity
        # ReLU(x) = max(0, x) - allows model to learn non-linear patterns
        x = F.relu(self.hidden1(x))
        
        # Hidden Layer 2: Linear transformation + ReLU activation
        x = F.relu(self.hidden2(x))
        
        # Output Layer: Linear transformation + Sigmoid activation
        # Why Sigmoid?
        # ============
        # 1. Sigmoid outputs values between 0 and 1 (probabilities)
        # 2. Perfect for binary classification (disease/no disease)
        # 3. Works well with Binary Cross Entropy loss
        # 4. Smooth gradient for backpropagation
        # 
        # Sigmoid(x) = 1 / (1 + e^(-x))
        # Output: 0.0 to 1.0, where:
        #   - Values close to 0 → No heart disease
        #   - Values close to 1 → Heart disease present
        x = torch.sigmoid(self.output(x))
        
        return x
    
    def get_model_parameters(self):
        """
        Get model parameters as a list of numpy arrays
        This method will be used in Phase 3 (Federated Learning) to share
        model weights between clients and server
        
        Returns:
            List of numpy arrays representing model parameters
        """
        return [param.data.cpu().numpy() for param in self.parameters()]
    
    def set_model_parameters(self, parameters):
        """
        Set model parameters from a list of numpy arrays
        This method will be used in Phase 3 (Federated Learning) to receive
        aggregated model weights from the server
        
        Args:
            parameters: List of numpy arrays representing model parameters
        """
        for param, new_param in zip(self.parameters(), parameters):
            param.data = torch.from_numpy(new_param).float()


def create_model(input_size: int) -> HeartDiseaseMLP:
    """
    Factory function to create a HeartDiseaseMLP model
    
    Args:
        input_size: Number of input features
        
    Returns:
        Initialized HeartDiseaseMLP model
    """
    model = HeartDiseaseMLP(input_size=input_size)
    return model


if __name__ == "__main__":
    # Example usage and model summary
    print("=" * 60)
    print("HEART DISEASE MLP MODEL - PHASE 2")
    print("=" * 60)
    
    # Example: Create model for 13 features (typical heart disease dataset)
    input_features = 13
    model = create_model(input_size=input_features)
    
    print(f"\nModel Architecture:")
    print(f"  Input Layer: {input_features} features")
    print(f"  Hidden Layer 1: 64 neurons (ReLU)")
    print(f"  Hidden Layer 2: 32 neurons (ReLU)")
    print(f"  Output Layer: 1 neuron (Sigmoid)")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Parameters:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    print(f"\nTesting forward pass...")
    batch_size = 4
    dummy_input = torch.randn(batch_size, input_features)
    output = model(dummy_input)
    
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
    print(f"  (Should be between 0 and 1 due to Sigmoid)")
    
    print("\n✓ Model created successfully!")

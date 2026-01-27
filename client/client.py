"""
Federated Learning Client Implementation using Flower
Each client trains the model locally and sends updates to the server
"""

import flwr as fl
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from client.model import HeartDiseaseMLP, train_epoch, evaluate_model
from client.data_preprocessing import DataPreprocessor


class FlowerClient(fl.client.NumPyClient):
    """
    Flower client for federated learning
    Handles local training and model updates
    """
    
    def __init__(
        self,
        client_id: int,
        model: HeartDiseaseMLP,
        train_loader,
        test_loader,
        num_features: int,
        local_epochs: int = 5,
        learning_rate: float = 0.001,
        device: str = "cpu"
    ):
        """
        Initialize Flower client
        
        Args:
            client_id: Unique identifier for the client
            model: PyTorch model instance
            train_loader: DataLoader for training data
            test_loader: DataLoader for test data
            num_features: Number of input features
            local_epochs: Number of local training epochs per round
            learning_rate: Learning rate for optimizer
            device: Device to use (cpu or cuda)
        """
        self.client_id = client_id
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.num_features = num_features
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Move model to device
        self.model.to(self.device)
        
        # Loss function and optimizer
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        print(f"[Client {self.client_id}] Initialized on device: {self.device}")
    
    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """
        Return current model parameters as a list of NumPy arrays
        Called by Flower to get model weights for aggregation
        
        Args:
            config: Configuration dictionary (unused in this implementation)
            
        Returns:
            List of model parameters as NumPy arrays
        """
        return [param.data.cpu().numpy() for param in self.model.parameters()]
    
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """
        Set model parameters from a list of NumPy arrays
        Called by Flower to update model with aggregated weights from server
        
        Args:
            parameters: List of NumPy arrays representing model parameters
        """
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=False)
    
    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict
    ) -> Tuple[List[np.ndarray], int, Dict]:
        """
        Train the model locally and return updated parameters
        This is the core federated learning training step
        
        Args:
            parameters: Current global model parameters from server
            config: Configuration dictionary with training settings
            
        Returns:
            Tuple of (updated_parameters, number_of_samples, metrics_dict)
        """
        # Set model parameters from server
        self.set_parameters(parameters)
        
        # Get number of local epochs from config or use default
        epochs = config.get("local_epochs", self.local_epochs)
        
        # Local training
        print(f"[Client {self.client_id}] Starting local training for {epochs} epochs...")
        for epoch in range(epochs):
            loss = train_epoch(
                self.model,
                self.train_loader,
                self.criterion,
                self.optimizer,
                self.device
            )
            if (epoch + 1) % 2 == 0 or epoch == 0:
                print(f"[Client {self.client_id}] Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")
        
        # Get updated parameters
        updated_parameters = self.get_parameters({})
        
        # Calculate number of training samples
        num_samples = len(self.train_loader.dataset)
        
        # Return updated parameters, sample count, and metrics
        return updated_parameters, num_samples, {"loss": loss, "client_id": self.client_id}
    
    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict
    ) -> Tuple[float, int, Dict]:
        """
        Evaluate the model on local test data
        Called by Flower to assess model performance
        
        Args:
            parameters: Model parameters to evaluate
            config: Configuration dictionary
            
        Returns:
            Tuple of (loss, number_of_samples, metrics_dict)
        """
        # Set model parameters
        self.set_parameters(parameters)
        
        # Evaluate model
        loss, predictions, labels = evaluate_model(
            self.model,
            self.test_loader,
            self.criterion,
            self.device
        )
        
        # Calculate accuracy
        predictions_array = np.array(predictions).flatten()
        labels_array = np.array(labels).flatten()
        accuracy = np.mean(predictions_array == labels_array)
        
        num_samples = len(self.test_loader.dataset)
        
        metrics = {
            "accuracy": float(accuracy),
            "loss": float(loss),
            "client_id": self.client_id
        }
        
        print(f"[Client {self.client_id}] Evaluation - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        
        return float(loss), num_samples, metrics


def create_client(
    client_id: int,
    data_path: str,
    local_epochs: int = 5,
    learning_rate: float = 0.001,
    apply_smote: bool = True
) -> FlowerClient:
    """
    Factory function to create a Flower client with preprocessed data
    
    Args:
        client_id: Unique identifier for the client
        data_path: Path to the client's local dataset
        local_epochs: Number of local training epochs
        learning_rate: Learning rate for optimizer
        apply_smote: Whether to apply SMOTE for class balancing
        
    Returns:
        Initialized FlowerClient instance
    """
    # Preprocess data
    preprocessor = DataPreprocessor(client_id, apply_smote=apply_smote)
    df = preprocessor.load_data(data_path)
    train_loader, test_loader, num_features = preprocessor.prepare_data(df)
    
    # Create model
    model = HeartDiseaseMLP(input_size=num_features)
    
    # Create and return client
    client = FlowerClient(
        client_id=client_id,
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        num_features=num_features,
        local_epochs=local_epochs,
        learning_rate=learning_rate
    )
    
    return client


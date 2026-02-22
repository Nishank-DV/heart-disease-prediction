"""
Federated Learning Client - Phase 3
Flower NumPyClient implementation for federated learning

This module implements a Flower client that:
- Loads local client dataset (from Phase 1)
- Uses MLP model (from Phase 2)
- Trains locally on client data
- Returns only model weights (not raw data)
- Evaluates model locally

PRIVACY PRESERVATION:
====================
- Raw patient data NEVER leaves the client
- Only model weights (parameters) are shared with server
- This ensures patient privacy while enabling collaborative learning
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
try:
    import flwr as fl
except Exception:
    class _FallbackNumPyClient:
        pass

    class _FallbackClientModule:
        NumPyClient = _FallbackNumPyClient

    class _FallbackFlwr:
        client = _FallbackClientModule()

    fl = _FallbackFlwr()

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from client.model import HeartDiseaseMLP
from client.data_preprocessing import DataPreprocessor


class FlowerClient(fl.client.NumPyClient):
    """
    Flower NumPyClient for Federated Learning
    
    This client implements the federated learning protocol:
    1. Receives global model weights from server
    2. Trains model locally on client's private data
    3. Returns updated model weights to server
    4. Never shares raw data
    
    Why Only Weights Are Shared?
    ============================
    Model weights are mathematical parameters (numbers) that represent
    what the model has learned. They do NOT contain any information about
    individual patients. This is the key to privacy-preserving federated learning:
    
    - Raw Data: Contains sensitive patient information (age, medical history, etc.)
    - Model Weights: Just numbers representing learned patterns (no patient data)
    
    Example:
    - Raw Data: "Patient 123: Age=65, Cholesterol=250, Disease=Yes"
    - Model Weights: [0.234, -0.567, 0.891, ...] (just numbers, no patient info)
    """
    
    def __init__(
        self,
        client_id: int,
        client_data_path: str,
        num_features: int,
        local_epochs: int = 5,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        device: str = "cpu"
    ):
        """
        Initialize the Flower client
        
        Args:
            client_id: Unique identifier for the client
            client_data_path: Path to client's local dataset CSV file
            num_features: Number of input features
            local_epochs: Number of local training epochs per federated round
            learning_rate: Learning rate for local training
            batch_size: Batch size for training
            device: Device to use ('cpu' or 'cuda')
        """
        self.client_id = client_id
        self.client_data_path = client_data_path
        self.num_features = num_features
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Load and preprocess client data (LOCALLY - never shared)
        print(f"[Client {self.client_id}] Loading local dataset...")
        self._prepare_local_data()
        
        # Initialize model
        self.model = HeartDiseaseMLP(input_size=num_features)
        self.model.to(self.device)
        
        # Loss function and optimizer
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        print(f"[Client {self.client_id}] Flower client initialized")
        print(f"[Client {self.client_id}] Device: {self.device}")
        print(f"[Client {self.client_id}] Local epochs per round: {local_epochs}")
    
    def _prepare_local_data(self):
        """
        Prepare local training and test data
        
        This data stays LOCAL and is NEVER sent to the server.
        Only model weights (learned parameters) are shared.
        """
        # Preprocess data using Phase 1 preprocessor
        preprocessor = DataPreprocessor(client_id=self.client_id)
        preprocessed_data = preprocessor.preprocess(
            file_path=self.client_data_path,
            test_size=0.2,
            random_state=42,
            normalize=True
        )
        
        # Convert to PyTorch tensors
        X_train = torch.FloatTensor(preprocessed_data['X_train'])
        y_train = torch.FloatTensor(preprocessed_data['y_train']).unsqueeze(1)
        X_test = torch.FloatTensor(preprocessed_data['X_test'])
        y_test = torch.FloatTensor(preprocessed_data['y_test']).unsqueeze(1)
        
        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
        
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        
        self.test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )
        
        self.num_train_samples = len(train_dataset)
        self.num_test_samples = len(test_dataset)
        
        print(f"[Client {self.client_id}] Local data prepared:")
        print(f"[Client {self.client_id}]   Train samples: {self.num_train_samples}")
        print(f"[Client {self.client_id}]   Test samples: {self.num_test_samples}")
    
    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """
        Return current model parameters as NumPy arrays
        
        This is called by Flower to get model weights for aggregation.
        Only WEIGHTS are returned, NOT raw data.
        
        Args:
            config: Configuration dictionary (unused in this implementation)
            
        Returns:
            List of model parameters as NumPy arrays
        """
        return [param.data.cpu().numpy().copy() for param in self.model.parameters()]
    
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """
        Set model parameters from NumPy arrays
        
        This is called by Flower to update model with aggregated weights
        from the server (after FedAvg aggregation).
        
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
        
        This is the core federated learning training step:
        1. Receive global model weights from server
        2. Train locally on client's private data
        3. Return updated weights (NOT data)
        
        PRIVACY: Raw data never leaves this function. Only weights are returned.
        
        Args:
            parameters: Current global model parameters from server
            config: Configuration dictionary with training settings
            
        Returns:
            Tuple of (updated_parameters, number_of_samples, metrics_dict)
        """
        # Set model parameters from server (global model)
        self.set_parameters(parameters)
        
        # Get number of local epochs from config or use default
        epochs = config.get("local_epochs", self.local_epochs)
        
        # Local training on PRIVATE data (never shared)
        print(f"[Client {self.client_id}] Training locally for {epochs} epochs...")
        
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for features, labels in self.train_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        
        # Get updated parameters (ONLY weights, NO data)
        updated_parameters = self.get_parameters({})
        
        # Return updated weights, sample count, and metrics
        # NOTE: We return num_train_samples for weighted averaging in FedAvg
        return updated_parameters, self.num_train_samples, {
            "loss": avg_loss,
            "client_id": self.client_id
        }
    
    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict
    ) -> Tuple[float, int, Dict]:
        """
        Evaluate the model on local test data
        
        This evaluates the global model on client's local test set.
        Only metrics are returned, NOT test data.
        
        Args:
            parameters: Model parameters to evaluate
            config: Configuration dictionary
            
        Returns:
            Tuple of (loss, number_of_samples, metrics_dict)
        """
        # Set model parameters
        self.set_parameters(parameters)
        
        # Evaluate on local test data (stays local)
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        num_batches = 0
        
        with torch.no_grad():
            for features, labels in self.test_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                
                # Calculate accuracy
                predictions = (outputs > 0.5).float()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0
        
        metrics = {
            "accuracy": float(accuracy),
            "loss": float(avg_loss),
            "client_id": self.client_id
        }
        
        print(f"[Client {self.client_id}] Evaluation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        return float(avg_loss), self.num_test_samples, metrics


def create_flower_client(
    client_id: int,
    client_data_path: str,
    num_features: Optional[int] = None,
    local_epochs: int = 5,
    learning_rate: float = 0.001,
    batch_size: int = 32
) -> FlowerClient:
    """
    Factory function to create a Flower client
    
    Args:
        client_id: Client identifier
        client_data_path: Path to client's dataset
        num_features: Number of input features
        local_epochs: Local training epochs per round
        learning_rate: Learning rate
        batch_size: Batch size
        
    Returns:
        Initialized FlowerClient instance
    """
    if num_features is None:
        preprocessor = DataPreprocessor(client_id=client_id)
        df = preprocessor.load_data(client_data_path)
        target_col = preprocessor.identify_target_column(df)
        num_features = len(df.columns) - 1 if target_col else len(df.columns)

    client = FlowerClient(
        client_id=client_id,
        client_data_path=client_data_path,
        num_features=num_features,
        local_epochs=local_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size
    )
    
    return client


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("FLOWER CLIENT - PHASE 3")
    print("=" * 60)
    
    # Example: Create client 1
    client_id = 1
    client_data_path = "dataset/processed/client_1.csv"
    num_features = 13  # Typical for heart disease dataset
    
    if os.path.exists(client_data_path):
        client = create_flower_client(
            client_id=client_id,
            client_data_path=client_data_path,
            num_features=num_features,
            local_epochs=5,
            learning_rate=0.001
        )
        print(f"\n✓ Client {client_id} created successfully!")
        print("Note: This client is ready to connect to a Flower server")
    else:
        print(f"Error: Client data not found at {client_data_path}")
        print("Please run Phase 1 first to create client datasets")


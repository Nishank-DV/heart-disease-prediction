"""
Local Training Pipeline - Phase 2
Trains the MLP model on client-side preprocessed data

This module handles:
- Loading preprocessed data from Phase 1
- Converting to PyTorch tensors
- Training the model locally
- Saving model weights

IMPORTANT: This is LOCAL training only. Federated learning will be added in Phase 3.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from client.model import HeartDiseaseMLP
from client.data_preprocessing import DataPreprocessor


class LocalTrainer:
    """
    Local trainer for client-side model training
    
    This class handles the complete training pipeline:
    1. Load preprocessed data
    2. Convert to PyTorch tensors
    3. Train model with Adam optimizer
    4. Track training metrics
    5. Save model weights
    """
    
    def __init__(
        self,
        client_id: int,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        device: str = "cpu"
    ):
        """
        Initialize the local trainer
        
        Args:
            client_id: Client identifier
            learning_rate: Learning rate for Adam optimizer
            batch_size: Batch size for training
            device: Device to use ('cpu' or 'cuda')
        """
        self.client_id = client_id
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        print(f"[Client {self.client_id}] Trainer initialized")
        print(f"[Client {self.client_id}] Device: {self.device}")
    
    def load_preprocessed_data(self, preprocessed_data: Dict) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """
        Convert preprocessed NumPy arrays to PyTorch DataLoaders
        
        Args:
            preprocessed_data: Dictionary from DataPreprocessor.preprocess()
                              Contains X_train, X_test, y_train, y_test
                              
        Returns:
            Tuple of (train_loader, test_loader)
        """
        # Extract data
        X_train = preprocessed_data['X_train']
        X_test = preprocessed_data['X_test']
        y_train = preprocessed_data['y_train']
        y_test = preprocessed_data['y_test']
        
        # Convert to PyTorch tensors
        # Float32 for features, Float32 for labels (required for BCE loss)
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)  # Add dimension for BCE
        
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)
        
        # Create datasets
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
        
        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True  # Shuffle for better training
        )
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False  # No need to shuffle test data
        )
        
        print(f"[Client {self.client_id}] Data converted to PyTorch tensors")
        print(f"[Client {self.client_id}]   Train batches: {len(train_loader)}")
        print(f"[Client {self.client_id}]   Test batches: {len(test_loader)}")
        
        return train_loader, test_loader
    
    def train_epoch(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer
    ) -> float:
        """
        Train the model for one epoch
        
        Args:
            model: PyTorch model
            train_loader: DataLoader for training data
            criterion: Loss function
            optimizer: Optimizer
            
        Returns:
            Average loss for the epoch
        """
        model.train()  # Set model to training mode
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (features, labels) in enumerate(train_loader):
            # Move data to device
            features = features.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()  # Clear gradients
            outputs = model(features)  # Get predictions
            
            # Calculate loss
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()  # Compute gradients
            optimizer.step()  # Update weights
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def validate(
        self,
        model: nn.Module,
        test_loader: torch.utils.data.DataLoader,
        criterion: nn.Module
    ) -> Tuple[float, float]:
        """
        Validate the model on test data
        
        Args:
            model: PyTorch model
            test_loader: DataLoader for test data
            criterion: Loss function
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        model.eval()  # Set model to evaluation mode
        total_loss = 0.0
        correct = 0
        total = 0
        num_batches = 0
        
        with torch.no_grad():  # No gradient computation during validation
            for features, labels in test_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                # Calculate accuracy
                # Convert probabilities to binary predictions (threshold = 0.5)
                predictions = (outputs > 0.5).float()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0
        
        return avg_loss, accuracy
    
    def train(
        self,
        preprocessed_data: Dict,
        num_epochs: int = 50,
        save_path: str = None
    ) -> HeartDiseaseMLP:
        """
        Complete training pipeline
        
        Args:
            preprocessed_data: Dictionary from DataPreprocessor.preprocess()
            num_epochs: Number of training epochs
            save_path: Path to save model weights (optional)
            
        Returns:
            Trained model
        """
        print(f"\n[Client {self.client_id}] " + "=" * 60)
        print(f"[Client {self.client_id}] STARTING LOCAL TRAINING")
        print(f"[Client {self.client_id}] " + "=" * 60)
        
        # Get number of features
        num_features = preprocessed_data['num_features']
        
        # Create model
        print(f"[Client {self.client_id}] Creating model with {num_features} input features...")
        model = HeartDiseaseMLP(input_size=num_features)
        model.to(self.device)
        
        # Load data
        train_loader, test_loader = self.load_preprocessed_data(preprocessed_data)
        
        # Define loss function
        # Why Binary Cross Entropy (BCE)?
        # =================================
        # 1. BCE is designed for binary classification (2 classes)
        # 2. Works perfectly with Sigmoid activation (outputs 0-1 probabilities)
        # 3. Penalizes confident wrong predictions more
        # 4. Standard choice for binary classification in deep learning
        # 
        # BCE Loss = -[y*log(ŷ) + (1-y)*log(1-ŷ)]
        # Where: y = true label, ŷ = predicted probability
        criterion = nn.BCELoss()
        
        # Define optimizer
        # Why Adam Optimizer?
        # ===================
        # 1. Adaptive learning rate (adjusts per parameter)
        # 2. Combines benefits of AdaGrad and RMSProp
        # 3. Works well with sparse gradients
        # 4. Generally requires less hyperparameter tuning
        # 5. Good default choice for most deep learning tasks
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        
        print(f"[Client {self.client_id}] Loss function: Binary Cross Entropy")
        print(f"[Client {self.client_id}] Optimizer: Adam (lr={self.learning_rate})")
        print(f"[Client {self.client_id}] Training for {num_epochs} epochs...")
        print(f"\n[Client {self.client_id}] Epoch | Train Loss | Test Loss | Accuracy")
        print(f"[Client {self.client_id}] " + "-" * 60)
        
        # Training loop
        best_accuracy = 0.0
        for epoch in range(1, num_epochs + 1):
            # Train
            train_loss = self.train_epoch(model, train_loader, criterion, optimizer)
            
            # Validate
            test_loss, accuracy = self.validate(model, test_loader, criterion)
            
            # Print progress
            if epoch % 5 == 0 or epoch == 1:
                print(f"[Client {self.client_id}] {epoch:5d} | {train_loss:10.4f} | {test_loss:9.4f} | {accuracy:8.4f}")
            
            # Track best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
        
        print(f"[Client {self.client_id}] " + "-" * 60)
        print(f"[Client {self.client_id}] Training complete!")
        print(f"[Client {self.client_id}] Best accuracy: {best_accuracy:.4f}")
        
        # Save model
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"[Client {self.client_id}] Model saved to: {save_path}")
        
        return model


def train_client_model(
    client_id: int,
    client_data_path: str,
    num_epochs: int = 50,
    learning_rate: float = 0.001,
    batch_size: int = 32,
    save_model: bool = True
) -> HeartDiseaseMLP:
    """
    Complete training workflow for a single client
    
    Args:
        client_id: Client identifier
        client_data_path: Path to client's dataset CSV file
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        batch_size: Batch size for training
        save_model: Whether to save model weights
        
    Returns:
        Trained model
    """
    # Step 1: Preprocess data (using Phase 1 preprocessor)
    print(f"\n[Client {client_id}] Step 1: Preprocessing data...")
    preprocessor = DataPreprocessor(client_id=client_id)
    preprocessed_data = preprocessor.preprocess(
        file_path=client_data_path,
        test_size=0.2,
        random_state=42,
        normalize=True
    )
    
    # Step 2: Train model
    print(f"\n[Client {client_id}] Step 2: Training model...")
    trainer = LocalTrainer(
        client_id=client_id,
        learning_rate=learning_rate,
        batch_size=batch_size
    )
    
    save_path = None
    if save_model:
        os.makedirs("models", exist_ok=True)
        save_path = f"models/client_{client_id}_model.pth"
    
    model = trainer.train(
        preprocessed_data=preprocessed_data,
        num_epochs=num_epochs,
        save_path=save_path
    )
    
    return model


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("LOCAL TRAINING - PHASE 2")
    print("=" * 60)
    
    # Example: Train client 1
    client_id = 1
    client_data_path = "dataset/processed/client_1.csv"
    
    if os.path.exists(client_data_path):
        model = train_client_model(
            client_id=client_id,
            client_data_path=client_data_path,
            num_epochs=50,
            learning_rate=0.001,
            batch_size=32,
            save_model=True
        )
        print(f"\n✓ Client {client_id} training complete!")
    else:
        print(f"Error: Client data not found at {client_data_path}")
        print("Please run Phase 1 first to create client datasets")


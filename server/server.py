"""
Federated Learning Server - Phase 3
Flower server implementation with Federated Averaging (FedAvg)

This module implements the federated learning server that:
- Initializes a global MLP model
- Coordinates federated learning rounds
- Aggregates client model updates using FedAvg
- Distributes aggregated model to clients

FEDERATED AVERAGING (FedAvg) ALGORITHM:
========================================
FedAvg is the core algorithm of federated learning:

1. Server initializes global model
2. Server sends global model weights to selected clients
3. Each client trains locally on their private data
4. Clients send updated weights (NOT data) back to server
5. Server aggregates weights using weighted average:
   
   w_global = Σ(n_i * w_i) / Σ(n_i)
   
   Where:
   - w_global = aggregated global weights
   - n_i = number of samples at client i
   - w_i = weights from client i

6. Server distributes aggregated model to clients
7. Repeat steps 2-6 for multiple rounds

WHY FEDERATED LEARNING IS BETTER THAN CENTRALIZED ML?
=====================================================
1. Privacy: Raw data never leaves clients (HIPAA/GDPR compliant)
2. Security: No single point of failure for data breaches
3. Scalability: Can handle distributed data across many clients
4. Efficiency: Clients train locally, reducing network traffic
5. Collaboration: Multiple hospitals can collaborate without sharing data
"""

import flwr as fl
from typing import Dict, List, Optional, Tuple
import numpy as np
from flwr.server.strategy import FedAvg
from flwr.common import Metrics
import sys
import os
import torch

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from client.model import HeartDiseaseMLP


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Custom metric aggregation function for weighted averaging
    
    This aggregates metrics from multiple clients, weighted by the number
    of samples each client has. This ensures clients with more data
    have more influence on the global metrics.
    
    Args:
        metrics: List of tuples (num_samples, metrics_dict) from clients
        
    Returns:
        Aggregated metrics dictionary
    """
    # Extract accuracies and losses weighted by sample count
    accuracies = [num_samples * m["accuracy"] for num_samples, m in metrics]
    losses = [num_samples * m["loss"] for num_samples, m in metrics]
    examples = [num_samples for num_samples, _ in metrics]
    
    # Calculate weighted averages
    aggregated_metrics = {
        "accuracy": sum(accuracies) / sum(examples) if sum(examples) > 0 else 0.0,
        "loss": sum(losses) / sum(examples) if sum(examples) > 0 else 0.0,
    }
    
    return aggregated_metrics


def create_federated_server(
    num_rounds: int = 10,
    fraction_fit: float = 1.0,
    fraction_evaluate: float = 1.0,
    min_fit_clients: int = 3,
    min_evaluate_clients: int = 3,
    min_available_clients: int = 3,
    local_epochs: int = 5,
    learning_rate: float = 0.001
) -> FedAvg:
    """
    Create and configure the federated learning server with FedAvg strategy
    
    Args:
        num_rounds: Number of federated learning rounds
        fraction_fit: Fraction of clients used for training (1.0 = all clients)
        fraction_evaluate: Fraction of clients used for evaluation (1.0 = all clients)
        min_fit_clients: Minimum number of clients required for training
        min_evaluate_clients: Minimum number of clients required for evaluation
        min_available_clients: Minimum number of clients that must be available
        local_epochs: Number of local training epochs per round
        learning_rate: Learning rate for clients
        
    Returns:
        FedAvg strategy configuration
    """
    # Define Federated Averaging strategy
    # FedAvg aggregates client updates using weighted average based on sample count
    strategy = FedAvg(
        fraction_fit=fraction_fit,  # Use all available clients for training
        fraction_evaluate=fraction_evaluate,  # Use all available clients for evaluation
        min_fit_clients=min_fit_clients,  # Require at least 3 clients for training
        min_evaluate_clients=min_evaluate_clients,  # Require at least 3 clients for evaluation
        min_available_clients=min_available_clients,  # Wait for at least 3 clients to connect
        
        # Configure client training parameters
        on_fit_config_fn=lambda round_num: {
            "local_epochs": local_epochs,
            "learning_rate": learning_rate,
            "round": round_num
        },
        
        # Configure client evaluation parameters
        on_evaluate_config_fn=lambda round_num: {
            "round": round_num
        },
        
        # Use weighted average for metrics aggregation
        evaluate_metrics_aggregation_fn=weighted_average,
    )
    
    return strategy


def start_federated_server(
    server_address: str = "localhost:8080",
    num_rounds: int = 10,
    local_epochs: int = 5,
    learning_rate: float = 0.001
):
    """
    Start the federated learning server
    
    This function starts the Flower server that will coordinate federated learning.
    The server will:
    1. Wait for clients to connect
    2. Distribute global model to clients
    3. Collect updated weights from clients
    4. Aggregate weights using FedAvg
    5. Repeat for multiple rounds
    
    Args:
        server_address: Address and port for the server (format: "host:port")
        num_rounds: Number of federated learning rounds
        local_epochs: Number of local training epochs per round
        learning_rate: Learning rate for clients
    """
    print("=" * 60)
    print("FEDERATED LEARNING SERVER - PHASE 3")
    print("=" * 60)
    print(f"Server Address: {server_address}")
    print(f"Number of Rounds: {num_rounds}")
    print(f"Local Epochs per Round: {local_epochs}")
    print(f"Learning Rate: {learning_rate}")
    print("=" * 60)
    
    print("\nFEDERATED LEARNING PROCESS:")
    print("1. Server initializes global model")
    print("2. Clients connect and receive global model")
    print("3. Clients train locally on their private data")
    print("4. Clients send updated weights (NOT data) to server")
    print("5. Server aggregates weights using FedAvg")
    print("6. Server distributes aggregated model to clients")
    print("7. Repeat for multiple rounds")
    print("\nPRIVACY: Raw data never leaves clients!")
    print("=" * 60)
    
    # Create strategy
    strategy = create_federated_server(
        num_rounds=num_rounds,
        local_epochs=local_epochs,
        learning_rate=learning_rate
    )
    
    # Parse server address
    host, port = server_address.split(":")
    port = int(port)
    
    # Start server
    print(f"\nStarting server on {host}:{port}...")
    print("Waiting for clients to connect...")
    print("(Start clients in separate terminals/processes)\n")
    
    fl.server.start_server(
        server_address=f"{host}:{port}",
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    # Default configuration
    import argparse
    
    parser = argparse.ArgumentParser(description="Federated Learning Server")
    parser.add_argument("--address", type=str, default="localhost:8080", 
                       help="Server address (default: localhost:8080)")
    parser.add_argument("--rounds", type=int, default=10, 
                       help="Number of federated rounds (default: 10)")
    parser.add_argument("--epochs", type=int, default=5, 
                       help="Local epochs per round (default: 5)")
    parser.add_argument("--lr", type=float, default=0.001, 
                       help="Learning rate (default: 0.001)")
    
    args = parser.parse_args()
    
    start_federated_server(
        server_address=args.address,
        num_rounds=args.rounds,
        local_epochs=args.epochs,
        learning_rate=args.lr
    )

"""
Main Execution Script for Federated Learning Heart Disease Prediction
Simulates multiple clients on a single machine
"""

import os
import sys
import pandas as pd
import numpy as np
import subprocess
import time
import multiprocessing
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from client.client import create_client
from client.model import HeartDiseaseMLP
from utils.evaluation import evaluate_model_comprehensive
import torch


def split_dataset_for_federated_learning(
    dataset_path: str,
    num_clients: int = 3,
    output_dir: str = "dataset/clients"
):
    """
    Split the main dataset into multiple client datasets for federated learning simulation
    
    Args:
        dataset_path: Path to the main dataset CSV file
        num_clients: Number of clients to create
        output_dir: Directory to save client datasets
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load main dataset
    print(f"Loading dataset from {dataset_path}...")
    df = pd.read_csv(dataset_path)
    print(f"Total samples: {len(df)}")
    
    # Shuffle dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split into approximately equal parts for each client
    samples_per_client = len(df) // num_clients
    
    for i in range(num_clients):
        start_idx = i * samples_per_client
        if i == num_clients - 1:
            # Last client gets remaining samples
            end_idx = len(df)
        else:
            end_idx = (i + 1) * samples_per_client
        
        client_df = df.iloc[start_idx:end_idx].copy()
        client_path = os.path.join(output_dir, f"client_{i+1}_data.csv")
        client_df.to_csv(client_path, index=False)
        print(f"Client {i+1}: {len(client_df)} samples -> {client_path}")
    
    print(f"\nDataset split complete! Created {num_clients} client datasets.")


def run_client(client_id: int, data_path: str, server_address: str = "localhost:8080"):
    """
    Run a single client in a separate process
    
    Args:
        client_id: Client identifier
        data_path: Path to client's dataset
        server_address: Server address to connect to
    """
    import flwr as fl
    
    print(f"\n[Client {client_id}] Starting client...")
    
    # Create client
    client = create_client(
        client_id=client_id,
        data_path=data_path,
        local_epochs=5,
        learning_rate=0.001,
        apply_smote=True
    )
    
    # Connect to server
    print(f"[Client {client_id}] Connecting to server at {server_address}...")
    fl.client.start_numpy_client(
        server_address=server_address,
        client=client
    )
    
    print(f"[Client {client_id}] Training complete!")


def run_server(num_rounds: int = 10, server_address: str = "localhost:8080"):
    """
    Run the federated learning server
    
    Args:
        num_rounds: Number of federated learning rounds
        server_address: Server address
    """
    from server.server import start_federated_server
    
    start_federated_server(
        server_address=server_address,
        num_rounds=num_rounds,
        local_epochs=5,
        learning_rate=0.001
    )


def evaluate_global_model(
    dataset_path: str,
    model_weights_path: str = None,
    num_features: int = None
):
    """
    Evaluate the final global model on the complete test dataset
    
    Args:
        dataset_path: Path to the complete dataset
        model_weights_path: Optional path to saved model weights
        num_features: Number of input features
    """
    from client.data_preprocessing import DataPreprocessor
    
    print("\n" + "=" * 60)
    print("EVALUATING GLOBAL MODEL")
    print("=" * 60)
    
    # Preprocess complete dataset
    preprocessor = DataPreprocessor(client_id=0, apply_smote=False)
    df = preprocessor.load_data(dataset_path)
    train_loader, test_loader, num_features = preprocessor.prepare_data(df, test_size=0.2)
    
    # Create model
    model = HeartDiseaseMLP(input_size=num_features)
    
    # Load weights if provided
    if model_weights_path and os.path.exists(model_weights_path):
        print(f"Loading model weights from {model_weights_path}...")
        model.load_state_dict(torch.load(model_weights_path))
    else:
        print("Warning: No model weights provided. Using random initialization.")
    
    # Evaluate model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    metrics = evaluate_model_comprehensive(
        model=model,
        test_loader=test_loader,
        device=device,
        plot_results=True,
        save_plots=True,
        save_prefix="results/global_model"
    )
    
    return metrics


def main():
    """
    Main execution function
    """
    print("=" * 60)
    print("FEDERATED LEARNING FOR HEART DISEASE PREDICTION")
    print("=" * 60)
    
    # Configuration
    dataset_path = "dataset/heart.csv"
    num_clients = 3
    num_rounds = 10
    server_address = "localhost:8080"
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        print("Please ensure the heart.csv file is in the dataset/ directory")
        return
    
    # Step 1: Split dataset for federated learning
    print("\n[Step 1] Splitting dataset for federated learning...")
    split_dataset_for_federated_learning(
        dataset_path=dataset_path,
        num_clients=num_clients,
        output_dir="dataset/clients"
    )
    
    # Step 2: Start server in a separate process
    print("\n[Step 2] Starting federated learning server...")
    print("Note: Server will run in a separate process")
    
    server_process = multiprocessing.Process(
        target=run_server,
        args=(num_rounds, server_address),
        daemon=False
    )
    server_process.start()
    time.sleep(5)  # Wait for server to start
    
    # Step 3: Start clients in separate processes
    print("\n[Step 3] Starting federated learning clients...")
    client_processes = []
    for i in range(num_clients):
        client_data_path = f"dataset/clients/client_{i+1}_data.csv"
        if os.path.exists(client_data_path):
            client_process = multiprocessing.Process(
                target=run_client,
                args=(i+1, client_data_path, server_address),
                daemon=False
            )
            client_process.start()
            client_processes.append(client_process)
            time.sleep(2)  # Stagger client connections
    
    # Wait for all clients to complete
    print("\n[Step 4] Waiting for federated learning to complete...")
    print("This may take several minutes depending on dataset size and number of rounds...")
    
    for process in client_processes:
        process.join(timeout=600)  # 10 minute timeout per client
        if process.is_alive():
            print(f"Warning: Client process timed out")
    
    # Wait for server to finish
    server_process.join(timeout=60)
    
    print("\n" + "=" * 60)
    print("FEDERATED LEARNING COMPLETE!")
    print("=" * 60)
    print("\nNote: To evaluate the final model, use the evaluation script separately.")
    print("The model weights would need to be saved during training for full evaluation.")
    print("\nAlternative: Run server and clients separately using:")
    print("  Terminal 1: python run_server.py")
    print("  Terminal 2: python run_client.py --client_id 1 --data_path dataset/clients/client_1_data.csv")
    print("  Terminal 3: python run_client.py --client_id 2 --data_path dataset/clients/client_2_data.csv")
    print("  Terminal 4: python run_client.py --client_id 3 --data_path dataset/clients/client_3_data.csv")


if __name__ == "__main__":
    main()


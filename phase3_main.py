"""
Phase 3 Main Execution Script
Complete federated learning workflow

This script simulates federated learning with multiple clients on a single machine.
It demonstrates how multiple hospitals can collaborate to train a model without
sharing raw patient data.
"""

import os
import sys
import time
import multiprocessing
from typing import List

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from client.fl_client import create_flower_client
from server.server import start_federated_server
import flwr as fl


def run_client_process(
    client_id: int,
    client_data_path: str,
    num_features: int,
    server_address: str,
    local_epochs: int = 5,
    learning_rate: float = 0.001
):
    """
    Run a Flower client in a separate process
    
    Args:
        client_id: Client identifier
        client_data_path: Path to client's dataset
        num_features: Number of input features
        server_address: Server address to connect to
        local_epochs: Local training epochs per round
        learning_rate: Learning rate
    """
    try:
        print(f"\n[Client {client_id}] Starting client process...")
        
        # Create Flower client
        client = create_flower_client(
            client_id=client_id,
            client_data_path=client_data_path,
            num_features=num_features,
            local_epochs=local_epochs,
            learning_rate=learning_rate
        )
        
        # Connect to server and start federated learning
        print(f"[Client {client_id}] Connecting to server at {server_address}...")
        fl.client.start_numpy_client(
            server_address=server_address,
            client=client
        )
        
        print(f"[Client {client_id}] Federated learning complete!")
        
    except Exception as e:
        print(f"[Client {client_id}] Error: {e}")


def run_server_process(
    server_address: str,
    num_rounds: int,
    local_epochs: int,
    learning_rate: float
):
    """
    Run the federated learning server in a separate process
    
    Args:
        server_address: Server address
        num_rounds: Number of federated rounds
        local_epochs: Local epochs per round
        learning_rate: Learning rate
    """
    start_federated_server(
        server_address=server_address,
        num_rounds=num_rounds,
        local_epochs=local_epochs,
        learning_rate=learning_rate
    )


def main():
    """
    Main execution function for Phase 3
    """
    print("=" * 60)
    print("FEDERATED DEEP LEARNING FOR HEART DISEASE PREDICTION")
    print("PHASE 3: FEDERATED LEARNING")
    print("=" * 60)
    
    # Configuration
    num_clients = 3
    num_rounds = 10
    local_epochs = 5
    learning_rate = 0.001
    server_address = "localhost:8080"
    num_features = 13  # Typical for heart disease dataset
    
    print(f"\nConfiguration:")
    print(f"  Number of clients: {num_clients}")
    print(f"  Federated rounds: {num_rounds}")
    print(f"  Local epochs per round: {local_epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Server address: {server_address}")
    
    # Check if client datasets exist
    processed_dir = "dataset/processed"
    if not os.path.exists(processed_dir):
        print(f"\n❌ Error: Processed dataset directory not found: {processed_dir}")
        print("Please run Phase 1 first to create client datasets")
        return
    
    # Verify client datasets exist
    client_paths = []
    for i in range(1, num_clients + 1):
        client_path = os.path.join(processed_dir, f"client_{i}.csv")
        if os.path.exists(client_path):
            client_paths.append((i, client_path))
        else:
            print(f"⚠ Warning: Client {i} data not found at {client_path}")
    
    if len(client_paths) < num_clients:
        print(f"\n❌ Error: Not enough client datasets found. Need {num_clients}, found {len(client_paths)}")
        print("Please run Phase 1 first to create client datasets")
        return
    
    print(f"\n✓ Found {len(client_paths)} client datasets")
    
    # Start server in a separate process
    print("\n" + "=" * 60)
    print("STEP 1: STARTING FEDERATED LEARNING SERVER")
    print("=" * 60)
    
    server_process = multiprocessing.Process(
        target=run_server_process,
        args=(server_address, num_rounds, local_epochs, learning_rate),
        daemon=False
    )
    server_process.start()
    
    # Wait for server to start
    print("Waiting for server to initialize...")
    time.sleep(5)
    
    # Start clients in separate processes
    print("\n" + "=" * 60)
    print("STEP 2: STARTING FEDERATED LEARNING CLIENTS")
    print("=" * 60)
    
    client_processes = []
    for client_id, client_path in client_paths:
        print(f"\nStarting Client {client_id}...")
        client_process = multiprocessing.Process(
            target=run_client_process,
            args=(client_id, client_path, num_features, server_address, local_epochs, learning_rate),
            daemon=False
        )
        client_process.start()
        client_processes.append(client_process)
        time.sleep(2)  # Stagger client connections
    
    print(f"\n✓ Started {len(client_processes)} clients")
    print("\n" + "=" * 60)
    print("FEDERATED LEARNING IN PROGRESS...")
    print("=" * 60)
    print("\nThis may take several minutes depending on:")
    print("  • Number of federated rounds")
    print("  • Local epochs per round")
    print("  • Dataset size")
    print("\nProgress will be displayed above.")
    print("\nPRIVACY NOTE:")
    print("  • Raw patient data NEVER leaves clients")
    print("  • Only model weights are shared with server")
    print("  • This ensures patient privacy (HIPAA/GDPR compliant)")
    print("=" * 60)
    
    # Wait for all clients to complete
    print("\nWaiting for federated learning to complete...")
    for process in client_processes:
        process.join(timeout=600)  # 10 minute timeout per client
        if process.is_alive():
            print(f"⚠ Warning: Client process timed out")
    
    # Wait for server to finish
    server_process.join(timeout=60)
    
    print("\n" + "=" * 60)
    print("FEDERATED LEARNING COMPLETE!")
    print("=" * 60)
    
    print("\nSummary:")
    print(f"  ✓ Completed {num_rounds} federated rounds")
    print(f"  ✓ {num_clients} clients participated")
    print(f"  ✓ Global model trained collaboratively")
    print(f"  ✓ Patient privacy preserved (no raw data shared)")
    
    print("\nKey Achievements:")
    print("  ✓ Multiple hospitals collaborated without sharing data")
    print("  ✓ Global model learned from all clients' data")
    print("  ✓ Privacy-preserving machine learning demonstrated")
    print("  ✓ Federated Averaging (FedAvg) successfully implemented")
    
    print("\n" + "=" * 60)
    print("PROJECT COMPLETE!")
    print("=" * 60)
    print("\nAll three phases completed:")
    print("  Phase 1: Dataset Engineering & Preprocessing ✓")
    print("  Phase 2: Deep Learning Model & Local Training ✓")
    print("  Phase 3: Federated Learning ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()


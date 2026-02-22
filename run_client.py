"""
Standalone script to run a Federated Learning Client
Run this in a separate terminal/process for each client

Usage:
    python run_client.py --client_id 1 --data_path dataset/clients/client_1_data.csv
"""

import sys
import os
import argparse
import flwr as fl

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from client.fl_client import create_flower_client

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated Learning Client")
    parser.add_argument("--client_id", type=int, required=True,
                       help="Client ID (e.g., 1, 2, 3)")
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to client's dataset CSV file")
    parser.add_argument("--server_address", type=str, default="localhost:8080",
                       help="Server address (default: localhost:8080)")
    parser.add_argument("--epochs", type=int, default=5,
                       help="Local epochs per round (default: 5)")
    parser.add_argument("--lr", type=float, default=0.001,
                       help="Learning rate (default: 0.001)")
    parser.add_argument("--no_smote", action="store_true",
                       help="Disable SMOTE for class balancing")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"FEDERATED LEARNING CLIENT {args.client_id}")
    print("=" * 60)
    print(f"Data Path: {args.data_path}")
    print(f"Server: {args.server_address}")
    print(f"Local Epochs: {args.epochs}")
    print(f"Learning Rate: {args.lr}")
    print("=" * 60)
    
    # Check if data file exists
    if not os.path.exists(args.data_path):
        print(f"Error: Data file not found at {args.data_path}")
        sys.exit(1)
    
    # Create client
    print(f"\n[Client {args.client_id}] Initializing...")
    client = create_flower_client(
        client_id=args.client_id,
        client_data_path=args.data_path,
        num_features=None,
        local_epochs=args.epochs,
        learning_rate=args.lr
    )
    
    # Connect to server and start training
    print(f"[Client {args.client_id}] Connecting to server at {args.server_address}...")
    try:
        fl.client.start_numpy_client(
            server_address=args.server_address,
            client=client
        )
        print(f"\n[Client {args.client_id}] Training completed successfully!")
    except Exception as e:
        print(f"\n[Client {args.client_id}] Error: {e}")
        sys.exit(1)


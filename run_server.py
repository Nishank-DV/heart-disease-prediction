"""
Standalone script to run the Federated Learning Server
Run this in a separate terminal/process
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from server.server import start_federated_server
import argparse

if __name__ == "__main__":
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


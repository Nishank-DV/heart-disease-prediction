"""
Simplified script to run federated learning using subprocess
This is an alternative to main.py that uses separate processes more reliably
"""

import os
import sys
import subprocess
import time
import pandas as pd

def split_dataset_for_federated_learning(
    dataset_path: str,
    num_clients: int = 3,
    output_dir: str = "dataset/clients"
):
    """Split dataset into client-specific datasets"""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading dataset from {dataset_path}...")
    df = pd.read_csv(dataset_path)
    print(f"Total samples: {len(df)}")
    
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    samples_per_client = len(df) // num_clients
    
    for i in range(num_clients):
        start_idx = i * samples_per_client
        end_idx = (i + 1) * samples_per_client if i < num_clients - 1 else len(df)
        
        client_df = df.iloc[start_idx:end_idx].copy()
        client_path = os.path.join(output_dir, f"client_{i+1}_data.csv")
        client_df.to_csv(client_path, index=False)
        print(f"Client {i+1}: {len(client_df)} samples -> {client_path}")
    
    print(f"\nDataset split complete! Created {num_clients} client datasets.")


def main():
    print("=" * 60)
    print("FEDERATED LEARNING FOR HEART DISEASE PREDICTION")
    print("=" * 60)
    
    dataset_path = "dataset/heart.csv"
    num_clients = 3
    num_rounds = 10
    
    # Check dataset
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        print("Run: python download_dataset.py")
        return
    
    # Split dataset
    print("\n[Step 1] Splitting dataset...")
    split_dataset_for_federated_learning(
        dataset_path=dataset_path,
        num_clients=num_clients,
        output_dir="dataset/clients"
    )
    
    # Start server
    print("\n[Step 2] Starting server...")
    server_process = subprocess.Popen(
        [sys.executable, "run_server.py", "--rounds", str(num_rounds)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    time.sleep(5)
    
    # Start clients
    print("\n[Step 3] Starting clients...")
    client_processes = []
    for i in range(num_clients):
        client_path = f"dataset/clients/client_{i+1}_data.csv"
        if os.path.exists(client_path):
            client_process = subprocess.Popen(
                [sys.executable, "run_client.py", 
                 "--client_id", str(i+1),
                 "--data_path", client_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            client_processes.append(client_process)
            time.sleep(2)
    
    print("\n[Step 4] Federated learning in progress...")
    print("Waiting for all processes to complete...")
    
    # Wait for clients
    for i, process in enumerate(client_processes, 1):
        process.wait()
        print(f"Client {i} completed")
    
    # Wait for server
    server_process.wait()
    print("Server completed")
    
    print("\n" + "=" * 60)
    print("FEDERATED LEARNING COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()


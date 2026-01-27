"""
Dataset Splitter Module
Splits the full dataset into multiple client datasets for federated learning simulation
"""

import pandas as pd
import numpy as np
import os
from typing import List, Tuple


def split_dataset_for_clients(
    dataset_path: str,
    num_clients: int = 3,
    output_dir: str = "dataset/processed",
    random_state: int = 42,
    stratify: bool = True
) -> List[str]:
    """
    Split the full dataset into multiple client datasets
    
    This function simulates the scenario where multiple hospitals (clients)
    have their own local datasets. In real federated learning, each client
    would have their own data that never leaves their premises.
    
    Args:
        dataset_path: Path to the full dataset CSV file
        num_clients: Number of clients to create (default: 3)
        output_dir: Directory to save client datasets
        random_state: Random seed for reproducibility
        stratify: Whether to maintain class distribution in each client
        
    Returns:
        List of paths to created client dataset files
        
    Raises:
        FileNotFoundError: If the dataset file doesn't exist
    """
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at: {dataset_path}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    print(f"Loading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)
    print(f"[OK] Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")
    
    # Identify target column
    target_col = identify_target_column(df)
    if target_col is None:
        print("Warning: Target column not found. Splitting without stratification.")
        stratify = False
    
    # Shuffle dataset for random distribution
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Split dataset into client-specific subsets
    client_paths = []
    
    if stratify and target_col:
        # Stratified split to maintain class distribution
        print(f"\nPerforming stratified split (maintaining class distribution)...")
        client_datasets = stratified_split(df, num_clients, target_col, random_state)
    else:
        # Simple random split
        print(f"\nPerforming random split...")
        client_datasets = random_split(df, num_clients)
    
    # Save each client dataset
    print(f"\nSaving client datasets to: {output_dir}")
    for i, client_df in enumerate(client_datasets, 1):
        client_path = os.path.join(output_dir, f"client_{i}.csv")
        client_df.to_csv(client_path, index=False)
        client_paths.append(client_path)
        
        # Print client statistics
        print(f"\n  Client {i}:")
        print(f"    Samples: {len(client_df)}")
        print(f"    Features: {len(client_df.columns)}")
        if target_col:
            target_dist = client_df[target_col].value_counts()
            print(f"    Class distribution:")
            for class_val, count in target_dist.items():
                percentage = (count / len(client_df)) * 100
                print(f"      Class {class_val}: {count} ({percentage:.2f}%)")
        print(f"    Saved to: {client_path}")
    
    print(f"\n[OK] Successfully created {num_clients} client datasets!")
    print("\n" + "=" * 60)
    print("FEDERATED LEARNING SIMULATION SETUP")
    print("=" * 60)
    print("\nThis simulates multiple hospitals participating in federated learning:")
    print("  • Each client (hospital) has its own local dataset")
    print("  • Raw data never leaves the client's premises")
    print("  • Only model parameters will be shared (in Phase 2)")
    print("  • This design ensures patient data privacy")
    print("=" * 60)
    
    return client_paths


def identify_target_column(df: pd.DataFrame) -> str:
    """
    Identify the target column in the dataset
    
    Args:
        df: Input DataFrame
        
    Returns:
        Name of the target column, or None if not found
    """
    possible_targets = ['target', 'HeartDisease', 'heart_disease', 'disease', 'condition']
    
    for col in possible_targets:
        if col in df.columns:
            return col
    
    # If not found, assume last column is target
    if len(df.columns) > 0:
        return df.columns[-1]
    
    return None


def stratified_split(
    df: pd.DataFrame,
    num_clients: int,
    target_col: str,
    random_state: int = 42
) -> List[pd.DataFrame]:
    """
    Split dataset into clients while maintaining class distribution
    
    Args:
        df: Input DataFrame
        num_clients: Number of clients
        target_col: Name of target column
        random_state: Random seed
        
    Returns:
        List of DataFrames, one for each client
    """
    from sklearn.model_selection import train_test_split
    
    client_datasets = []
    
    # Get unique classes
    unique_classes = df[target_col].unique()
    
    # Split each class separately to maintain distribution
    for class_val in unique_classes:
        class_data = df[df[target_col] == class_val].copy()
        class_data = class_data.sample(frac=1, random_state=random_state).reset_index(drop=True)
        
        # Calculate samples per client for this class
        samples_per_client = len(class_data) // num_clients
        
        for i in range(num_clients):
            start_idx = i * samples_per_client
            if i == num_clients - 1:
                # Last client gets remaining samples
                end_idx = len(class_data)
            else:
                end_idx = (i + 1) * samples_per_client
            
            client_samples = class_data.iloc[start_idx:end_idx]
            
            if i >= len(client_datasets):
                client_datasets.append(client_samples)
            else:
                client_datasets[i] = pd.concat([client_datasets[i], client_samples], ignore_index=True)
    
    # Shuffle each client dataset
    for i in range(len(client_datasets)):
        client_datasets[i] = client_datasets[i].sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    return client_datasets


def random_split(
    df: pd.DataFrame,
    num_clients: int
) -> List[pd.DataFrame]:
    """
    Split dataset randomly into clients
    
    Args:
        df: Input DataFrame
        num_clients: Number of clients
        
    Returns:
        List of DataFrames, one for each client
    """
    client_datasets = []
    samples_per_client = len(df) // num_clients
    
    for i in range(num_clients):
        start_idx = i * samples_per_client
        if i == num_clients - 1:
            # Last client gets remaining samples
            end_idx = len(df)
        else:
            end_idx = (i + 1) * samples_per_client
        
        client_df = df.iloc[start_idx:end_idx].copy()
        client_datasets.append(client_df)
    
    return client_datasets


def verify_client_datasets(client_paths: List[str]) -> bool:
    """
    Verify that all client datasets are valid and have consistent structure
    
    Args:
        client_paths: List of paths to client dataset files
        
    Returns:
        True if all datasets are valid, False otherwise
    """
    print("\n" + "=" * 60)
    print("VERIFYING CLIENT DATASETS")
    print("=" * 60)
    
    if not client_paths:
        print("[ERROR] No client datasets found!")
        return False
    
    # Load first client to get reference structure
    reference_df = pd.read_csv(client_paths[0])
    reference_columns = set(reference_df.columns)
    reference_shape = reference_df.shape[1]
    
    print(f"\nReference dataset: {client_paths[0]}")
    print(f"  Columns: {reference_shape}")
    print(f"  Column names: {list(reference_columns)}")
    
    all_valid = True
    
    for i, path in enumerate(client_paths, 1):
        if not os.path.exists(path):
            print(f"\n[ERROR] Client {i}: File not found at {path}")
            all_valid = False
            continue
        
        try:
            df = pd.read_csv(path)
            current_columns = set(df.columns)
            
            if current_columns != reference_columns:
                print(f"\n[ERROR] Client {i}: Column mismatch!")
                print(f"  Expected: {reference_columns}")
                print(f"  Found: {current_columns}")
                all_valid = False
            elif df.shape[1] != reference_shape:
                print(f"\n[ERROR] Client {i}: Feature count mismatch!")
                print(f"  Expected: {reference_shape}, Found: {df.shape[1]}")
                all_valid = False
            else:
                print(f"\n[OK] Client {i}: Valid ({df.shape[0]} samples, {df.shape[1]} features)")
        
        except Exception as e:
            print(f"\n[ERROR] Client {i}: Error reading file - {e}")
            all_valid = False
    
    if all_valid:
        print("\n[OK] All client datasets are valid and consistent!")
    
    return all_valid


if __name__ == "__main__":
    # Example usage
    dataset_path = "dataset/raw/heart.csv"
    output_dir = "dataset/processed"
    num_clients = 3
    
    try:
        print("=" * 60)
        print("DATASET SPLITTING FOR FEDERATED LEARNING")
        print("=" * 60)
        
        client_paths = split_dataset_for_clients(
            dataset_path=dataset_path,
            num_clients=num_clients,
            output_dir=output_dir,
            random_state=42,
            stratify=True
        )
        
        # Verify datasets
        verify_client_datasets(client_paths)
        
        print("\n" + "=" * 60)
        print("SPLITTING COMPLETE!")
        print("=" * 60)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the dataset is placed at: dataset/raw/heart.csv")


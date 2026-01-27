"""
Script to download the Heart Disease dataset
This script helps download the UCI Heart Disease dataset
"""

import os
import urllib.request
import pandas as pd
from pathlib import Path


def download_heart_disease_dataset():
    """
    Download the Heart Disease dataset from UCI ML Repository
    """
    # Create dataset directory
    os.makedirs("dataset", exist_ok=True)
    
    dataset_path = "dataset/heart.csv"
    
    # Check if dataset already exists
    if os.path.exists(dataset_path):
        print(f"Dataset already exists at {dataset_path}")
        return dataset_path
    
    print("Downloading Heart Disease dataset...")
    print("Note: This script attempts to download from common sources.")
    print("If automatic download fails, please manually download from:")
    print("  - Kaggle: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset")
    print("  - UCI ML: https://archive.ics.uci.edu/ml/datasets/heart+disease")
    
    # Try to download from a public source
    # Note: Direct download URLs may change, so manual download is recommended
    try:
        # Common dataset URL (may need to be updated)
        url = "https://raw.githubusercontent.com/plotly/datasets/master/heart.csv"
        
        print(f"Attempting to download from: {url}")
        urllib.request.urlretrieve(url, dataset_path)
        print(f"Dataset downloaded successfully to {dataset_path}")
        
        # Verify the dataset
        df = pd.read_csv(dataset_path)
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        return dataset_path
        
    except Exception as e:
        print(f"Automatic download failed: {e}")
        print("\nPlease manually download the dataset:")
        print("1. Visit: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset")
        print("2. Download the dataset")
        print(f"3. Save it as: {dataset_path}")
        return None


def create_sample_dataset():
    """
    Create a sample dataset if download fails
    This is a fallback option with synthetic data
    """
    import numpy as np
    
    dataset_path = "dataset/heart.csv"
    
    if os.path.exists(dataset_path):
        print(f"Dataset already exists at {dataset_path}")
        return dataset_path
    
    print("Creating sample dataset for testing...")
    print("Note: This is synthetic data. For real results, use the actual UCI dataset.")
    
    # Generate synthetic heart disease data
    np.random.seed(42)
    n_samples = 1000
    
    # Features based on UCI Heart Disease dataset
    data = {
        'age': np.random.randint(29, 80, n_samples),
        'sex': np.random.randint(0, 2, n_samples),
        'cp': np.random.randint(0, 4, n_samples),  # chest pain type
        'trestbps': np.random.randint(94, 200, n_samples),  # resting blood pressure
        'chol': np.random.randint(126, 564, n_samples),  # serum cholesterol
        'fbs': np.random.randint(0, 2, n_samples),  # fasting blood sugar
        'restecg': np.random.randint(0, 3, n_samples),  # resting ECG
        'thalach': np.random.randint(71, 202, n_samples),  # max heart rate
        'exang': np.random.randint(0, 2, n_samples),  # exercise induced angina
        'oldpeak': np.random.uniform(0, 6.2, n_samples),  # ST depression
        'slope': np.random.randint(0, 3, n_samples),  # slope of peak exercise
        'ca': np.random.randint(0, 4, n_samples),  # number of major vessels
        'thal': np.random.randint(0, 4, n_samples),  # thalassemia
    }
    
    # Create target with some correlation to features
    # Higher age, cholesterol, and oldpeak increase disease probability
    disease_prob = (
        0.1 * (data['age'] - 50) / 30 +
        0.2 * (data['chol'] - 250) / 200 +
        0.3 * data['oldpeak'] / 6.2 +
        0.1 * data['cp'] / 3 +
        np.random.normal(0, 0.2, n_samples)
    )
    data['target'] = (disease_prob > 0.5).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv(dataset_path, index=False)
    print(f"Sample dataset created at {dataset_path}")
    print(f"Dataset shape: {df.shape}")
    
    return dataset_path


if __name__ == "__main__":
    print("=" * 60)
    print("HEART DISEASE DATASET DOWNLOADER")
    print("=" * 60)
    
    # Try to download
    path = download_heart_disease_dataset()
    
    # If download fails, create sample dataset
    if path is None:
        print("\nFalling back to sample dataset creation...")
        path = create_sample_dataset()
    
    if path:
        print(f"\n✓ Dataset ready at: {path}")
        print("You can now run: python main.py")
    else:
        print("\n✗ Failed to obtain dataset. Please download manually.")


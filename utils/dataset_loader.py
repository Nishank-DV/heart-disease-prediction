"""
Dataset Loader Module
Loads and analyzes the Heart Disease dataset for Phase 1
"""

import pandas as pd
import numpy as np
import os
from typing import Tuple, Dict, List


def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load the heart disease dataset from CSV file
    
    Args:
        file_path: Path to the heart.csv file
        
    Returns:
        DataFrame containing the dataset
        
    Raises:
        FileNotFoundError: If the dataset file doesn't exist
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at: {file_path}")
    
    print(f"Loading dataset from: {file_path}")
    df = pd.read_csv(file_path)
    print(f"[OK] Dataset loaded successfully!")
    
    return df


def analyze_dataset(df: pd.DataFrame) -> Dict:
    """
    Perform comprehensive analysis of the dataset
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary containing analysis results
    """
    print("\n" + "=" * 60)
    print("DATASET ANALYSIS")
    print("=" * 60)
    
    # Basic information
    print(f"\nDataset Shape: {df.shape}")
    print(f"  - Rows (samples): {df.shape[0]}")
    print(f"  - Columns (features): {df.shape[1]}")
    
    # Column names
    print(f"\nColumn Names ({len(df.columns)}):")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:2d}. {col}")
    
    # Data types
    print(f"\nData Types:")
    print(df.dtypes)
    
    # Missing values
    missing_values = df.isnull().sum()
    total_missing = missing_values.sum()
    
    print(f"\nMissing Values Analysis:")
    if total_missing > 0:
        print(f"  Total missing values: {total_missing}")
        print("\n  Missing values per column:")
        for col, count in missing_values.items():
            if count > 0:
                percentage = (count / len(df)) * 100
                print(f"    {col}: {count} ({percentage:.2f}%)")
    else:
        print("  [OK] No missing values found!")
    
    # Identify numerical and categorical features
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Remove target from numerical if it exists
    target_col = identify_target_column(df)
    if target_col in numerical_cols:
        numerical_cols.remove(target_col)
    
    print(f"\nFeature Types:")
    print(f"  Numerical features ({len(numerical_cols)}): {numerical_cols}")
    print(f"  Categorical features ({len(categorical_cols)}): {categorical_cols}")
    
    # Target column analysis
    if target_col:
        print(f"\nTarget Column: {target_col}")
        target_distribution = df[target_col].value_counts()
        print(f"  Class distribution:")
        for class_val, count in target_distribution.items():
            percentage = (count / len(df)) * 100
            print(f"    Class {class_val}: {count} samples ({percentage:.2f}%)")
        
        # Imbalance ratio
        if len(target_distribution) == 2:
            minority = min(target_distribution.values)
            majority = max(target_distribution.values)
            imbalance_ratio = majority / minority
            print(f"  Imbalance ratio: {imbalance_ratio:.2f}:1")
    
    # Statistical summary
    print(f"\nStatistical Summary (Numerical Features):")
    if numerical_cols:
        print(df[numerical_cols].describe())
    
    # Store analysis results
    analysis = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'numerical_features': numerical_cols,
        'categorical_features': categorical_cols,
        'target_column': target_col,
        'missing_values': missing_values.to_dict(),
        'total_missing': total_missing,
        'dtypes': df.dtypes.to_dict()
    }
    
    if target_col:
        analysis['target_distribution'] = df[target_col].value_counts().to_dict()
    
    return analysis


def identify_target_column(df: pd.DataFrame) -> str:
    """
    Identify the target column in the dataset
    
    Args:
        df: Input DataFrame
        
    Returns:
        Name of the target column, or None if not found
    """
    # Common target column names
    possible_targets = ['target', 'HeartDisease', 'heart_disease', 'disease', 'condition']
    
    for col in possible_targets:
        if col in df.columns:
            return col
    
    # If not found, assume last column is target
    if len(df.columns) > 0:
        print(f"Warning: Target column not found in common names. Assuming last column: {df.columns[-1]}")
        return df.columns[-1]
    
    return None


def get_feature_descriptions() -> Dict[str, str]:
    """
    Get medical descriptions for each feature in the heart disease dataset
    
    Returns:
        Dictionary mapping feature names to their medical descriptions
    """
    descriptions = {
        'age': 'Age of the patient in years',
        'sex': 'Sex of the patient (0 = female, 1 = male)',
        'cp': 'Chest pain type (0 = typical angina, 1 = atypical angina, 2 = non-anginal pain, 3 = asymptomatic)',
        'trestbps': 'Resting blood pressure in mm Hg on admission to hospital',
        'chol': 'Serum cholesterol in mg/dl',
        'fbs': 'Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)',
        'restecg': 'Resting electrocardiographic results (0 = normal, 1 = ST-T wave abnormality, 2 = left ventricular hypertrophy)',
        'thalach': 'Maximum heart rate achieved during exercise',
        'exang': 'Exercise induced angina (1 = yes, 0 = no)',
        'oldpeak': 'ST depression induced by exercise relative to rest',
        'slope': 'Slope of the peak exercise ST segment (0 = upsloping, 1 = flat, 2 = downsloping)',
        'ca': 'Number of major vessels colored by flourosopy (0-3)',
        'thal': 'Thalassemia type (0 = normal, 1 = fixed defect, 2 = reversable defect, 3 = unknown)',
        'target': 'Heart disease presence (0 = no disease, 1 = heart disease present)'
    }
    
    return descriptions


def print_feature_descriptions(df: pd.DataFrame):
    """
    Print medical descriptions for all features in the dataset
    
    Args:
        df: Input DataFrame
    """
    descriptions = get_feature_descriptions()
    
    print("\n" + "=" * 60)
    print("FEATURE DESCRIPTIONS (Medical Meaning)")
    print("=" * 60)
    
    for col in df.columns:
        if col in descriptions:
            print(f"\n{col.upper()}:")
            print(f"  {descriptions[col]}")
        else:
            print(f"\n{col.upper()}:")
            print(f"  (Description not available)")


def load_and_analyze(file_path: str) -> Tuple[pd.DataFrame, Dict]:
    """
    Load dataset and perform complete analysis
    
    Args:
        file_path: Path to the dataset CSV file
        
    Returns:
        Tuple of (DataFrame, analysis dictionary)
    """
    # Load dataset
    df = load_dataset(file_path)
    
    # Analyze dataset
    analysis = analyze_dataset(df)
    
    # Print feature descriptions
    print_feature_descriptions(df)
    
    return df, analysis


if __name__ == "__main__":
    # Example usage
    dataset_path = "dataset/raw/heart.csv"
    
    try:
        df, analysis = load_and_analyze(dataset_path)
        print("\n" + "=" * 60)
        print("DATASET LOADING COMPLETE")
        print("=" * 60)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the dataset is placed at: dataset/raw/heart.csv")


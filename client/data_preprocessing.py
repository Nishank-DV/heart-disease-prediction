"""
Data Preprocessing Module for Phase 1
Client-side data preprocessing pipeline for federated learning

IMPORTANT PRIVACY DESIGN:
- All preprocessing happens LOCALLY at each client
- No centralized preprocessing allowed
- This ensures raw data never leaves the client's premises
- Only preprocessed features (X, y) are used locally
- In Phase 2, only model parameters (weights) will be shared, not data
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict
import os
import sys

# Add utils to path for feature information
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.feature_info import feature_info


class DataPreprocessor:
    """
    Client-side data preprocessing pipeline
    
    This class implements a complete preprocessing pipeline that:
    1. Handles missing values
    2. Encodes categorical variables
    3. Scales numerical features
    4. Separates features and target
    5. Analyzes class imbalance
    
    All preprocessing is done LOCALLY at the client side.
    """
    
    def __init__(self, client_id: int = 0, apply_smote: bool = False):
        """
        Initialize the preprocessor
        
        Args:
            client_id: Unique identifier for the client (for logging)
            apply_smote: Whether to apply SMOTE (not implemented here)
        """
        self.client_id = client_id
        self.apply_smote = apply_smote
        self.scaler = StandardScaler()
        self.label_encoders = {}  # Store encoders for each categorical feature
        self.target_column = None
        self.feature_columns = None
        self.num_features = None
        
        print(f"[Client {self.client_id}] Preprocessor initialized")
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load dataset from CSV file
        
        Args:
            file_path: Path to the client's dataset CSV file
            
        Returns:
            DataFrame containing the dataset
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"[Client {self.client_id}] Dataset not found: {file_path}")
        
        print(f"[Client {self.client_id}] Loading dataset from: {file_path}")
        df = pd.read_csv(file_path)
        print(f"[Client {self.client_id}] [OK] Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")
        
        return df
    
    def identify_target_column(self, df: pd.DataFrame) -> str:
        """
        Identify the target column in the dataset
        
        Args:
            df: Input DataFrame
            
        Returns:
            Name of the target column
        """
        possible_targets = ['target', 'HeartDisease', 'heart_disease', 'disease', 'condition']
        
        for col in possible_targets:
            if col in df.columns:
                self.target_column = col
                return col
        
        # If not found, assume last column is target
        if len(df.columns) > 0:
            self.target_column = df.columns[-1]
            print(f"[Client {self.client_id}] Warning: Target column not found. Using: {self.target_column}")
            return self.target_column
        
        raise ValueError(f"[Client {self.client_id}] Could not identify target column")
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset
        
        Strategy:
        - Numerical features: Fill with median (robust to outliers)
        - Categorical features: Fill with mode (most frequent value)
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with missing values handled
        """
        df_cleaned = df.copy()
        missing_count = df_cleaned.isnull().sum().sum()
        
        if missing_count == 0:
            print(f"[Client {self.client_id}] [OK] No missing values found")
            return df_cleaned
        
        print(f"[Client {self.client_id}] Found {missing_count} missing values")
        
        # Handle numerical columns
        numerical_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df_cleaned[col].isnull().sum() > 0:
                median_val = df_cleaned[col].median()
                df_cleaned[col].fillna(median_val, inplace=True)
                print(f"[Client {self.client_id}]   Filled {col} (numerical) with median: {median_val:.2f}")
        
        # Handle categorical columns
        categorical_cols = df_cleaned.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df_cleaned[col].isnull().sum() > 0:
                mode_val = df_cleaned[col].mode()[0] if len(df_cleaned[col].mode()) > 0 else 'Unknown'
                df_cleaned[col].fillna(mode_val, inplace=True)
                print(f"[Client {self.client_id}]   Filled {col} (categorical) with mode: {mode_val}")
        
        print(f"[Client {self.client_id}] [OK] Missing values handled")
        return df_cleaned
    
    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical features using Label Encoding
        
        Label Encoding is chosen over One-Hot Encoding because:
        - Reduces dimensionality (important for federated learning)
        - Maintains feature count consistent across clients
        - Suitable for ordinal categorical features
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with encoded categorical features
        """
        df_encoded = df.copy()
        categorical_cols = df_encoded.select_dtypes(include=['object']).columns.tolist()
        
        # Remove target column if it's categorical
        if self.target_column in categorical_cols:
            categorical_cols.remove(self.target_column)
        
        if len(categorical_cols) == 0:
            print(f"[Client {self.client_id}] [OK] No categorical features to encode")
            return df_encoded
        
        print(f"[Client {self.client_id}] Encoding {len(categorical_cols)} categorical features...")
        
        for col in categorical_cols:
            if col in df_encoded.columns:
                # Create label encoder for this column
                encoder = LabelEncoder()
                df_encoded[col] = encoder.fit_transform(df_encoded[col].astype(str))
                
                # Store encoder for potential inverse transformation
                self.label_encoders[col] = encoder
                
                print(f"[Client {self.client_id}]   Encoded {col}: {len(encoder.classes_)} unique values")
        
        print(f"[Client {self.client_id}] [OK] Categorical features encoded")
        return df_encoded
    
    def separate_features_and_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Separate features (X) and target (y)
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        if self.target_column is None:
            self.identify_target_column(df)
        
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]
        
        # Ensure target is binary (0 or 1)
        if y.nunique() > 2:
            print(f"[Client {self.client_id}] Warning: Target has {y.nunique()} classes. Converting to binary.")
            y = (y > 0).astype(int)
        
        self.feature_columns = X.columns.tolist()
        self.num_features = X.shape[1]
        
        print(f"[Client {self.client_id}] [OK] Separated features and target")
        print(f"[Client {self.client_id}]   Features: {self.num_features}")
        print(f"[Client {self.client_id}]   Target distribution: {y.value_counts().to_dict()}")
        
        return X, y
    
    def normalize_features(self, X_train: np.ndarray, X_test: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Normalize numerical features using StandardScaler
        
        StandardScaler transforms features to have:
        - Mean = 0
        - Standard deviation = 1
        
        This is important for:
        - Neural network training (in Phase 2)
        - Feature comparison across clients
        - Preventing features with large scales from dominating
        
        IMPORTANT: Fit scaler only on training data to prevent data leakage
        
        Args:
            X_train: Training features
            X_test: Test features (optional)
            
        Returns:
            Tuple of (normalized X_train, normalized X_test)
        """
        print(f"[Client {self.client_id}] Normalizing features using StandardScaler...")
        
        # Fit scaler on training data only
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Transform test data using training statistics
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
        else:
            X_test_scaled = None
        
        print(f"[Client {self.client_id}] [OK] Features normalized")
        print(f"[Client {self.client_id}]   Training mean: {X_train_scaled.mean(axis=0)[:3]}... (should be ~0)")
        print(f"[Client {self.client_id}]   Training std: {X_train_scaled.std(axis=0)[:3]}... (should be ~1)")
        
        return X_train_scaled, X_test_scaled
    
    def analyze_class_imbalance(self, y: pd.Series) -> Dict:
        """
        Analyze class distribution and calculate imbalance metrics
        
        This analysis helps determine if class balancing techniques
        (like SMOTE or class weights) are needed in Phase 2.
        
        Args:
            y: Target Series
            
        Returns:
            Dictionary containing imbalance analysis
        """
        print(f"\n[Client {self.client_id}] " + "=" * 50)
        print(f"[Client {self.client_id}] CLASS IMBALANCE ANALYSIS")
        print(f"[Client {self.client_id}] " + "=" * 50)
        
        class_counts = y.value_counts().sort_index()
        total_samples = len(y)
        
        print(f"[Client {self.client_id}] Class Distribution:")
        for class_val, count in class_counts.items():
            percentage = (count / total_samples) * 100
            print(f"[Client {self.client_id}]   Class {class_val}: {count} samples ({percentage:.2f}%)")
        
        if len(class_counts) == 2:
            minority_class = class_counts.idxmin()
            majority_class = class_counts.idxmax()
            minority_count = class_counts.min()
            majority_count = class_counts.max()
            
            imbalance_ratio = majority_count / minority_count
            
            print(f"[Client {self.client_id}] Imbalance Ratio: {imbalance_ratio:.2f}:1")
            
            if imbalance_ratio > 2.0:
                print(f"[Client {self.client_id}] [WARNING] Significant class imbalance detected!")
                print(f"[Client {self.client_id}]   Recommendation: Use class weights or SMOTE in Phase 2")
            else:
                print(f"[Client {self.client_id}] [OK] Classes are relatively balanced")
        
        analysis = {
            'class_counts': class_counts.to_dict(),
            'total_samples': total_samples,
            'num_classes': len(class_counts),
            'imbalance_ratio': imbalance_ratio if len(class_counts) == 2 else None,
            'is_imbalanced': imbalance_ratio > 2.0 if len(class_counts) == 2 else False
        }
        
        return analysis
    
    def preprocess(
        self,
        file_path: str,
        test_size: float = 0.2,
        random_state: int = 42,
        normalize: bool = True
    ) -> Dict:
        """
        Complete preprocessing pipeline
        
        This is the main method that orchestrates all preprocessing steps.
        It returns preprocessed data ready for machine learning.
        
        Args:
            file_path: Path to the client's dataset CSV file
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            normalize: Whether to normalize features
            
        Returns:
            Dictionary containing:
                - X_train: Training features (numpy array)
                - X_test: Test features (numpy array)
                - y_train: Training labels (numpy array)
                - y_test: Test labels (numpy array)
                - feature_columns: List of feature names
                - num_features: Number of features
                - imbalance_analysis: Class imbalance analysis
        """
        print(f"\n[Client {self.client_id}] " + "=" * 60)
        print(f"[Client {self.client_id}] STARTING PREPROCESSING PIPELINE")
        print(f"[Client {self.client_id}] " + "=" * 60)
        
        # Step 1: Load data
        df = self.load_data(file_path)
        
        # Step 2: Identify target column
        self.identify_target_column(df)
        
        # Step 3: Handle missing values
        df = self.handle_missing_values(df)
        
        # Step 4: Encode categorical features
        df = self.encode_categorical_features(df)
        
        # Step 5: Separate features and target
        X, y = self.separate_features_and_target(df)
        
        # Step 6: Split into train and test sets
        print(f"[Client {self.client_id}] Splitting into train/test sets (test_size={test_size})...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        print(f"[Client {self.client_id}] [OK] Train: {len(X_train)} samples, Test: {len(X_test)} samples")
        
        # Step 7: Convert to numpy arrays
        X_train = X_train.values
        X_test = X_test.values
        y_train = y_train.values
        y_test = y_test.values
        
        # Step 8: Normalize features (if requested)
        if normalize:
            X_train, X_test = self.normalize_features(X_train, X_test)
        else:
            print(f"[Client {self.client_id}] Skipping feature normalization")
        
        # Step 9: Analyze class imbalance
        imbalance_analysis = self.analyze_class_imbalance(pd.Series(y_train))
        
        print(f"\n[Client {self.client_id}] " + "=" * 60)
        print(f"[Client {self.client_id}] PREPROCESSING COMPLETE!")
        print(f"[Client {self.client_id}] " + "=" * 60)
        print(f"[Client {self.client_id}] Final Data Shape:")
        print(f"[Client {self.client_id}]   X_train: {X_train.shape}")
        print(f"[Client {self.client_id}]   X_test: {X_test.shape}")
        print(f"[Client {self.client_id}]   y_train: {y_train.shape}")
        print(f"[Client {self.client_id}]   y_test: {y_test.shape}")
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_columns': self.feature_columns,
            'num_features': self.num_features,
            'imbalance_analysis': imbalance_analysis,
            'target_column': self.target_column
        }


if __name__ == "__main__":
    # Example usage for Phase 1
    print("=" * 60)
    print("DATA PREPROCESSING - PHASE 1 EXAMPLE")
    print("=" * 60)
    
    # Example: Preprocess client 1 data
    preprocessor = DataPreprocessor(client_id=1)
    
    client_data_path = "dataset/processed/client_1.csv"
    
    if os.path.exists(client_data_path):
        result = preprocessor.preprocess(
            file_path=client_data_path,
            test_size=0.2,
            random_state=42,
            normalize=True
        )
        
        print("\n" + "=" * 60)
        print("PREPROCESSING RESULTS SUMMARY")
        print("=" * 60)
        print(f"Number of features: {result['num_features']}")
        print(f"Training samples: {result['X_train'].shape[0]}")
        print(f"Test samples: {result['X_test'].shape[0]}")
        print(f"Feature columns: {result['feature_columns']}")
    else:
        print(f"Error: Client data not found at {client_data_path}")
        print("Please run dataset_splitter.py first to create client datasets")

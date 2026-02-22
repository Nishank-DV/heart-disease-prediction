"""
Unit Tests for Data Preprocessing
Tests data loading, cleaning, encoding, and normalization
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os
import tempfile

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from client.data_preprocessing import DataPreprocessor
from typing import Tuple, Dict


class TestDataPreprocessing:
    """
    Test suite for DataPreprocessor class
    """
    
    def create_sample_dataset(self, num_samples=100):
        """Create a sample dataset for testing"""
        np.random.seed(42)
        data = {
            'age': np.random.randint(29, 80, num_samples),
            'sex': np.random.randint(0, 2, num_samples),
            'cp': np.random.randint(0, 4, num_samples),
            'trestbps': np.random.randint(94, 200, num_samples),
            'chol': np.random.randint(126, 564, num_samples),
            'fbs': np.random.randint(0, 2, num_samples),
            'restecg': np.random.randint(0, 3, num_samples),
            'thalach': np.random.randint(71, 202, num_samples),
            'exang': np.random.randint(0, 2, num_samples),
            'oldpeak': np.random.uniform(0, 6.2, num_samples),
            'slope': np.random.randint(0, 3, num_samples),
            'ca': np.random.randint(0, 4, num_samples),
            'thal': np.random.randint(0, 4, num_samples),
            'target': np.random.randint(0, 2, num_samples)
        }
        return pd.DataFrame(data)
    
    def test_preprocessor_initialization(self):
        """Test preprocessor initialization"""
        preprocessor = DataPreprocessor(client_id=1)
        assert preprocessor.client_id == 1
        assert preprocessor.scaler is not None
    
    def test_load_data(self):
        """Test data loading from CSV file"""
        # Create temporary CSV file
        df = self.create_sample_dataset(50)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            preprocessor = DataPreprocessor(client_id=1)
            loaded_df = preprocessor.load_data(temp_path)
            assert loaded_df.shape == df.shape
            assert list(loaded_df.columns) == list(df.columns)
        finally:
            os.unlink(temp_path)
    
    def test_handle_missing_values_numerical(self):
        """Test handling missing values in numerical columns"""
        df = self.create_sample_dataset(50)
        # Introduce missing values
        df.loc[0:5, 'age'] = np.nan
        df.loc[10:12, 'chol'] = np.nan
        
        preprocessor = DataPreprocessor(client_id=1)
        cleaned_df = preprocessor.handle_missing_values(df)
        
        # Should have no missing values
        assert cleaned_df.isnull().sum().sum() == 0
    
    def test_handle_missing_values_categorical(self):
        """Test handling missing values in categorical columns"""
        df = self.create_sample_dataset(50)
        # Add a categorical column with missing values
        df['test_cat'] = ['A', 'B', 'C'] * 16 + [np.nan] * 2
        df.loc[48:49, 'test_cat'] = np.nan
        
        preprocessor = DataPreprocessor(client_id=1)
        cleaned_df = preprocessor.handle_missing_values(df)
        
        # Should have no missing values
        assert cleaned_df['test_cat'].isnull().sum() == 0
    
    def test_separate_features_and_target(self):
        """Test separation of features and target"""
        df = self.create_sample_dataset(50)
        preprocessor = DataPreprocessor(client_id=1)
        preprocessor.target_column = 'target'
        
        X, y = preprocessor.separate_features_and_target(df)
        
        # Check shapes
        assert X.shape[1] == df.shape[1] - 1  # All columns except target
        assert len(y) == len(df)
        assert 'target' not in X.columns
        assert preprocessor.num_features == X.shape[1]
    
    def test_normalize_features(self):
        """Test feature normalization"""
        X_train = np.random.randn(100, 13) * 10 + 50  # Mean ~50, std ~10
        X_test = np.random.randn(20, 13) * 10 + 50
        
        preprocessor = DataPreprocessor(client_id=1)
        X_train_scaled, X_test_scaled = preprocessor.normalize_features(X_train, X_test)
        
        # Check normalization (mean ~0, std ~1)
        assert np.abs(X_train_scaled.mean()) < 1e-10  # Very close to 0
        assert np.abs(X_train_scaled.std() - 1.0) < 1e-10  # Very close to 1
    
    def test_analyze_class_imbalance(self):
        """Test class imbalance analysis"""
        # Create imbalanced dataset
        y = pd.Series([0] * 80 + [1] * 20)  # 4:1 imbalance
        
        preprocessor = DataPreprocessor(client_id=1)
        analysis = preprocessor.analyze_class_imbalance(y)
        
        assert analysis['num_classes'] == 2
        assert analysis['imbalance_ratio'] == 4.0
        assert analysis['is_imbalanced'] == True
    
    def test_complete_preprocessing_pipeline(self):
        """Test complete preprocessing pipeline"""
        df = self.create_sample_dataset(100)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            preprocessor = DataPreprocessor(client_id=1)
            result = preprocessor.preprocess(
                file_path=temp_path,
                test_size=0.2,
                random_state=42,
                normalize=True
            )
            
            # Check all required keys exist
            assert 'X_train' in result
            assert 'X_test' in result
            assert 'y_train' in result
            assert 'y_test' in result
            assert 'num_features' in result
            assert 'imbalance_analysis' in result
            
            # Check shapes
            assert result['X_train'].shape[0] + result['X_test'].shape[0] == len(df)
            assert result['X_train'].shape[1] == result['num_features']
            assert result['X_test'].shape[1] == result['num_features']
            
        finally:
            os.unlink(temp_path)


class TestDataPreprocessingEdgeCases:
    """
    Test edge cases for data preprocessing
    """
    
    def test_empty_dataset(self):
        """Test preprocessing with empty dataset (edge case)"""
        df = pd.DataFrame()
        preprocessor = DataPreprocessor(client_id=1)
        
        with pytest.raises((ValueError, KeyError, IndexError)):
            preprocessor.separate_features_and_target(df)
    
    def test_single_row_dataset(self):
        """Test preprocessing with single row (edge case)"""
        df = pd.DataFrame({
            'age': [50], 'sex': [1], 'cp': [2], 'trestbps': [120],
            'chol': [200], 'fbs': [0], 'restecg': [1], 'thalach': [150],
            'exang': [0], 'oldpeak': [1.5], 'slope': [1], 'ca': [0],
            'thal': [2], 'target': [1]
        })
        
        preprocessor = DataPreprocessor(client_id=1)
        preprocessor.target_column = 'target'
        
        # Should handle single row (though train/test split may fail)
        X, y = preprocessor.separate_features_and_target(df)
        assert len(X) == 1
        assert len(y) == 1
    
    def test_all_same_values(self):
        """Test preprocessing with all same values (edge case)"""
        df = pd.DataFrame({
            'age': [50] * 100,
            'sex': [1] * 100,
            'cp': [2] * 100,
            'trestbps': [120] * 100,
            'chol': [200] * 100,
            'fbs': [0] * 100,
            'restecg': [1] * 100,
            'thalach': [150] * 100,
            'exang': [0] * 100,
            'oldpeak': [1.5] * 100,
            'slope': [1] * 100,
            'ca': [0] * 100,
            'thal': [2] * 100,
            'target': [1] * 100
        })
        
        preprocessor = DataPreprocessor(client_id=1)
        preprocessor.target_column = 'target'
        X, y = preprocessor.separate_features_and_target(df)
        
        # Normalization with zero variance should handle gracefully
        X_train = X.values[:80]
        X_test = X.values[80:]
        X_train_scaled, X_test_scaled = preprocessor.normalize_features(X_train, X_test)
        
        # Should not crash
        assert X_train_scaled.shape == X_train.shape
    
    def test_extreme_values(self):
        """Test preprocessing with extreme values (edge case)"""
        df = pd.DataFrame({
            'age': [200, -10, 1000],  # Extreme ages
            'sex': [1, 0, 1],
            'cp': [2, 1, 3],
            'trestbps': [500, 50, 300],  # Extreme BP
            'chol': [1000, 50, 800],  # Extreme cholesterol
            'fbs': [0, 1, 0],
            'restecg': [1, 0, 2],
            'thalach': [300, 40, 250],  # Extreme heart rate
            'exang': [0, 1, 0],
            'oldpeak': [10, -5, 8],  # Extreme ST depression
            'slope': [1, 0, 2],
            'ca': [0, 3, 2],
            'thal': [2, 0, 3],
            'target': [1, 0, 1]
        })
        
        preprocessor = DataPreprocessor(client_id=1)
        preprocessor.target_column = 'target'
        X, y = preprocessor.separate_features_and_target(df)
        
        # Should handle extreme values without error
        assert X.shape[0] == 3
        assert len(y) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


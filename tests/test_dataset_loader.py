"""
Unit Tests for Dataset Loader
Tests dataset loading, analysis, and validation
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os
import tempfile

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.dataset_loader import (
    load_dataset,
    analyze_dataset,
    identify_target_column,
    load_and_analyze
)


class TestDatasetLoader:
    """
    Test suite for dataset loading functions
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
    
    def test_load_dataset_success(self):
        """Test successful dataset loading"""
        df = self.create_sample_dataset(50)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            loaded_df = load_dataset(temp_path)
            assert loaded_df.shape == df.shape
            assert list(loaded_df.columns) == list(df.columns)
        finally:
            os.unlink(temp_path)
    
    def test_load_dataset_file_not_found(self):
        """Test dataset loading with non-existent file"""
        with pytest.raises(FileNotFoundError):
            load_dataset("non_existent_file.csv")
    
    def test_identify_target_column(self):
        """Test target column identification"""
        df = self.create_sample_dataset(50)
        
        # Test with 'target' column
        target = identify_target_column(df)
        assert target == 'target'
        
        # Test with 'HeartDisease' column
        df_renamed = df.rename(columns={'target': 'HeartDisease'})
        target = identify_target_column(df_renamed)
        assert target == 'HeartDisease'
    
    def test_analyze_dataset(self):
        """Test dataset analysis"""
        df = self.create_sample_dataset(100)
        analysis = analyze_dataset(df)
        
        # Check analysis contains required keys
        assert 'shape' in analysis
        assert 'columns' in analysis
        assert 'numerical_features' in analysis
        assert 'categorical_features' in analysis
        assert 'target_column' in analysis
        assert 'missing_values' in analysis
    
    def test_analyze_dataset_with_missing_values(self):
        """Test analysis with missing values"""
        df = self.create_sample_dataset(50)
        # Introduce missing values
        df.loc[0:5, 'age'] = np.nan
        df.loc[10:12, 'chol'] = np.nan
        
        analysis = analyze_dataset(df)
        assert analysis['total_missing'] > 0
    
    def test_load_and_analyze(self):
        """Test complete load and analyze function"""
        df = self.create_sample_dataset(100)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            loaded_df, analysis = load_and_analyze(temp_path)
            assert loaded_df.shape == df.shape
            assert 'shape' in analysis
        finally:
            os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


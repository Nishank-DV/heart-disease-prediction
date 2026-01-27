"""
Edge Case Tests
Tests realistic medical edge cases and boundary conditions
"""

import pytest
import pandas as pd
import numpy as np
import torch
import sys
import os
import tempfile

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from client.model import HeartDiseaseMLP
from client.data_preprocessing import DataPreprocessor
from utils.evaluation import ModelEvaluator


class TestMedicalEdgeCases:
    """
    Test realistic medical edge cases
    """
    
    def test_extreme_age_values(self):
        """Test with extreme age values (very young/very old patients)"""
        # Medical edge case: Age outside normal range
        data = {
            'age': [15, 25, 95, 100],  # Extreme ages
            'sex': [0, 1, 0, 1],
            'cp': [2, 1, 3, 0],
            'trestbps': [120, 130, 140, 150],
            'chol': [200, 220, 240, 260],
            'fbs': [0, 1, 0, 1],
            'restecg': [1, 0, 2, 1],
            'thalach': [150, 160, 170, 180],
            'exang': [0, 1, 0, 1],
            'oldpeak': [1.5, 2.0, 2.5, 3.0],
            'slope': [1, 0, 2, 1],
            'ca': [0, 1, 2, 3],
            'thal': [2, 1, 3, 0],
            'target': [0, 1, 1, 1]
        }
        df = pd.DataFrame(data)
        
        preprocessor = DataPreprocessor(client_id=1)
        preprocessor.target_column = 'target'
        X, y = preprocessor.separate_features_and_target(df)
        
        # Should handle extreme ages without error
        assert X.shape[0] == 4
        assert len(y) == 4
    
    def test_critical_cholesterol_levels(self):
        """Test with critical cholesterol levels (medical edge case)"""
        # Medical edge case: Very high cholesterol (>500 mg/dl)
        data = {
            'age': [50, 55, 60, 65],
            'sex': [1, 0, 1, 0],
            'cp': [2, 1, 3, 0],
            'trestbps': [140, 150, 160, 170],
            'chol': [600, 700, 800, 900],  # Critical levels
            'fbs': [1, 1, 1, 1],  # High blood sugar
            'restecg': [2, 2, 2, 2],  # Abnormal ECG
            'thalach': [100, 110, 120, 130],  # Low heart rate
            'exang': [1, 1, 1, 1],  # Exercise angina
            'oldpeak': [5.0, 6.0, 7.0, 8.0],  # High ST depression
            'slope': [2, 2, 2, 2],  # Downsloping
            'ca': [3, 3, 3, 3],  # Multiple vessels
            'thal': [3, 3, 3, 3],  # Reversible defect
            'target': [1, 1, 1, 1]  # All have disease
        }
        df = pd.DataFrame(data)
        
        preprocessor = DataPreprocessor(client_id=1)
        preprocessor.target_column = 'target'
        X, y = preprocessor.separate_features_and_target(df)
        
        # Should handle critical values
        assert X.shape[0] == 4
        assert all(y == 1)  # All should be positive cases
    
    def test_normal_healthy_patient(self):
        """Test with normal healthy patient (medical edge case)"""
        # Medical edge case: All normal values, no disease
        data = {
            'age': [35],
            'sex': [0],
            'cp': [0],  # Typical angina
            'trestbps': [120],  # Normal BP
            'chol': [180],  # Normal cholesterol
            'fbs': [0],  # Normal blood sugar
            'restecg': [0],  # Normal ECG
            'thalach': [180],  # Good heart rate
            'exang': [0],  # No angina
            'oldpeak': [0.0],  # No ST depression
            'slope': [0],  # Upsloping
            'ca': [0],  # No blocked vessels
            'thal': [0],  # Normal
            'target': [0]  # No disease
        }
        df = pd.DataFrame(data)
        
        preprocessor = DataPreprocessor(client_id=1)
        preprocessor.target_column = 'target'
        X, y = preprocessor.separate_features_and_target(df)
        
        model = HeartDiseaseMLP(input_size=X.shape[1])
        input_tensor = torch.FloatTensor(X.values)
        output = model(input_tensor)
        
        # Should predict low probability of disease
        assert output[0][0] < 0.5  # Low probability
    
    def test_high_risk_patient(self):
        """Test with high-risk patient profile (medical edge case)"""
        # Medical edge case: Multiple risk factors
        data = {
            'age': [75],  # Old age
            'sex': [1],  # Male
            'cp': [3],  # Asymptomatic (dangerous)
            'trestbps': [180],  # High BP
            'chol': [350],  # High cholesterol
            'fbs': [1],  # High blood sugar
            'restecg': [2],  # Abnormal ECG
            'thalach': [100],  # Low heart rate
            'exang': [1],  # Exercise angina
            'oldpeak': [5.5],  # High ST depression
            'slope': [2],  # Downsloping
            'ca': [3],  # Multiple vessels blocked
            'thal': [2],  # Reversible defect
            'target': [1]  # Has disease
        }
        df = pd.DataFrame(data)
        
        preprocessor = DataPreprocessor(client_id=1)
        preprocessor.target_column = 'target'
        X, y = preprocessor.separate_features_and_target(df)
        
        model = HeartDiseaseMLP(input_size=X.shape[1])
        input_tensor = torch.FloatTensor(X.values)
        output = model(input_tensor)
        
        # Should predict high probability (after training)
        # For untrained model, just check it doesn't crash
        assert 0 <= output[0][0] <= 1
    
    def test_missing_critical_features(self):
        """Test handling when critical features are missing"""
        # Medical edge case: Missing important features
        df = pd.DataFrame({
            'age': [50, 55, 60],
            'sex': [1, 0, 1],
            'cp': [2, 1, 3],
            'trestbps': [np.nan, 140, 150],  # Missing BP
            'chol': [200, np.nan, 220],  # Missing cholesterol
            'fbs': [0, 1, 0],
            'restecg': [1, 0, 2],
            'thalach': [150, 160, 170],
            'exang': [0, 1, 0],
            'oldpeak': [1.5, 2.0, np.nan],  # Missing ST depression
            'slope': [1, 0, 2],
            'ca': [0, 1, 2],
            'thal': [2, 1, 3],
            'target': [0, 1, 1]
        })
        
        preprocessor = DataPreprocessor(client_id=1)
        preprocessor.target_column = 'target'
        cleaned_df = preprocessor.handle_missing_values(df)
        
        # Should fill missing values
        assert cleaned_df.isnull().sum().sum() == 0
    
    def test_boundary_values(self):
        """Test boundary values for all features"""
        # Test minimum and maximum values for each feature
        data = {
            'age': [29, 79],  # Min and max age
            'sex': [0, 1],  # Both sexes
            'cp': [0, 3],  # Min and max chest pain types
            'trestbps': [94, 200],  # Min and max BP
            'chol': [126, 564],  # Min and max cholesterol
            'fbs': [0, 1],
            'restecg': [0, 2],  # Min and max ECG results
            'thalach': [71, 202],  # Min and max heart rate
            'exang': [0, 1],
            'oldpeak': [0.0, 6.2],  # Min and max ST depression
            'slope': [0, 2],  # Min and max slope
            'ca': [0, 3],  # Min and max vessels
            'thal': [0, 3],  # Min and max thalassemia
            'target': [0, 1]
        }
        df = pd.DataFrame(data)
        
        preprocessor = DataPreprocessor(client_id=1)
        preprocessor.target_column = 'target'
        X, y = preprocessor.separate_features_and_target(df)
        
        # Should handle boundary values
        assert X.shape[0] == 2
        assert len(y) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


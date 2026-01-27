"""
Feature Information and Metadata Module
Maintains comprehensive information about dataset features
Used for explainability and viva presentations
"""

from typing import Dict, List, Optional


class FeatureInfo:
    """
    Class to maintain feature metadata and information
    """
    
    def __init__(self):
        """Initialize feature information dictionary"""
        self.features = self._initialize_feature_info()
    
    def _initialize_feature_info(self) -> Dict:
        """
        Initialize comprehensive feature information
        
        Returns:
            Dictionary containing feature metadata
        """
        features = {
            'age': {
                'name': 'Age',
                'type': 'numerical',
                'description': 'Age of the patient in years',
                'medical_meaning': 'Patient age is a key risk factor for heart disease. Risk increases with age.',
                'range': (29, 77),
                'unit': 'years',
                'encoding': None
            },
            'sex': {
                'name': 'Sex',
                'type': 'categorical',
                'description': 'Sex of the patient',
                'medical_meaning': 'Biological sex can influence heart disease risk patterns.',
                'values': {0: 'Female', 1: 'Male'},
                'encoding': 'label'
            },
            'cp': {
                'name': 'Chest Pain Type',
                'type': 'categorical',
                'description': 'Type of chest pain experienced',
                'medical_meaning': 'Different types of chest pain can indicate various cardiac conditions.',
                'values': {
                    0: 'Typical Angina',
                    1: 'Atypical Angina',
                    2: 'Non-anginal Pain',
                    3: 'Asymptomatic'
                },
                'encoding': 'label'
            },
            'trestbps': {
                'name': 'Resting Blood Pressure',
                'type': 'numerical',
                'description': 'Resting blood pressure in mm Hg on admission',
                'medical_meaning': 'High resting blood pressure is a major risk factor for cardiovascular disease.',
                'range': (94, 200),
                'unit': 'mm Hg',
                'encoding': None
            },
            'chol': {
                'name': 'Serum Cholesterol',
                'type': 'numerical',
                'description': 'Serum cholesterol level',
                'medical_meaning': 'High cholesterol can lead to plaque buildup in arteries, increasing heart disease risk.',
                'range': (126, 564),
                'unit': 'mg/dl',
                'encoding': None
            },
            'fbs': {
                'name': 'Fasting Blood Sugar',
                'type': 'categorical',
                'description': 'Fasting blood sugar > 120 mg/dl',
                'medical_meaning': 'Elevated fasting blood sugar may indicate diabetes, a risk factor for heart disease.',
                'values': {0: 'False (≤120 mg/dl)', 1: 'True (>120 mg/dl)'},
                'encoding': 'label'
            },
            'restecg': {
                'name': 'Resting ECG Results',
                'type': 'categorical',
                'description': 'Resting electrocardiographic results',
                'medical_meaning': 'ECG abnormalities can indicate underlying heart conditions or damage.',
                'values': {
                    0: 'Normal',
                    1: 'ST-T Wave Abnormality',
                    2: 'Left Ventricular Hypertrophy'
                },
                'encoding': 'label'
            },
            'thalach': {
                'name': 'Maximum Heart Rate',
                'type': 'numerical',
                'description': 'Maximum heart rate achieved during exercise',
                'medical_meaning': 'Exercise capacity and heart rate response are indicators of cardiovascular fitness.',
                'range': (71, 202),
                'unit': 'bpm',
                'encoding': None
            },
            'exang': {
                'name': 'Exercise Induced Angina',
                'type': 'categorical',
                'description': 'Presence of exercise-induced chest pain',
                'medical_meaning': 'Angina during exercise suggests reduced blood flow to the heart muscle.',
                'values': {0: 'No', 1: 'Yes'},
                'encoding': 'label'
            },
            'oldpeak': {
                'name': 'ST Depression',
                'type': 'numerical',
                'description': 'ST depression induced by exercise relative to rest',
                'medical_meaning': 'ST segment depression during exercise can indicate myocardial ischemia.',
                'range': (0.0, 6.2),
                'unit': 'mm',
                'encoding': None
            },
            'slope': {
                'name': 'ST Slope',
                'type': 'categorical',
                'description': 'Slope of the peak exercise ST segment',
                'medical_meaning': 'ST segment slope during exercise provides information about coronary artery disease.',
                'values': {
                    0: 'Upsloping',
                    1: 'Flat',
                    2: 'Downsloping'
                },
                'encoding': 'label'
            },
            'ca': {
                'name': 'Number of Major Vessels',
                'type': 'categorical',
                'description': 'Number of major vessels colored by fluoroscopy',
                'medical_meaning': 'More blocked vessels indicate more severe coronary artery disease.',
                'values': {0: '0 vessels', 1: '1 vessel', 2: '2 vessels', 3: '3 vessels'},
                'encoding': 'label'
            },
            'thal': {
                'name': 'Thalassemia',
                'type': 'categorical',
                'description': 'Thalassemia type',
                'medical_meaning': 'Thalassemia is a blood disorder that can affect heart function.',
                'values': {
                    0: 'Normal',
                    1: 'Fixed Defect',
                    2: 'Reversible Defect',
                    3: 'Unknown'
                },
                'encoding': 'label'
            },
            'target': {
                'name': 'Heart Disease',
                'type': 'target',
                'description': 'Presence of heart disease',
                'medical_meaning': 'Binary classification target: 0 = no heart disease, 1 = heart disease present',
                'values': {0: 'No Disease', 1: 'Heart Disease'},
                'encoding': None
            }
        }
        
        return features
    
    def get_feature(self, feature_name: str) -> Optional[Dict]:
        """
        Get information about a specific feature
        
        Args:
            feature_name: Name of the feature
            
        Returns:
            Dictionary containing feature information, or None if not found
        """
        return self.features.get(feature_name)
    
    def get_numerical_features(self) -> List[str]:
        """
        Get list of numerical feature names
        
        Returns:
            List of numerical feature names
        """
        return [name for name, info in self.features.items() 
                if info['type'] == 'numerical' and name != 'target']
    
    def get_categorical_features(self) -> List[str]:
        """
        Get list of categorical feature names
        
        Returns:
            List of categorical feature names
        """
        return [name for name, info in self.features.items() 
                if info['type'] == 'categorical']
    
    def get_target_feature(self) -> str:
        """
        Get the target feature name
        
        Returns:
            Target feature name
        """
        return 'target'
    
    def print_feature_summary(self):
        """
        Print a summary of all features
        """
        print("\n" + "=" * 60)
        print("FEATURE INFORMATION SUMMARY")
        print("=" * 60)
        
        numerical = self.get_numerical_features()
        categorical = self.get_categorical_features()
        
        print(f"\nNumerical Features ({len(numerical)}):")
        for feat in numerical:
            info = self.features[feat]
            print(f"  • {feat}: {info['description']}")
        
        print(f"\nCategorical Features ({len(categorical)}):")
        for feat in categorical:
            info = self.features[feat]
            print(f"  • {feat}: {info['description']}")
        
        print(f"\nTarget Feature:")
        target_info = self.features['target']
        print(f"  • {target_info['name']}: {target_info['description']}")
    
    def get_encoding_strategy(self, feature_name: str) -> str:
        """
        Get the recommended encoding strategy for a feature
        
        Args:
            feature_name: Name of the feature
            
        Returns:
            Encoding strategy ('label', 'onehot', or None)
        """
        feature = self.get_feature(feature_name)
        if feature:
            return feature.get('encoding', None)
        return None


# Global instance for easy access
feature_info = FeatureInfo()


if __name__ == "__main__":
    # Example usage
    info = FeatureInfo()
    info.print_feature_summary()
    
    print("\n" + "=" * 60)
    print("EXAMPLE: Getting specific feature information")
    print("=" * 60)
    
    example_feature = info.get_feature('cp')
    if example_feature:
        print(f"\nFeature: {example_feature['name']}")
        print(f"Type: {example_feature['type']}")
        print(f"Description: {example_feature['description']}")
        print(f"Medical Meaning: {example_feature['medical_meaning']}")
        if 'values' in example_feature:
            print(f"Values: {example_feature['values']}")


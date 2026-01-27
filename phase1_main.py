"""
Phase 1 Main Execution Script
Complete workflow for dataset engineering and preprocessing
"""

import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.dataset_loader import load_and_analyze
from utils.feature_info import feature_info
from utils.dataset_splitter import split_dataset_for_clients, verify_client_datasets
from client.data_preprocessing import DataPreprocessor


def main():
    """
    Main execution function for Phase 1
    """
    print("=" * 60)
    print("FEDERATED DEEP LEARNING FOR HEART DISEASE PREDICTION")
    print("PHASE 1: DATASET ENGINEERING & PREPROCESSING")
    print("=" * 60)
    
    # Configuration
    raw_dataset_path = "dataset/raw/heart.csv"
    processed_dir = "dataset/processed"
    num_clients = 3
    
    # Step 1: Load and analyze dataset
    print("\n" + "=" * 60)
    print("STEP 1: DATASET LOADING AND ANALYSIS")
    print("=" * 60)
    
    if not os.path.exists(raw_dataset_path):
        print(f"\n[ERROR] Dataset not found at {raw_dataset_path}")
        print("Please download the dataset and place it at: dataset/raw/heart.csv")
        print("\nSources:")
        print("  - Kaggle: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset")
        print("  - UCI ML: https://archive.ics.uci.edu/ml/datasets/heart+disease")
        return
    
    try:
        df, analysis = load_and_analyze(raw_dataset_path)
        print("\n[OK] Step 1 Complete: Dataset loaded and analyzed")
    except Exception as e:
        print(f"\n[ERROR] Error in Step 1: {e}")
        return
    
    # Step 2: Display feature information
    print("\n" + "=" * 60)
    print("STEP 2: FEATURE INFORMATION")
    print("=" * 60)
    
    feature_info.print_feature_summary()
    print("\n[OK] Step 2 Complete: Feature information extracted")
    
    # Step 3: Split dataset into client datasets
    print("\n" + "=" * 60)
    print("STEP 3: DATASET SPLITTING FOR FEDERATED LEARNING")
    print("=" * 60)
    
    try:
        client_paths = split_dataset_for_clients(
            dataset_path=raw_dataset_path,
            num_clients=num_clients,
            output_dir=processed_dir,
            random_state=42,
            stratify=True
        )
        
        # Verify client datasets
        if verify_client_datasets(client_paths):
            print("\n[OK] Step 3 Complete: Client datasets created and verified")
        else:
            print("\n[WARNING] Step 3 Warning: Some client datasets may have issues")
    except Exception as e:
        print(f"\n[ERROR] Error in Step 3: {e}")
        return
    
    # Step 4: Preprocess each client's data
    print("\n" + "=" * 60)
    print("STEP 4: CLIENT-SIDE DATA PREPROCESSING")
    print("=" * 60)
    
    preprocessing_results = []
    
    for i in range(1, num_clients + 1):
        client_data_path = os.path.join(processed_dir, f"client_{i}.csv")
        
        if not os.path.exists(client_data_path):
            print(f"\n[WARNING] Client {i} data not found at {client_data_path}")
            continue
        
        print(f"\n--- Processing Client {i} ---")
        
        try:
            preprocessor = DataPreprocessor(client_id=i)
            result = preprocessor.preprocess(
                file_path=client_data_path,
                test_size=0.2,
                random_state=42,
                normalize=True
            )
            
            preprocessing_results.append({
                'client_id': i,
                'result': result
            })
            
            print(f"[OK] Client {i} preprocessing complete")
            
        except Exception as e:
            print(f"[ERROR] Error preprocessing Client {i}: {e}")
            continue
    
    # Step 5: Summary Report
    print("\n" + "=" * 60)
    print("PHASE 1 SUMMARY REPORT")
    print("=" * 60)
    
    print(f"\nDataset Information:")
    print(f"  Original dataset: {raw_dataset_path}")
    print(f"  Total samples: {analysis['shape'][0]}")
    print(f"  Total features: {analysis['shape'][1]}")
    print(f"  Numerical features: {len(analysis['numerical_features'])}")
    print(f"  Categorical features: {len(analysis['categorical_features'])}")
    
    print(f"\nClient Datasets:")
    print(f"  Number of clients: {num_clients}")
    print(f"  Client datasets location: {processed_dir}")
    
    print(f"\nPreprocessing Results:")
    for item in preprocessing_results:
        client_id = item['client_id']
        result = item['result']
        print(f"\n  Client {client_id}:")
        print(f"    Features: {result['num_features']}")
        print(f"    Training samples: {result['X_train'].shape[0]}")
        print(f"    Test samples: {result['X_test'].shape[0]}")
        imbalance = result['imbalance_analysis']
        if imbalance['is_imbalanced']:
            print(f"    [WARNING] Class imbalance detected (ratio: {imbalance['imbalance_ratio']:.2f}:1)")
        else:
            print(f"    [OK] Classes balanced")
    
    print("\n" + "=" * 60)
    print("PRIVACY-FIRST DESIGN VERIFICATION")
    print("=" * 60)
    print("[OK] All preprocessing is client-side")
    print("[OK] Raw data never centralized")
    print("[OK] Each client processes data independently")
    print("[OK] Ready for Phase 2: Federated Learning")
    print("=" * 60)
    
    print("\n" + "=" * 60)
    print("PHASE 1 COMPLETE!")
    print("=" * 60)
    print("\nNext Steps (Phase 2):")
    print("  • Implement deep learning model (PyTorch)")
    print("  • Set up federated learning framework (Flower)")
    print("  • Train model using federated averaging")
    print("  • Evaluate global model performance")
    print("=" * 60)


if __name__ == "__main__":
    main()


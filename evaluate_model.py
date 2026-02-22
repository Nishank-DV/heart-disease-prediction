"""
Script to evaluate a trained federated learning model
This script loads a model and evaluates it on the test dataset
"""

import os
import sys
import torch
import argparse

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from client.model import HeartDiseaseMLP
from client.data_preprocessing import DataPreprocessor
from utils.evaluation import evaluate_model_comprehensive


def main():
    parser = argparse.ArgumentParser(description="Evaluate Federated Learning Model")
    parser.add_argument("--dataset", type=str, default="dataset/heart.csv",
                       help="Path to dataset CSV file")
    parser.add_argument("--model_weights", type=str, default=None,
                       help="Path to saved model weights (.pth file)")
    parser.add_argument("--num_features", type=int, default=None,
                       help="Number of input features (auto-detected if not provided)")
    parser.add_argument("--save_plots", action="store_true",
                       help="Save evaluation plots to files")
    
    args = parser.parse_args()
    
    # Check dataset
    if not os.path.exists(args.dataset):
        print(f"Error: Dataset not found at {args.dataset}")
        return
    
    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    
    # Preprocess data
    print("\n[Step 1] Preprocessing data...")
    preprocessor = DataPreprocessor(client_id=0)
    preprocessed_data = preprocessor.preprocess(
        file_path=args.dataset,
        test_size=0.2,
        random_state=42,
        normalize=True
    )
    num_features = preprocessed_data["num_features"]
    X_test = torch.FloatTensor(preprocessed_data["X_test"])
    y_test = torch.FloatTensor(preprocessed_data["y_test"]).unsqueeze(1)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False
    )
    
    print(f"Number of features: {num_features}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    print("\n[Step 2] Creating model...")
    model = HeartDiseaseMLP(input_size=num_features)
    
    # Load weights if provided
    if args.model_weights and os.path.exists(args.model_weights):
        print(f"[Step 3] Loading model weights from {args.model_weights}...")
        try:
            model.load_state_dict(torch.load(args.model_weights, map_location='cpu'))
            print("Model weights loaded successfully!")
        except Exception as e:
            print(f"Warning: Could not load weights: {e}")
            print("Using randomly initialized model.")
    else:
        print("[Step 3] No model weights provided. Using randomly initialized model.")
        print("Note: For meaningful results, provide trained model weights.")
    
    # Evaluate model
    print("\n[Step 4] Evaluating model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    metrics = evaluate_model_comprehensive(
        model=model,
        test_loader=test_loader,
        device=device,
        plot_results=True,
        save_plots=args.save_plots,
        save_prefix="results/evaluation"
    )
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE!")
    print("=" * 60)
    print(f"\nFinal Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']:.4f}")
    print(f"  AUC:       {metrics['auc']:.4f}")
    print(f"  Loss:      {metrics['loss']:.4f}")
    
    if args.save_plots:
        print(f"\nPlots saved to results/ directory")


if __name__ == "__main__":
    main()


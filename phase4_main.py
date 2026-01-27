"""
Phase 4 Main Execution Script
Comprehensive evaluation, visualization, and result analysis

This script:
- Evaluates local models from Phase 2
- Evaluates federated model from Phase 3
- Compares local vs federated performance
- Generates visualizations
- Provides healthcare-specific analysis
"""

import os
import sys
import torch
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from client.model import HeartDiseaseMLP
from client.data_preprocessing import DataPreprocessor
from utils.evaluation import ModelEvaluator, evaluate_model_comprehensive
import torch.utils.data


def load_and_prepare_test_data(client_id: int, client_data_path: str):
    """
    Load and prepare test data for evaluation
    
    Args:
        client_id: Client identifier
        client_data_path: Path to client's dataset
        
    Returns:
        Tuple of (test_loader, num_features)
    """
    # Preprocess data
    preprocessor = DataPreprocessor(client_id=client_id)
    preprocessed_data = preprocessor.preprocess(
        file_path=client_data_path,
        test_size=0.2,
        random_state=42,
        normalize=True
    )
    
    # Convert to PyTorch tensors
    X_test = torch.FloatTensor(preprocessed_data['X_test'])
    y_test = torch.FloatTensor(preprocessed_data['y_test']).unsqueeze(1)
    
    # Create data loader
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False
    )
    
    return test_loader, preprocessed_data['num_features']


def evaluate_local_model(
    client_id: int,
    model_path: str,
    client_data_path: str,
    save_dir: str = "results"
):
    """
    Evaluate a locally trained model
    
    Args:
        client_id: Client identifier
        model_path: Path to saved model weights
        client_data_path: Path to client's dataset
        save_dir: Directory to save results
        
    Returns:
        Dictionary containing evaluation metrics
    """
    print(f"\n{'='*60}")
    print(f"EVALUATING LOCAL MODEL - CLIENT {client_id}")
    print(f"{'='*60}")
    
    # Load test data
    test_loader, num_features = load_and_prepare_test_data(client_id, client_data_path)
    
    # Load model
    model = HeartDiseaseMLP(input_size=num_features)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        print(f"✓ Model loaded from {model_path}")
    else:
        print(f"⚠ Model not found at {model_path}, using random initialization")
    
    # Evaluate
    evaluator = ModelEvaluator(model_name=f"Local Model - Client {client_id}")
    metrics = evaluator.evaluate_model(model, test_loader, device="cpu")
    
    # Print report
    evaluator.print_evaluation_report(metrics, f"Local Model - Client {client_id}")
    
    # Visualize
    client_save_dir = os.path.join(save_dir, f"client_{client_id}_local")
    evaluator.plot_confusion_matrix(
        metrics['confusion_matrix'],
        title=f"Confusion Matrix - Local Model (Client {client_id})",
        save_path=os.path.join(client_save_dir, "confusion_matrix.png")
    )
    
    metrics_to_plot = {
        'Accuracy': metrics['accuracy'],
        'Precision': metrics['precision'],
        'Recall': metrics['recall'],
        'F1-Score': metrics['f1_score']
    }
    
    evaluator.plot_metrics_comparison(
        metrics_to_plot,
        title=f"Performance Metrics - Local Model (Client {client_id})",
        save_path=os.path.join(client_save_dir, "metrics.png")
    )
    
    return metrics


def evaluate_federated_model(
    federated_model_path: str,
    test_data_paths: list,
    save_dir: str = "results"
):
    """
    Evaluate federated model on all client test sets
    
    Args:
        federated_model_path: Path to federated model weights
        test_data_paths: List of client data paths
        save_dir: Directory to save results
        
    Returns:
        Dictionary containing aggregated evaluation metrics
    """
    print(f"\n{'='*60}")
    print("EVALUATING FEDERATED MODEL")
    print(f"{'='*60}")
    
    all_metrics = []
    
    for i, client_data_path in enumerate(test_data_paths, 1):
        print(f"\n--- Evaluating on Client {i} Test Data ---")
        
        # Load test data
        test_loader, num_features = load_and_prepare_test_data(i, client_data_path)
        
        # Load federated model
        model = HeartDiseaseMLP(input_size=num_features)
        if os.path.exists(federated_model_path):
            model.load_state_dict(torch.load(federated_model_path, map_location='cpu'))
        else:
            print(f"⚠ Federated model not found, using random initialization")
            print("Note: In real federated learning, model would be saved after training")
        
        # Evaluate
        evaluator = ModelEvaluator(model_name=f"Federated Model")
        metrics = evaluator.evaluate_model(model, test_loader, device="cpu")
        all_metrics.append(metrics)
        
        print(f"Client {i} - Accuracy: {metrics['accuracy']:.4f}, Recall: {metrics['recall']:.4f}")
    
    # Aggregate metrics (weighted average by number of samples)
    total_samples = sum(m['num_samples'] for m in all_metrics)
    
    aggregated_metrics = {
        'accuracy': sum(m['accuracy'] * m['num_samples'] for m in all_metrics) / total_samples,
        'precision': sum(m['precision'] * m['num_samples'] for m in all_metrics) / total_samples,
        'recall': sum(m['recall'] * m['num_samples'] for m in all_metrics) / total_samples,
        'f1_score': sum(m['f1_score'] * m['num_samples'] for m in all_metrics) / total_samples,
        'num_samples': total_samples,
        'confusion_matrix': sum(m['confusion_matrix'] for m in all_metrics),
        'auc': np.mean([m['auc'] for m in all_metrics if m['auc'] is not None])
    }
    
    # Print aggregated report
    evaluator = ModelEvaluator(model_name="Federated Model")
    evaluator.print_evaluation_report(aggregated_metrics, "Federated Model (Aggregated)")
    
    # Visualize aggregated results
    federated_save_dir = os.path.join(save_dir, "federated")
    evaluator.plot_confusion_matrix(
        aggregated_metrics['confusion_matrix'],
        title="Confusion Matrix - Federated Model (All Clients)",
        save_path=os.path.join(federated_save_dir, "confusion_matrix.png")
    )
    
    metrics_to_plot = {
        'Accuracy': aggregated_metrics['accuracy'],
        'Precision': aggregated_metrics['precision'],
        'Recall': aggregated_metrics['recall'],
        'F1-Score': aggregated_metrics['f1_score']
    }
    
    evaluator.plot_metrics_comparison(
        metrics_to_plot,
        title="Performance Metrics - Federated Model",
        save_path=os.path.join(federated_save_dir, "metrics.png")
    )
    
    return aggregated_metrics


def main():
    """
    Main execution function for Phase 4
    """
    print("=" * 60)
    print("FEDERATED DEEP LEARNING FOR HEART DISEASE PREDICTION")
    print("PHASE 4: EVALUATION, VISUALIZATION & RESULT ANALYSIS")
    print("=" * 60)
    
    # Configuration
    num_clients = 3
    processed_dir = "dataset/processed"
    models_dir = "models"
    results_dir = "results"
    
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    
    # Check if required files exist
    client_data_paths = []
    local_model_paths = []
    
    for i in range(1, num_clients + 1):
        client_path = os.path.join(processed_dir, f"client_{i}.csv")
        model_path = os.path.join(models_dir, f"client_{i}_model.pth")
        
        if os.path.exists(client_path):
            client_data_paths.append(client_path)
        else:
            print(f"⚠ Warning: Client {i} data not found at {client_path}")
        
        if os.path.exists(model_path):
            local_model_paths.append((i, model_path))
        else:
            print(f"⚠ Warning: Client {i} model not found at {model_path}")
    
    if not client_data_paths:
        print("\n❌ Error: No client datasets found")
        print("Please run Phase 1 first to create client datasets")
        return
    
    # Step 1: Evaluate local models
    print("\n" + "=" * 60)
    print("STEP 1: EVALUATING LOCAL MODELS")
    print("=" * 60)
    
    local_metrics_list = []
    
    for client_id, model_path in local_model_paths:
        client_data_path = os.path.join(processed_dir, f"client_{client_id}.csv")
        metrics = evaluate_local_model(
            client_id=client_id,
            model_path=model_path,
            client_data_path=client_data_path,
            save_dir=results_dir
        )
        local_metrics_list.append(metrics)
    
    # Calculate average local metrics
    if local_metrics_list:
        total_samples = sum(m['num_samples'] for m in local_metrics_list)
        avg_local_metrics = {
            'accuracy': sum(m['accuracy'] * m['num_samples'] for m in local_metrics_list) / total_samples,
            'precision': sum(m['precision'] * m['num_samples'] for m in local_metrics_list) / total_samples,
            'recall': sum(m['recall'] * m['num_samples'] for m in local_metrics_list) / total_samples,
            'f1_score': sum(m['f1_score'] * m['num_samples'] for m in local_metrics_list) / total_samples,
            'num_samples': total_samples
        }
        
        print(f"\n{'='*60}")
        print("AVERAGE LOCAL MODEL PERFORMANCE")
        print(f"{'='*60}")
        evaluator = ModelEvaluator()
        evaluator.print_evaluation_report(avg_local_metrics, "Average Local Model")
    
    # Step 2: Evaluate federated model (if available)
    print("\n" + "=" * 60)
    print("STEP 2: EVALUATING FEDERATED MODEL")
    print("=" * 60)
    
    federated_model_path = os.path.join(models_dir, "federated_model.pth")
    
    if os.path.exists(federated_model_path):
        federated_metrics = evaluate_federated_model(
            federated_model_path=federated_model_path,
            test_data_paths=client_data_paths,
            save_dir=results_dir
        )
        
        # Step 3: Compare local vs federated
        if local_metrics_list:
            print("\n" + "=" * 60)
            print("STEP 3: COMPARING LOCAL vs FEDERATED MODELS")
            print("=" * 60)
            
            evaluator = ModelEvaluator()
            evaluator.compare_models(
                local_metrics=avg_local_metrics,
                federated_metrics=federated_metrics,
                save_dir=results_dir
            )
    else:
        print(f"⚠ Federated model not found at {federated_model_path}")
        print("Note: Federated model would be saved after Phase 3 training")
        print("For now, evaluating with local models only")
    
    # Summary
    print("\n" + "=" * 60)
    print("PHASE 4 COMPLETE!")
    print("=" * 60)
    print(f"\nResults saved to: {results_dir}/")
    print("\nGenerated visualizations:")
    print("  • Confusion matrices")
    print("  • Performance metrics comparison")
    print("  • Local vs Federated model comparison")
    print("\nAll plots are saved with high resolution (300 DPI)")
    print("Suitable for academic presentations and reports")
    print("=" * 60)


if __name__ == "__main__":
    main()


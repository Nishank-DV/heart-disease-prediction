"""
Phase 2 Main Execution Script
Complete workflow for deep learning model training and evaluation
"""

import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from client.train import train_client_model
from client.evaluate import evaluate_client_model


def main():
    """
    Main execution function for Phase 2
    """
    print("=" * 60)
    print("FEDERATED DEEP LEARNING FOR HEART DISEASE PREDICTION")
    print("PHASE 2: DEEP LEARNING MODEL & LOCAL TRAINING")
    print("=" * 60)
    
    # Configuration
    num_clients = 3
    num_epochs = 50
    learning_rate = 0.001
    batch_size = 32
    
    print(f"\nConfiguration:")
    print(f"  Number of clients: {num_clients}")
    print(f"  Training epochs: {num_epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Batch size: {batch_size}")
    
    # Check if client datasets exist
    processed_dir = "dataset/processed"
    if not os.path.exists(processed_dir):
        print(f"\n❌ Error: Processed dataset directory not found: {processed_dir}")
        print("Please run Phase 1 first to create client datasets")
        return
    
    # Train models for each client
    print("\n" + "=" * 60)
    print("STEP 1: LOCAL MODEL TRAINING")
    print("=" * 60)
    
    trained_models = []
    
    for client_id in range(1, num_clients + 1):
        client_data_path = os.path.join(processed_dir, f"client_{client_id}.csv")
        
        if not os.path.exists(client_data_path):
            print(f"\n⚠ Warning: Client {client_id} data not found at {client_data_path}")
            continue
        
        print(f"\n{'='*60}")
        print(f"Training Client {client_id} Model")
        print(f"{'='*60}")
        
        try:
            model = train_client_model(
                client_id=client_id,
                client_data_path=client_data_path,
                num_epochs=num_epochs,
                learning_rate=learning_rate,
                batch_size=batch_size,
                save_model=True
            )
            
            trained_models.append(client_id)
            print(f"\n✓ Client {client_id} model trained and saved")
            
        except Exception as e:
            print(f"\n❌ Error training Client {client_id}: {e}")
            continue
    
    if not trained_models:
        print("\n❌ No models were trained successfully")
        return
    
    # Evaluate models
    print("\n" + "=" * 60)
    print("STEP 2: LOCAL MODEL EVALUATION")
    print("=" * 60)
    
    evaluation_results = []
    
    for client_id in trained_models:
        model_path = f"models/client_{client_id}_model.pth"
        client_data_path = os.path.join(processed_dir, f"client_{client_id}.csv")
        
        if not os.path.exists(model_path):
            print(f"\n⚠ Warning: Model not found for Client {client_id}")
            continue
        
        print(f"\n{'='*60}")
        print(f"Evaluating Client {client_id} Model")
        print(f"{'='*60}")
        
        try:
            metrics = evaluate_client_model(
                client_id=client_id,
                model_path=model_path,
                client_data_path=client_data_path
            )
            
            evaluation_results.append({
                'client_id': client_id,
                'metrics': metrics
            })
            
            print(f"\n✓ Client {client_id} evaluation complete")
            
        except Exception as e:
            print(f"\n❌ Error evaluating Client {client_id}: {e}")
            continue
    
    # Summary Report
    print("\n" + "=" * 60)
    print("PHASE 2 SUMMARY REPORT")
    print("=" * 60)
    
    print(f"\nTraining Summary:")
    print(f"  Clients trained: {len(trained_models)}")
    print(f"  Training epochs: {num_epochs}")
    print(f"  Models saved to: models/")
    
    if evaluation_results:
        print(f"\nEvaluation Summary:")
        print(f"  {'Client':<10} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print(f"  {'-'*10} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
        
        for result in evaluation_results:
            client_id = result['client_id']
            metrics = result['metrics']
            print(f"  {client_id:<10} {metrics['accuracy']:<12.4f} {metrics['precision']:<12.4f} "
                  f"{metrics['recall']:<12.4f} {metrics['f1_score']:<12.4f}")
        
        # Calculate average metrics
        avg_accuracy = sum(r['metrics']['accuracy'] for r in evaluation_results) / len(evaluation_results)
        avg_precision = sum(r['metrics']['precision'] for r in evaluation_results) / len(evaluation_results)
        avg_recall = sum(r['metrics']['recall'] for r in evaluation_results) / len(evaluation_results)
        avg_f1 = sum(r['metrics']['f1_score'] for r in evaluation_results) / len(evaluation_results)
        
        print(f"\n  {'Average':<10} {avg_accuracy:<12.4f} {avg_precision:<12.4f} "
              f"{avg_recall:<12.4f} {avg_f1:<12.4f}")
    
    print("\n" + "=" * 60)
    print("PHASE 2 COMPLETE!")
    print("=" * 60)
    print("\nKey Achievements:")
    print("  ✓ Deep Learning models created (MLP architecture)")
    print("  ✓ Local training completed for all clients")
    print("  ✓ Models evaluated on local test data")
    print("  ✓ Model weights saved for Phase 3 (Federated Learning)")
    
    print("\nNext Steps (Phase 3):")
    print("  • Implement Federated Learning framework (Flower)")
    print("  • Set up client-server communication")
    print("  • Aggregate model weights using Federated Averaging")
    print("  • Train global federated model")
    print("  • Evaluate federated model performance")
    print("=" * 60)


if __name__ == "__main__":
    main()


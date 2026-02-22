"""
Comprehensive Evaluation and Visualization Module - Phase 4
Provides evaluation metrics, visualizations, and result analysis

This module focuses on:
- Performance evaluation metrics
- Visualization of results
- Comparison between local and federated models
- Healthcare-specific metric interpretation
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    classification_report
)
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import os


# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray] = None
) -> Dict:
    """
    Backward-compatible utility wrapper for metric calculation.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (optional)

    Returns:
        Dictionary of computed metrics
    """
    evaluator = ModelEvaluator(model_name="Evaluation")
    return evaluator.calculate_metrics(y_true, y_pred, y_pred_proba)


class ModelEvaluator:
    """
    Comprehensive model evaluator for heart disease prediction
    
    This class provides:
    - Metric calculation (Accuracy, Precision, Recall, F1-Score)
    - Confusion matrix generation
    - ROC curve analysis
    - Visualization of results
    - Comparison between models
    """
    
    def __init__(self, model_name: str = "Model"):
        """
        Initialize the evaluator
        
        Args:
            model_name: Name of the model being evaluated
        """
        self.model_name = model_name
        self.metrics_history = []  # For tracking metrics over rounds
    
    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Calculate comprehensive classification metrics
        
        WHY THESE METRICS MATTER IN HEALTHCARE:
        =======================================
        1. Accuracy: Overall correctness - important but can be misleading with imbalanced data
        2. Precision: Of all predicted diseases, how many are actually diseases?
           - Low precision = many false alarms (unnecessary stress, tests)
        3. Recall: Of all actual diseases, how many did we catch?
           - Low recall = missed diseases (LIFE-THREATENING!)
        4. F1-Score: Balance between precision and recall
           - Important when both false positives and false negatives matter
        
        For heart disease prediction:
        - RECALL is CRITICAL: Missing a heart disease case can be fatal
        - Precision is also important: False alarms cause unnecessary anxiety
        - F1-Score balances both concerns
        
        Args:
            y_true: True labels (binary: 0 or 1)
            y_pred: Predicted labels (binary: 0 or 1)
            y_pred_proba: Predicted probabilities (optional, for ROC curve)
            
        Returns:
            Dictionary containing all metrics
        """
        # Ensure arrays are flattened
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        
        # Calculate basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate AUC-ROC if probabilities provided
        auc_score = None
        if y_pred_proba is not None:
            y_pred_proba = np.array(y_pred_proba).flatten()
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            auc_score = auc(fpr, tpr)
        
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": cm,
            "auc": auc_score,
            "num_samples": len(y_true)
        }
        
        return metrics
    
    def evaluate_model(
        self,
        model: nn.Module,
        test_loader: torch.utils.data.DataLoader,
        device: str = "cpu"
    ) -> Dict:
        """
        Evaluate a PyTorch model on test data
        
        Args:
            model: PyTorch model to evaluate
            test_loader: DataLoader for test data
            device: Device to use ('cpu' or 'cuda')
            
        Returns:
            Dictionary containing evaluation metrics
        """
        device = torch.device(device if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        
        all_predictions = []
        all_probabilities = []
        all_labels = []
        
        criterion = nn.BCELoss()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for features, labels in test_loader:
                features = features.to(device)
                labels = labels.to(device)
                
                # Forward pass
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                num_batches += 1
                
                # Store predictions and probabilities
                probabilities = outputs.cpu().numpy()
                predictions = (outputs > 0.5).float().cpu().numpy()
                
                all_probabilities.extend(probabilities)
                all_predictions.extend(predictions)
                all_labels.extend(labels.cpu().numpy())
        
        # Convert to numpy arrays
        y_true = np.array(all_labels).flatten()
        y_pred = np.array(all_predictions).flatten()
        y_pred_proba = np.array(all_probabilities).flatten()
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_true, y_pred, y_pred_proba)
        metrics["loss"] = total_loss / num_batches if num_batches > 0 else 0.0
        
        return metrics
    
    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        title: str = "Confusion Matrix",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (8, 6)
    ):
        """
        Plot confusion matrix as a heatmap
        
        WHY CONFUSION MATRIX MATTERS:
        ==============================
        In healthcare, we need to understand:
        - True Negatives (TN): Correctly identified healthy patients
        - False Positives (FP): Healthy patients flagged as diseased (unnecessary stress)
        - False Negatives (FN): Diseased patients missed (CRITICAL - can be fatal!)
        - True Positives (TP): Correctly identified diseased patients
        
        For heart disease:
        - False Negatives are the most dangerous (missed disease)
        - False Positives cause unnecessary anxiety and tests
        
        Args:
            cm: Confusion matrix (2x2 array)
            title: Plot title
            save_path: Optional path to save the figure
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        
        # Create heatmap
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["No Disease", "Heart Disease"],
            yticklabels=["No Disease", "Heart Disease"],
            cbar_kws={"label": "Count"},
            linewidths=0.5,
            linecolor='gray'
        )
        
        plt.title(title, fontsize=14, fontweight="bold", pad=20)
        plt.ylabel("True Label", fontsize=12, fontweight="bold")
        plt.xlabel("Predicted Label", fontsize=12, fontweight="bold")
        
        # Add interpretation text
        tn, fp, fn, tp = cm.ravel()
        plt.text(0.5, -0.15, 
                f"TN={tn} | FP={fp} | FN={fn} | TP={tp}",
                ha='center', transform=plt.gca().transAxes,
                fontsize=10, style='italic')
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        title: str = "ROC Curve",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (8, 6)
    ):
        """
        Plot ROC (Receiver Operating Characteristic) curve
        
        WHY ROC CURVE MATTERS:
        ======================
        ROC curve shows the trade-off between:
        - True Positive Rate (Recall/Sensitivity): Ability to detect diseases
        - False Positive Rate: Rate of false alarms
        
        For heart disease:
        - Higher AUC = better at distinguishing disease from no disease
        - AUC > 0.8 is considered good
        - AUC = 1.0 is perfect (never achieved in practice)
        - AUC = 0.5 is random guessing
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            title: Plot title
            save_path: Optional path to save the figure
            figsize: Figure size
        """
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=figsize)
        
        # Plot ROC curve
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        
        # Plot diagonal (random classifier)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random Classifier (AUC = 0.5)')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12, fontweight="bold")
        plt.ylabel('True Positive Rate (Sensitivity/Recall)', fontsize=12, fontweight="bold")
        plt.title(title, fontsize=14, fontweight="bold", pad=20)
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(alpha=0.3)
        
        # Add interpretation
        if roc_auc >= 0.9:
            interpretation = "Excellent"
        elif roc_auc >= 0.8:
            interpretation = "Good"
        elif roc_auc >= 0.7:
            interpretation = "Fair"
        else:
            interpretation = "Poor"
        
        plt.text(0.6, 0.2, f'Model Performance: {interpretation}',
                transform=plt.gca().transAxes, fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"ROC curve saved to {save_path}")
        
        plt.show()
    
    def plot_metrics_comparison(
        self,
        metrics_dict: Dict,
        title: str = "Model Performance Metrics",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 6)
    ):
        """
        Plot bar chart comparing different metrics
        
        Args:
            metrics_dict: Dictionary with metric names as keys and values
            title: Plot title
            save_path: Optional path to save the figure
            figsize: Figure size
        """
        metrics_names = list(metrics_dict.keys())
        metrics_values = list(metrics_dict.values())
        
        plt.figure(figsize=figsize)
        
        # Create color map
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
        bars = plt.bar(metrics_names, metrics_values, color=colors[:len(metrics_names)])
        
        # Add value labels on bars
        for bar, value in zip(bars, metrics_values):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f'{value:.3f}',
                ha='center',
                va='bottom',
                fontsize=11,
                fontweight='bold'
            )
        
        plt.ylim([0, 1.1])
        plt.ylabel('Score', fontsize=12, fontweight="bold")
        plt.title(title, fontsize=14, fontweight="bold", pad=20)
        plt.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Metrics comparison saved to {save_path}")
        
        plt.show()
    
    def plot_training_history(
        self,
        history: List[Dict],
        title: str = "Training History",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 5)
    ):
        """
        Plot training history (loss and accuracy over rounds)
        
        This is useful for federated learning to see how the model
        improves over multiple rounds.
        
        Args:
            history: List of dictionaries with 'round', 'loss', 'accuracy' keys
            title: Plot title
            save_path: Optional path to save the figure
            figsize: Figure size
        """
        if not history:
            print("No history data to plot")
            return
        
        rounds = [h['round'] for h in history]
        losses = [h['loss'] for h in history]
        accuracies = [h['accuracy'] for h in history]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot loss
        ax1.plot(rounds, losses, 'o-', color='#e74c3c', linewidth=2, markersize=6)
        ax1.set_xlabel('Federated Round', fontsize=11, fontweight="bold")
        ax1.set_ylabel('Loss', fontsize=11, fontweight="bold")
        ax1.set_title('Loss vs Rounds', fontsize=12, fontweight="bold")
        ax1.grid(alpha=0.3)
        
        # Plot accuracy
        ax2.plot(rounds, accuracies, 'o-', color='#2ecc71', linewidth=2, markersize=6)
        ax2.set_xlabel('Federated Round', fontsize=11, fontweight="bold")
        ax2.set_ylabel('Accuracy', fontsize=11, fontweight="bold")
        ax2.set_title('Accuracy vs Rounds', fontsize=12, fontweight="bold")
        ax2.set_ylim([0, 1.1])
        ax2.grid(alpha=0.3)
        
        plt.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Training history saved to {save_path}")
        
        plt.show()
    
    def compare_models(
        self,
        local_metrics: Dict,
        federated_metrics: Dict,
        save_dir: Optional[str] = None
    ):
        """
        Compare local model vs federated model performance
        
        WHY COMPARE LOCAL VS FEDERATED?
        ================================
        Federated Learning should improve generalization:
        - Local models: Learn only from one hospital's data (limited diversity)
        - Federated model: Learns from multiple hospitals (more diverse data)
        - Better generalization: Federated model should perform better on unseen data
        
        HOW FL AFFECTS GENERALIZATION:
        ===============================
        1. More diverse training data (from multiple clients)
        2. Better feature representation (learned from varied populations)
        3. Reduced overfitting (model sees more varied patterns)
        4. Improved robustness (works across different data distributions)
        
        Args:
            local_metrics: Metrics from local model
            federated_metrics: Metrics from federated model
            save_dir: Optional directory to save comparison plots
        """
        print("\n" + "=" * 60)
        print("MODEL COMPARISON: LOCAL vs FEDERATED")
        print("=" * 60)
        
        # Create comparison table
        print(f"\n{'Metric':<20} {'Local Model':<20} {'Federated Model':<20} {'Difference':<20}")
        print("-" * 80)
        
        metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1_score']
        if 'auc' in local_metrics and 'auc' in federated_metrics:
            metrics_to_compare.append('auc')
        
        comparison_data = {}
        
        for metric in metrics_to_compare:
            if metric in local_metrics and metric in federated_metrics:
                local_val = local_metrics[metric]
                fed_val = federated_metrics[metric]
                diff = fed_val - local_val
                
                comparison_data[metric] = {
                    'local': local_val,
                    'federated': fed_val,
                    'difference': diff
                }
                
                print(f"{metric.capitalize():<20} {local_val:<20.4f} {fed_val:<20.4f} {diff:+.4f}")
        
        # Visualize comparison
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        # Bar chart comparison
        metrics_names = list(comparison_data.keys())
        local_values = [comparison_data[m]['local'] for m in metrics_names]
        fed_values = [comparison_data[m]['federated'] for m in metrics_names]
        
        x = np.arange(len(metrics_names))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars1 = ax.bar(x - width/2, local_values, width, label='Local Model', color='#3498db')
        bars2 = ax.bar(x + width/2, fed_values, width, label='Federated Model', color='#2ecc71')
        
        ax.set_xlabel('Metrics', fontsize=12, fontweight="bold")
        ax.set_ylabel('Score', fontsize=12, fontweight="bold")
        ax.set_title('Local vs Federated Model Comparison', fontsize=14, fontweight="bold", pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels([m.capitalize() for m in metrics_names])
        ax.legend(fontsize=11)
        ax.set_ylim([0, 1.1])
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save_dir:
            save_path = os.path.join(save_dir, "model_comparison.png")
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"\nComparison plot saved to {save_path}")
        
        plt.show()
        
        # Print interpretation
        print("\n" + "=" * 60)
        print("INTERPRETATION")
        print("=" * 60)
        
        avg_local = np.mean(local_values)
        avg_fed = np.mean(fed_values)
        
        if avg_fed > avg_local:
            print(f"✓ Federated model performs better (avg improvement: {avg_fed - avg_local:.4f})")
            print("  This demonstrates the benefit of collaborative learning!")
        elif avg_fed < avg_local:
            print(f"⚠ Local model performs better (difference: {avg_local - avg_fed:.4f})")
            print("  This may indicate need for more federated rounds or better aggregation")
        else:
            print("≈ Both models perform similarly")
        
        print("=" * 60)
    
    def print_evaluation_report(
        self,
        metrics: Dict,
        model_name: Optional[str] = None
    ):
        """
        Print comprehensive evaluation report
        
        Args:
            metrics: Dictionary containing evaluation metrics
            model_name: Name of the model (optional)
        """
        name = model_name or self.model_name
        
        print("\n" + "=" * 60)
        print(f"EVALUATION REPORT: {name}")
        print("=" * 60)
        
        print(f"\nTest Samples: {metrics.get('num_samples', 'N/A')}")
        
        print(f"\nClassification Metrics:")
        print(f"  Accuracy:  {metrics.get('accuracy', 0):.4f} ({metrics.get('accuracy', 0)*100:.2f}%)")
        print(f"  Precision: {metrics.get('precision', 0):.4f} ({metrics.get('precision', 0)*100:.2f}%)")
        print(f"  Recall:    {metrics.get('recall', 0):.4f} ({metrics.get('recall', 0)*100:.2f}%)")
        print(f"  F1-Score:  {metrics.get('f1_score', 0):.4f} ({metrics.get('f1_score', 0)*100:.2f}%)")
        
        if metrics.get('auc') is not None:
            print(f"  AUC-ROC:   {metrics.get('auc', 0):.4f}")
        
        if metrics.get('loss') is not None:
            print(f"  Loss:      {metrics.get('loss', 0):.4f}")
        
        # Healthcare interpretation
        print(f"\nHealthcare Interpretation:")
        recall = metrics.get('recall', 0)
        precision = metrics.get('precision', 0)
        
        if recall >= 0.9:
            print(f"  ✓ Excellent disease detection (Recall: {recall*100:.1f}%)")
        elif recall >= 0.8:
            print(f"  ✓ Good disease detection (Recall: {recall*100:.1f}%)")
        elif recall >= 0.7:
            print(f"  ⚠ Moderate disease detection (Recall: {recall*100:.1f}%)")
        else:
            print(f"  ❌ Poor disease detection (Recall: {recall*100:.1f}%) - MISSING CASES!")
        
        if precision >= 0.9:
            print(f"  ✓ Very few false alarms (Precision: {precision*100:.1f}%)")
        elif precision >= 0.8:
            print(f"  ✓ Acceptable false alarm rate (Precision: {precision*100:.1f}%)")
        else:
            print(f"  ⚠ High false alarm rate (Precision: {precision*100:.1f}%)")
        
        print("=" * 60)


def evaluate_model_comprehensive(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    model_name: str = "Model",
    device: str = "cpu",
    save_dir: Optional[str] = None,
    plot_results: bool = True
) -> Dict:
    """
    Comprehensive model evaluation with visualizations
    
    Args:
        model: PyTorch model to evaluate
        test_loader: DataLoader for test data
        model_name: Name of the model
        device: Device to use
        save_dir: Directory to save plots
        plot_results: Whether to plot results
        
    Returns:
        Dictionary containing evaluation metrics
    """
    evaluator = ModelEvaluator(model_name=model_name)
    
    # Evaluate model
    metrics = evaluator.evaluate_model(model, test_loader, device)
    
    # Print report
    evaluator.print_evaluation_report(metrics, model_name)
    
    # Plot results
    if plot_results:
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        # Confusion matrix
        evaluator.plot_confusion_matrix(
            metrics['confusion_matrix'],
            title=f"Confusion Matrix - {model_name}",
            save_path=os.path.join(save_dir, f"{model_name}_confusion_matrix.png") if save_dir else None
        )
        
        # ROC curve (if probabilities available)
        # Note: This would need to be called with probabilities from evaluate_model
        # For now, we'll skip ROC if not available
        
        # Metrics comparison
        metrics_to_plot = {
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-Score': metrics['f1_score']
        }
        
        evaluator.plot_metrics_comparison(
            metrics_to_plot,
            title=f"Performance Metrics - {model_name}",
            save_path=os.path.join(save_dir, f"{model_name}_metrics.png") if save_dir else None
        )
    
    return metrics


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("MODEL EVALUATION MODULE - PHASE 4")
    print("=" * 60)
    print("\nThis module provides comprehensive evaluation and visualization")
    print("for heart disease prediction models.")
    print("\nKey features:")
    print("  • Accuracy, Precision, Recall, F1-Score calculation")
    print("  • Confusion matrix visualization")
    print("  • ROC curve analysis")
    print("  • Local vs Federated model comparison")
    print("  • Healthcare-specific metric interpretation")

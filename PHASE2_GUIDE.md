# Phase 2 Quick Reference Guide

## ✅ Phase 2 Implementation Complete

All required files for Phase 2 have been created:

### New Files Created
- ✅ `client/model.py` - MLP model architecture
- ✅ `client/train.py` - Local training pipeline
- ✅ `client/evaluate.py` - Local model evaluation
- ✅ `phase2_main.py` - Main execution script
- ✅ `requirements.txt` - Updated with PyTorch

### Updated Files
- ✅ `README.md` - Added Phase 2 documentation

---

## 🧠 Model Architecture

**Multilayer Perceptron (MLP):**
```
Input Layer (dynamic) → Hidden Layer 1 (64 neurons, ReLU) → Hidden Layer 2 (32 neurons, ReLU) → Output Layer (1 neuron, Sigmoid)
```

**Key Design Decisions:**
- **Sigmoid**: Outputs probabilities (0-1) for binary classification
- **BCE Loss**: Standard for binary classification with Sigmoid
- **Adam Optimizer**: Adaptive learning rate, good default choice
- **Xavier Initialization**: Prevents vanishing/exploding gradients

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Ensure Phase 1 is Complete
- Client datasets should exist in `dataset/processed/`
- If not, run: `python phase1_main.py`

### 3. Run Phase 2
```bash
python phase2_main.py
```

This will:
- Train MLP models for all 3 clients
- Save models to `models/` directory
- Evaluate each model
- Generate summary report

---

## 📋 Individual Component Usage

### Train a Single Client Model
```python
from client.train import train_client_model

model = train_client_model(
    client_id=1,
    client_data_path="dataset/processed/client_1.csv",
    num_epochs=50,
    learning_rate=0.001,
    batch_size=32,
    save_model=True
)
```

### Evaluate a Single Client Model
```python
from client.evaluate import evaluate_client_model

metrics = evaluate_client_model(
    client_id=1,
    model_path="models/client_1_model.pth",
    client_data_path="dataset/processed/client_1.csv"
)
```

### Create Model Only
```python
from client.model import HeartDiseaseMLP

model = HeartDiseaseMLP(input_size=13)  # 13 features typical for heart disease dataset
```

---

## 📊 Expected Output

### Training Output
```
[Client 1] STARTING LOCAL TRAINING
[Client 1] Creating model with 13 input features...
[Client 1] Loss function: Binary Cross Entropy
[Client 1] Optimizer: Adam (lr=0.001)
[Client 1] Training for 50 epochs...

Epoch | Train Loss | Test Loss | Accuracy
------------------------------------------------------------
    1 |     0.6234 |    0.6123 |   0.6500
    5 |     0.5123 |    0.5012 |   0.7200
   10 |     0.4567 |    0.4456 |   0.7800
   ...
   50 |     0.3456 |    0.3345 |   0.8500

[Client 1] Training complete!
[Client 1] Best accuracy: 0.8500
[Client 1] Model saved to: models/client_1_model.pth
```

### Evaluation Output
```
[Client 1] EVALUATION RESULTS
============================================================
Test Samples: 67

Classification Metrics:
  Accuracy:  0.8507 (85.07%)
  Precision: 0.8333 (83.33%)
  Recall:    0.8333 (83.33%)
  F1-Score:  0.8333 (83.33%)

Confusion Matrix:
                  Predicted
                  No Disease  Disease
  Actual No Disease        30        6
  Actual Disease            4       27
```

---

## 🎓 Viva Points

### Why MLP for Medical Tabular Data?
1. Learns non-linear relationships between features
2. Handles both numerical and categorical features
3. Interpretable compared to complex models
4. Works well with moderate datasets
5. Captures feature interactions (e.g., age + cholesterol)

### Why Sigmoid?
- Outputs probabilities (0-1)
- Perfect for binary classification
- Works with BCE loss
- Smooth gradients

### Why BCE Loss?
- Designed for binary classification
- Works with Sigmoid
- Penalizes wrong predictions
- Standard in deep learning

### Why Adam Optimizer?
- Adaptive learning rate
- Less hyperparameter tuning
- Good default choice
- Handles sparse gradients

### Why Local Training First?
1. Establishes baseline performance
2. Validates model can learn
3. Easier debugging
4. Comparison with federated model
5. Clear demonstration of DL concepts

---

## 📁 File Structure

```
client/
├── model.py          # MLP architecture
├── train.py          # Training pipeline
└── evaluate.py       # Evaluation pipeline

models/               # Trained model weights
├── client_1_model.pth
├── client_2_model.pth
└── client_3_model.pth
```

---

## 🔧 Configuration

Default training parameters:
- **Epochs**: 50
- **Learning Rate**: 0.001
- **Batch Size**: 32
- **Device**: CPU (automatically detects CUDA if available)

Modify in `phase2_main.py` or pass as arguments to functions.

---

## ❓ Troubleshooting

**Issue**: Model not found
- **Solution**: Train model first using `client/train.py` or `phase2_main.py`

**Issue**: CUDA out of memory
- **Solution**: Code automatically falls back to CPU. Reduce batch_size if needed.

**Issue**: Import errors
- **Solution**: Install PyTorch: `pip install torch`

**Issue**: Client data not found
- **Solution**: Run Phase 1 first: `python phase1_main.py`

---

## 🔮 Next Steps (Phase 3)

Phase 3 will implement:
- Federated Learning framework (Flower)
- Client-server communication
- Model weight aggregation
- Global federated model
- Federated evaluation

---

**Phase 2 Complete! Ready for Phase 3: Federated Learning**


# Step-by-Step Execution Guide

## How to Run the Project and View Outputs

---

## 📋 Prerequisites Check

### Step 1: Verify Python Installation
```bash
python --version
```
**Expected**: Python 3.8 or higher

### Step 2: Check if Dataset Exists
```bash
# Check if dataset file exists
dir dataset\raw\heart.csv
```
**Expected**: File should exist. If not, download from:
- Kaggle: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset
- UCI ML: https://archive.ics.uci.edu/ml/datasets/heart+disease

---

## 🚀 Installation

### Step 1: Install Dependencies
Open Command Prompt or PowerShell in the project directory and run:

```bash
pip install -r requirements.txt
```

**This installs:**
- PyTorch (Deep Learning)
- Flower (Federated Learning)
- Pandas, NumPy (Data Processing)
- Scikit-learn (ML Utilities)
- Matplotlib, Seaborn (Visualization)

**Expected Output:**
```
Successfully installed torch-2.x.x flwr-1.x.x pandas-1.x.x ...
```

---

## 🎯 Execution Methods

### Method 1: Run Complete Project (Recommended)

**Command:**
```bash
python run_complete_project.py
```

**What it does:**
1. Checks prerequisites
2. Runs Phase 1: Dataset Engineering
3. Runs Phase 2: Local Training
4. Runs Phase 3: Federated Learning
5. Runs Phase 4: Evaluation

**Expected Output:**
- Console output showing progress for each phase
- Execution time summary
- Success/failure status for each phase

**Where to see output:**
- **Console/Terminal**: Real-time progress and results
- **Files created**: See "Output Locations" section below

---

### Method 2: Run Individual Phases

#### Phase 1: Dataset Engineering
```bash
python phase1_main.py
```

**Output Location:**
- **Console**: Dataset statistics, feature information
- **Files**: `dataset/processed/client_1.csv`, `client_2.csv`, `client_3.csv`

**What you'll see:**
```
============================================================
FEDERATED DEEP LEARNING FOR HEART DISEASE PREDICTION
PHASE 1: DATASET ENGINEERING & PREPROCESSING
============================================================

[Step 1] Loading dataset...
Dataset Shape: (1000, 14)
✓ Dataset loaded successfully!

[Step 2] Splitting dataset...
Client 1: 333 samples
Client 2: 333 samples
Client 3: 334 samples
✓ Client datasets created!
```

---

#### Phase 2: Local Training
```bash
python phase2_main.py
```

**Output Location:**
- **Console**: Training progress, loss, accuracy per epoch
- **Files**: `models/client_1_model.pth`, `client_2_model.pth`, `client_3_model.pth`

**What you'll see:**
```
============================================================
PHASE 2: DEEP LEARNING MODEL & LOCAL TRAINING
============================================================

[Client 1] Training...
Epoch | Train Loss | Test Loss | Accuracy
------------------------------------------------------------
    1 |     0.6234 |    0.6123 |   0.6500
    5 |     0.5123 |    0.5012 |   0.7200
   10 |     0.4567 |    0.4456 |   0.7800
   ...
   50 |     0.3456 |    0.3345 |   0.8500

✓ Client 1 model saved to: models/client_1_model.pth
```

---

#### Phase 3: Federated Learning
```bash
python phase3_main.py
```

**Output Location:**
- **Console**: Server and client logs, federated rounds progress
- **Files**: Federated model weights (if saved)

**What you'll see:**
```
============================================================
PHASE 3: FEDERATED LEARNING
============================================================

[Server] Starting on localhost:8080...
[Client 1] Connecting to server...
[Client 1] Training locally for 5 epochs...
[Client 2] Connecting to server...
[Client 3] Connecting to server...

Round 1/10:
  [Client 1] Loss: 0.3456, Accuracy: 0.8500
  [Client 2] Loss: 0.3321, Accuracy: 0.8600
  [Client 3] Loss: 0.3489, Accuracy: 0.8400
  Global Accuracy: 0.8500

...
```

---

#### Phase 4: Evaluation & Visualization
```bash
python phase4_main.py
```

**Output Location:**
- **Console**: Evaluation metrics, comparison results
- **Files**: `results/` directory with all plots

**What you'll see:**
```
============================================================
PHASE 4: EVALUATION & VISUALIZATION
============================================================

EVALUATION REPORT: Local Model - Client 1
============================================================
Accuracy:  0.8507 (85.07%)
Precision: 0.8333 (83.33%)
Recall:    0.8333 (83.33%)
F1-Score:  0.8333 (83.33%)

MODEL COMPARISON: LOCAL vs FEDERATED
============================================================
Metric        Local Model    Federated Model    Difference
------------------------------------------------------------
Accuracy      0.8200         0.8507             +0.0307
Precision     0.8100         0.8333             +0.0233
Recall        0.8000         0.8333             +0.0333
```

---

## 📁 Output Locations

### Console Output
- **Where**: Command Prompt/PowerShell/Terminal window
- **What**: Real-time progress, metrics, results
- **How to view**: Just run the scripts - output appears automatically

### Generated Files

#### 1. Client Datasets
**Location**: `dataset/processed/`
- `client_1.csv`
- `client_2.csv`
- `client_3.csv`

**How to view:**
```bash
# View file contents
type dataset\processed\client_1.csv
# Or open in Excel/Notepad
```

#### 2. Trained Models
**Location**: `models/`
- `client_1_model.pth`
- `client_2_model.pth`
- `client_3_model.pth`
- `federated_model.pth` (if saved)

**How to verify:**
```bash
dir models
```

#### 3. Evaluation Results & Plots
**Location**: `results/`
```
results/
├── client_1_local/
│   ├── confusion_matrix.png
│   └── metrics.png
├── client_2_local/
│   ├── confusion_matrix.png
│   └── metrics.png
├── client_3_local/
│   ├── confusion_matrix.png
│   └── metrics.png
├── federated/
│   ├── confusion_matrix.png
│   └── metrics.png
└── model_comparison.png
```

**How to view:**
- Open PNG files in any image viewer
- Or double-click to open with default image viewer
- All plots are 300 DPI (high quality for presentations)

---

## 🖥️ Step-by-Step Execution (Windows)

### Complete Example:

1. **Open Command Prompt**
   - Press `Win + R`
   - Type `cmd` and press Enter
   - Navigate to project directory:
     ```bash
     cd "C:\Users\91934\OneDrive\Desktop\Major projects\Sql Injection"
     ```

2. **Install Dependencies** (First time only)
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Complete Project**
   ```bash
   python run_complete_project.py
   ```

4. **View Outputs**
   - **Console**: See real-time progress
   - **Plots**: Open `results/` folder and view PNG files
   - **Models**: Check `models/` folder for saved weights

---

## 📊 Understanding the Output

### Console Output Sections

1. **Prerequisites Check**
   ```
   ✓ Python 3.10.x
   ✓ Dataset found
   ✓ Directories exist
   ```

2. **Phase Execution**
   ```
   PHASE 1: DATASET ENGINEERING
   [Step 1] Loading dataset...
   [Step 2] Splitting dataset...
   ✓ Phase 1 Complete!
   ```

3. **Training Progress** (Phase 2 & 3)
   ```
   Epoch | Train Loss | Test Loss | Accuracy
   ------------------------------------------------------------
       1 |     0.6234 |    0.6123 |   0.6500
       5 |     0.5123 |    0.5012 |   0.7200
   ```

4. **Evaluation Results** (Phase 4)
   ```
   EVALUATION REPORT
   Accuracy:  0.8507 (85.07%)
   Precision: 0.8333 (83.33%)
   Recall:    0.8333 (83.33%)
   ```

### Plot Files

- **Confusion Matrix**: Shows TN, FP, FN, TP
- **Metrics Comparison**: Bar chart of accuracy, precision, recall, F1
- **Model Comparison**: Local vs Federated side-by-side

---

## 🔍 Troubleshooting

### Issue: "Module not found"
**Solution:**
```bash
pip install -r requirements.txt
```

### Issue: "Dataset not found"
**Solution:**
1. Download heart.csv from Kaggle/UCI
2. Place in `dataset/raw/heart.csv`

### Issue: "No output files created"
**Solution:**
- Check if phases completed successfully
- Look for error messages in console
- Ensure you have write permissions

### Issue: "Plots not showing"
**Solution:**
- Plots are saved as PNG files in `results/` folder
- Open them with image viewer
- If using `plt.show()`, ensure GUI backend is available

---

## 💡 Quick Tips

1. **First Run**: Run `run_complete_project.py` to see everything
2. **Subsequent Runs**: Use `--skip-phase1` if dataset already processed
3. **View Plots**: Open `results/` folder - all PNG files are there
4. **Check Progress**: Watch console output for real-time updates
5. **Save Output**: Console output can be saved by redirecting:
   ```bash
   python run_complete_project.py > output.txt
   ```

---

## 📍 Summary: Where to See Output

| Output Type | Location | How to View |
|------------|----------|-------------|
| **Console Output** | Terminal/CMD | Automatic (appears while running) |
| **Plots/Charts** | `results/` folder | Open PNG files with image viewer |
| **Model Weights** | `models/` folder | Files are binary (PyTorch format) |
| **Client Data** | `dataset/processed/` | Open CSV files in Excel/Notepad |
| **Logs** | Console | Real-time during execution |

---

**Ready to run? Start with: `python run_complete_project.py`**


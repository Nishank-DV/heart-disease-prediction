# Quick Start - How to Execute & View Output

## 🚀 Fastest Way to Run

### Step 1: Open Command Prompt
- Press `Win + R`
- Type `cmd` and press Enter
- Navigate to your project folder:
  ```bash
  cd "C:\Users\91934\OneDrive\Desktop\Major projects\Sql Injection"
  ```

### Step 2: Install Dependencies (First Time Only)
```bash
pip install -r requirements.txt
```
**Wait for installation to complete** (takes 2-5 minutes)

### Step 3: Run the Complete Project
```bash
python run_complete_project.py
```

**That's it!** The script will:
- ✅ Check prerequisites
- ✅ Run all 4 phases automatically
- ✅ Show progress in the console
- ✅ Generate all outputs

---

## 📺 Where to See Output

### 1. **Console Output** (Terminal Window)
**What you'll see:**
- Real-time progress for each phase
- Training metrics (loss, accuracy)
- Evaluation results
- Success/failure messages

**Example:**
```
============================================================
PHASE 2: DEEP LEARNING MODEL & LOCAL TRAINING
============================================================

[Client 1] Training...
Epoch | Train Loss | Test Loss | Accuracy
    1 |     0.6234 |    0.6123 |   0.6500
    5 |     0.5123 |    0.5012 |   0.7200
   50 |     0.3456 |    0.3345 |   0.8500

✓ Client 1 model saved!
```

### 2. **Generated Files**

#### A. **Plots & Visualizations**
**Location**: `results/` folder

**Files created:**
- `results/client_1_local/confusion_matrix.png`
- `results/client_1_local/metrics.png`
- `results/federated/confusion_matrix.png`
- `results/model_comparison.png`

**How to view:**
- Open Windows File Explorer
- Navigate to `results` folder
- Double-click any PNG file to view

#### B. **Trained Models**
**Location**: `models/` folder

**Files created:**
- `models/client_1_model.pth`
- `models/client_2_model.pth`
- `models/client_3_model.pth`

**Note**: These are binary files (PyTorch format), not meant for direct viewing

#### C. **Client Datasets**
**Location**: `dataset/processed/` folder

**Files created:**
- `dataset/processed/client_1.csv`
- `dataset/processed/client_2.csv`
- `dataset/processed/client_3.csv`

**How to view:**
- Open in Excel or Notepad
- Or use: `type dataset\processed\client_1.csv`

---

## 🎯 Step-by-Step Visual Guide

### After Running `python run_complete_project.py`:

1. **Watch the Console**
   ```
   ============================================================
   CHECKING PREREQUISITES
   ============================================================
   ✓ Python 3.10.9
   ✓ Dataset found
   ✓ Directories exist
   
   ============================================================
   PHASE 1: DATASET ENGINEERING
   ============================================================
   Loading dataset...
   ✓ Dataset loaded: 1000 samples
   Splitting into 3 clients...
   ✓ Client datasets created!
   ```

2. **Check Generated Folders**
   - Open File Explorer
   - Look for `results/` folder (created after Phase 4)
   - Look for `models/` folder (created after Phase 2)
   - Look for `dataset/processed/` folder (created after Phase 1)

3. **View Results**
   - Go to `results/` folder
   - Open PNG files to see:
     - Confusion matrices
     - Performance metrics charts
     - Model comparisons

---

## 📊 What Output to Expect

### Console Output Structure:

```
[Prerequisites Check]
  ✓ Python version
  ✓ Dataset exists
  ✓ Dependencies installed

[Phase 1: Dataset Engineering]
  Dataset statistics
  Feature information
  Client dataset creation

[Phase 2: Local Training]
  Training progress (epochs)
  Loss and accuracy per epoch
  Model saving confirmation

[Phase 3: Federated Learning]
  Server starting
  Clients connecting
  Federated rounds progress
  Aggregated metrics

[Phase 4: Evaluation]
  Evaluation metrics
  Comparison results
  Plot generation confirmation

[Summary]
  Total execution time
  Success/failure status
```

### File Output Structure:

```
results/
├── client_1_local/
│   ├── confusion_matrix.png    ← Open this!
│   └── metrics.png            ← Open this!
├── client_2_local/
│   ├── confusion_matrix.png
│   └── metrics.png
├── client_3_local/
│   ├── confusion_matrix.png
│   └── metrics.png
├── federated/
│   ├── confusion_matrix.png
│   └── metrics.png
└── model_comparison.png        ← Open this!
```

---

## 🔍 Quick Verification

### Check if Everything Worked:

1. **Check Console for:**
   ```
   🎉 ALL PHASES COMPLETED SUCCESSFULLY!
   ```

2. **Check Files:**
   ```bash
   # In Command Prompt:
   dir results
   dir models
   dir dataset\processed
   ```

3. **View a Plot:**
   - Open `results/client_1_local/confusion_matrix.png`
   - Should see a heatmap with numbers

---

## ⚡ Alternative: Run Individual Phases

If you want to run phases separately:

```bash
# Phase 1 only
python phase1_main.py

# Phase 2 only
python phase2_main.py

# Phase 3 only
python phase3_main.py

# Phase 4 only
python phase4_main.py
```

**Output locations remain the same!**

---

## 💡 Pro Tips

1. **Save Console Output:**
   ```bash
   python run_complete_project.py > output_log.txt
   ```
   Then open `output_log.txt` to review

2. **Skip Completed Phases:**
   ```bash
   python run_complete_project.py --skip-phase1
   ```

3. **View Plots Immediately:**
   - After Phase 4 completes, plots are ready
   - No need to wait - open `results/` folder right away

4. **Check for Errors:**
   - Look for ❌ or "Error" in console
   - Check if files were created in expected folders

---

## 🆘 Common Issues

### "Module not found"
**Fix:**
```bash
pip install -r requirements.txt
```

### "Dataset not found"
**Fix:**
- Download `heart.csv` from Kaggle
- Place in `dataset/raw/heart.csv`

### "No plots generated"
**Fix:**
- Ensure Phase 4 completed successfully
- Check `results/` folder exists
- Re-run Phase 4: `python phase4_main.py`

---

## ✅ Success Checklist

After running, you should have:

- [ ] Console shows "ALL PHASES COMPLETED SUCCESSFULLY"
- [ ] `results/` folder exists with PNG files
- [ ] `models/` folder exists with .pth files
- [ ] `dataset/processed/` folder exists with CSV files
- [ ] Can open and view PNG plots
- [ ] No error messages in console

---

**Ready? Run: `python run_complete_project.py`**

**Then check the `results/` folder for your visualizations!**


# Quick Start Guide

## 🚀 Getting Started in 5 Minutes

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Verify Setup
```bash
python verify_setup.py
```

### Step 3: Get Dataset
```bash
python download_dataset.py
```

### Step 4: Run Federated Learning

**Easiest Method:**
```bash
python main.py
```

**Alternative (4 Terminals):**

Terminal 1:
```bash
python run_server.py
```

Terminal 2:
```bash
python run_client.py --client_id 1 --data_path dataset/clients/client_1_data.csv
```

Terminal 3:
```bash
python run_client.py --client_id 2 --data_path dataset/clients/client_2_data.csv
```

Terminal 4:
```bash
python run_client.py --client_id 3 --data_path dataset/clients/client_3_data.csv
```

### Step 5: Evaluate Model (Optional)
```bash
python evaluate_model.py --dataset dataset/heart.csv --save_plots
```

## 📁 Project Structure

```
federated-heart-disease/
├── client/              # Client-side code
├── server/              # Server-side code
├── utils/               # Evaluation utilities
├── dataset/             # Dataset directory
├── main.py              # Main execution script
├── run_server.py        # Standalone server
├── run_client.py        # Standalone client
└── evaluate_model.py    # Model evaluation
```

## 🔧 Common Commands

**Check everything is working:**
```bash
python verify_setup.py
```

**Split dataset for clients:**
```python
from main import split_dataset_for_federated_learning
split_dataset_for_federated_learning("dataset/heart.csv", num_clients=3)
```

**Run with custom parameters:**
```bash
python run_server.py --rounds 20 --epochs 10 --lr 0.0005
```

## ❓ Troubleshooting

**Problem:** Import errors
**Solution:** `pip install -r requirements.txt`

**Problem:** Dataset not found
**Solution:** `python download_dataset.py` or manually download from Kaggle

**Problem:** Port already in use
**Solution:** Change port in `run_server.py --address localhost:8081`

**Problem:** Clients can't connect
**Solution:** Make sure server is running first, wait 5 seconds after starting server

## 📊 Expected Output

When running successfully, you should see:
- Server waiting for clients
- Clients connecting and training
- Round-by-round progress updates
- Final aggregated metrics

## 🎓 For Academic Use

This project includes:
- ✅ Complete implementation
- ✅ Well-commented code
- ✅ Academic-quality structure
- ✅ Comprehensive evaluation
- ✅ Privacy-preserving approach

Perfect for B.Tech/BE major projects!


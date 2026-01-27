import os
import shutil

# Create directories
os.makedirs("dataset/raw", exist_ok=True)
os.makedirs("dataset/processed", exist_ok=True)

# Copy dataset if it exists
if os.path.exists("dataset/heart.csv"):
    shutil.copy2("dataset/heart.csv", "dataset/raw/heart.csv")
    print("Dataset copied to dataset/raw/heart.csv")
else:
    print("Dataset not found at dataset/heart.csv")

print("Directory structure ready!")


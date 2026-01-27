"""
Setup script for Phase 1
Creates directory structure and helps prepare the project
"""

import os
import shutil
from pathlib import Path


def create_directory_structure():
    """Create the required directory structure for Phase 1"""
    directories = [
        "dataset/raw",
        "dataset/processed",
        "client",
        "utils"
    ]
    
    print("Creating directory structure...")
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"  ✓ {directory}/")
    
    print("\n✓ Directory structure created!")


def move_existing_dataset():
    """Move existing dataset to raw folder if it exists"""
    old_paths = [
        "dataset/heart.csv",
        "heart.csv"
    ]
    
    new_path = "dataset/raw/heart.csv"
    
    # Check if dataset already exists in new location
    if os.path.exists(new_path):
        print(f"✓ Dataset already exists at {new_path}")
        return
    
    # Try to move from old locations
    for old_path in old_paths:
        if os.path.exists(old_path):
            print(f"Moving dataset from {old_path} to {new_path}...")
            shutil.move(old_path, new_path)
            print(f"✓ Dataset moved to {new_path}")
            return
    
    print(f"⚠ Dataset not found. Please place heart.csv at: {new_path}")


def check_setup():
    """Check if Phase 1 setup is complete"""
    print("\n" + "=" * 60)
    print("PHASE 1 SETUP CHECK")
    print("=" * 60)
    
    # Check directories
    required_dirs = [
        "dataset/raw",
        "dataset/processed",
        "client",
        "utils"
    ]
    
    all_dirs_exist = True
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"✓ {directory}/")
        else:
            print(f"❌ {directory}/ (missing)")
            all_dirs_exist = False
    
    # Check files
    required_files = [
        "utils/dataset_loader.py",
        "utils/feature_info.py",
        "utils/dataset_splitter.py",
        "client/data_preprocessing.py",
        "phase1_main.py",
        "requirements.txt",
        "README.md"
    ]
    
    all_files_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path}")
        else:
            print(f"❌ {file_path} (missing)")
            all_files_exist = False
    
    # Check dataset
    dataset_path = "dataset/raw/heart.csv"
    if os.path.exists(dataset_path):
        print(f"✓ {dataset_path}")
    else:
        print(f"⚠ {dataset_path} (not found - please download dataset)")
    
    print("=" * 60)
    
    if all_dirs_exist and all_files_exist:
        print("\n✓ Phase 1 setup is complete!")
        if not os.path.exists(dataset_path):
            print("\n⚠ Please download the dataset and place it at: dataset/raw/heart.csv")
        return True
    else:
        print("\n❌ Setup incomplete. Please check missing items above.")
        return False


def main():
    """Main setup function"""
    print("=" * 60)
    print("PHASE 1 SETUP")
    print("=" * 60)
    
    # Create directories
    create_directory_structure()
    
    # Move existing dataset if present
    move_existing_dataset()
    
    # Check setup
    check_setup()
    
    print("\n" + "=" * 60)
    print("SETUP COMPLETE")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Download dataset and place at: dataset/raw/heart.csv")
    print("3. Run Phase 1: python phase1_main.py")
    print("=" * 60)


if __name__ == "__main__":
    main()


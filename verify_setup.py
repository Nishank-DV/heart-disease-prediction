"""
Script to verify that the project setup is correct
Checks dependencies, directory structure, and basic imports
"""

import sys
import os

def check_python_version():
    """Check if Python version is 3.8+"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ required. Current: {version.major}.{version.minor}")
        return False
    print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    return True

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'numpy', 'pandas', 'sklearn', 'torch', 'flwr',
        'matplotlib', 'seaborn'
    ]
    
    missing = []
    for package in required_packages:
        try:
            if package == 'sklearn':
                __import__('sklearn')
            elif package == 'flwr':
                __import__('flwr')
            elif package == 'imblearn':
                __import__('imblearn')
            else:
                __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"❌ {package} not installed")
            missing.append(package)
    
    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    return True

def check_directory_structure():
    """Check if required directories and files exist"""
    required_dirs = ['client', 'server', 'utils', 'dataset']
    required_files = [
        'client/client.py',
        'client/model.py',
        'client/data_preprocessing.py',
        'server/server.py',
        'utils/evaluation.py',
        'main.py',
        'requirements.txt',
        'README.md'
    ]
    
    all_good = True
    
    for dir_name in required_dirs:
        if os.path.exists(dir_name) and os.path.isdir(dir_name):
            print(f"✓ Directory: {dir_name}/")
        else:
            print(f"❌ Directory missing: {dir_name}/")
            all_good = False
    
    for file_name in required_files:
        if os.path.exists(file_name):
            print(f"✓ File: {file_name}")
        else:
            print(f"❌ File missing: {file_name}")
            all_good = False
    
    return all_good

def check_imports():
    """Check if project modules can be imported"""
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    modules = [
        ('client.model', 'HeartDiseaseMLP'),
        ('client.client', 'FlowerClient'),
        ('client.data_preprocessing', 'DataPreprocessor'),
        ('server.server', 'start_federated_server'),
        ('utils.evaluation', 'evaluate_model_comprehensive'),
    ]
    
    all_good = True
    for module_path, class_name in modules:
        try:
            module = __import__(module_path, fromlist=[class_name])
            getattr(module, class_name)
            print(f"✓ Import: {module_path}.{class_name}")
        except Exception as e:
            print(f"❌ Import failed: {module_path}.{class_name} - {e}")
            all_good = False
    
    return all_good

def check_dataset():
    """Check if dataset exists"""
    dataset_path = os.path.join("dataset", "raw", "heart.csv")
    legacy_path = os.path.join("dataset", "heart.csv")
    if os.path.exists(dataset_path):
        print(f"✓ Dataset found: {dataset_path}")
        try:
            import pandas as pd
            df = pd.read_csv(dataset_path)
            print(f"  Dataset shape: {df.shape}")
            return True
        except Exception as e:
            print(f"❌ Error reading dataset: {e}")
            return False
    if os.path.exists(legacy_path):
        print(f"✓ Dataset found: {legacy_path}")
        try:
            import pandas as pd
            df = pd.read_csv(legacy_path)
            print(f"  Dataset shape: {df.shape}")
            return True
        except Exception as e:
            print(f"❌ Error reading dataset: {e}")
            return False
    else:
        print(f"⚠ Dataset not found: {dataset_path}")
        print("  Run: python download_dataset.py")
        return False

def main():
    print("=" * 60)
    print("PROJECT SETUP VERIFICATION")
    print("=" * 60)
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Directory Structure", check_directory_structure),
        ("Module Imports", check_imports),
        ("Dataset", check_dataset),
    ]
    
    results = []
    for name, check_func in checks:
        print(f"\n[{name}]")
        print("-" * 40)
        result = check_func()
        results.append((name, result))
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, result in results:
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
        if not result:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n🎉 All checks passed! Project is ready to use.")
        print("\nNext steps:")
        print("1. Ensure dataset/heart.csv exists")
        print("2. Run: python main.py")
        print("   OR")
        print("   Terminal 1: python run_server.py")
        print("   Terminal 2-4: python run_client.py --client_id X --data_path dataset/clients/client_X_data.csv")
    else:
        print("\n⚠ Some checks failed. Please fix the issues above.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


"""
Complete Project Execution Script - Phase 5
Runs all phases end-to-end for demonstration

This script executes the complete federated learning project:
1. Phase 1: Dataset Engineering & Preprocessing
2. Phase 2: Deep Learning Model & Local Training
3. Phase 3: Federated Learning
4. Phase 4: Evaluation & Visualization

Usage:
    python run_complete_project.py [--skip-phase1] [--skip-phase2] [--skip-phase3] [--skip-phase4]
"""

import os
import sys
import argparse
import subprocess
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def run_phase(phase_name: str, script_path: str, description: str) -> bool:
    """
    Run a phase of the project
    
    Args:
        phase_name: Name of the phase
        script_path: Path to the phase script
        description: Description of what the phase does
        
    Returns:
        True if successful, False otherwise
    """
    print("\n" + "=" * 60)
    print(f"PHASE: {phase_name}")
    print("=" * 60)
    print(f"Description: {description}")
    print(f"Script: {script_path}")
    print("=" * 60)
    
    if not os.path.exists(script_path):
        print(f"❌ Error: Script not found at {script_path}")
        return False
    
    try:
        print(f"\nStarting {phase_name}...")
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=False,
            text=True
        )
        
        if result.returncode == 0:
            print(f"\n✓ {phase_name} completed successfully!")
            return True
        else:
            print(f"\n❌ {phase_name} failed with return code {result.returncode}")
            return False
            
    except Exception as e:
        print(f"\n❌ Error running {phase_name}: {e}")
        return False


def check_prerequisites():
    """
    Check if prerequisites are met
    
    Returns:
        True if all prerequisites are met, False otherwise
    """
    print("=" * 60)
    print("CHECKING PREREQUISITES")
    print("=" * 60)
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print(f"❌ Python 3.8+ required. Current: {python_version.major}.{python_version.minor}")
        return False
    print(f"✓ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check dataset
    dataset_path = "dataset/raw/heart.csv"
    if os.path.exists(dataset_path):
        print(f"✓ Dataset found: {dataset_path}")
    else:
        print(f"⚠ Dataset not found: {dataset_path}")
        print("  Please download the heart disease dataset and place it at the above path")
        print("  Sources:")
        print("    - Kaggle: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset")
        print("    - UCI ML: https://archive.ics.uci.edu/ml/datasets/heart+disease")
        return False
    
    # Check required directories
    required_dirs = ["client", "server", "utils"]
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"✓ Directory exists: {dir_name}/")
        else:
            print(f"❌ Directory missing: {dir_name}/")
            return False
    
    print("\n✓ All prerequisites met!")
    return True


def main():
    """
    Main execution function
    """
    parser = argparse.ArgumentParser(
        description="Complete Federated Learning Project Execution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all phases
  python run_complete_project.py
  
  # Skip Phase 1 (if already done)
  python run_complete_project.py --skip-phase1
  
  # Run only evaluation
  python run_complete_project.py --skip-phase1 --skip-phase2 --skip-phase3
        """
    )
    
    parser.add_argument("--skip-phase1", action="store_true", 
                       help="Skip Phase 1 (Dataset Engineering)")
    parser.add_argument("--skip-phase2", action="store_true", 
                       help="Skip Phase 2 (Local Training)")
    parser.add_argument("--skip-phase3", action="store_true", 
                       help="Skip Phase 3 (Federated Learning)")
    parser.add_argument("--skip-phase4", action="store_true", 
                       help="Skip Phase 4 (Evaluation)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("FEDERATED DEEP LEARNING FOR HEART DISEASE PREDICTION")
    print("COMPLETE PROJECT EXECUTION - PHASE 5")
    print("=" * 60)
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites not met. Please fix the issues above.")
        return
    
    # Phase execution plan
    phases = [
        {
            "name": "Phase 1: Dataset Engineering & Preprocessing",
            "script": "phase1_main.py",
            "description": "Load dataset, split into client datasets, preprocess data",
            "skip": args.skip_phase1
        },
        {
            "name": "Phase 2: Deep Learning & Local Training",
            "script": "phase2_main.py",
            "description": "Train MLP models locally for each client",
            "skip": args.skip_phase2
        },
        {
            "name": "Phase 3: Federated Learning",
            "script": "phase3_main.py",
            "description": "Train federated model using Flower framework",
            "skip": args.skip_phase3
        },
        {
            "name": "Phase 4: Evaluation & Visualization",
            "script": "phase4_main.py",
            "description": "Evaluate models and generate visualizations",
            "skip": args.skip_phase4
        }
    ]
    
    # Execute phases
    print("\n" + "=" * 60)
    print("EXECUTION PLAN")
    print("=" * 60)
    
    for phase in phases:
        status = "⏭ SKIPPED" if phase["skip"] else "▶ RUN"
        print(f"{status}: {phase['name']}")
    
    print("\n" + "=" * 60)
    print("STARTING EXECUTION")
    print("=" * 60)
    
    start_time = time.time()
    results = []
    
    for phase in phases:
        if phase["skip"]:
            print(f"\n⏭ Skipping {phase['name']}")
            results.append((phase["name"], True, "Skipped"))
            continue
        
        success = run_phase(
            phase_name=phase["name"],
            script_path=phase["script"],
            description=phase["description"]
        )
        results.append((phase["name"], success, "Completed" if success else "Failed"))
        
        if not success:
            print(f"\n⚠ Warning: {phase['name']} failed. Continuing with next phase...")
            time.sleep(2)
    
    # Summary
    elapsed_time = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("EXECUTION SUMMARY")
    print("=" * 60)
    
    for phase_name, success, status in results:
        symbol = "✓" if success else "❌"
        print(f"{symbol} {phase_name}: {status}")
    
    print(f"\nTotal execution time: {elapsed_time/60:.2f} minutes")
    
    # Check results
    successful_phases = sum(1 for _, success, _ in results if success)
    total_phases = len([p for p in phases if not p["skip"]])
    
    print("\n" + "=" * 60)
    if successful_phases == total_phases:
        print("🎉 ALL PHASES COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nProject is ready for:")
        print("  ✓ Demonstration")
        print("  ✓ Viva presentation")
        print("  ✓ Academic submission")
        print("\nResults are saved in:")
        print("  • models/ - Trained model weights")
        print("  • results/ - Evaluation plots and metrics")
        print("  • dataset/processed/ - Client datasets")
    else:
        print("⚠ SOME PHASES FAILED")
        print("=" * 60)
        print(f"Completed: {successful_phases}/{total_phases} phases")
        print("Please check the errors above and rerun failed phases")
    
    print("=" * 60)


if __name__ == "__main__":
    main()


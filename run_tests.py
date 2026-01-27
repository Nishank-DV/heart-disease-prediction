"""
Test Runner Script
Run all tests for the Heart Disease Prediction System
"""

import pytest
import sys
import os

def main():
    """Run all tests"""
    print("=" * 60)
    print("HEART DISEASE PREDICTION - TEST SUITE")
    print("=" * 60)
    print("\nRunning all tests...")
    print("-" * 60)
    
    # Run pytest with verbose output
    exit_code = pytest.main([
        "tests/",
        "-v",
        "--tb=short",
        "--color=yes"
    ])
    
    print("\n" + "=" * 60)
    if exit_code == 0:
        print("✅ ALL TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED")
    print("=" * 60)
    
    return exit_code

if __name__ == "__main__":
    sys.exit(main())

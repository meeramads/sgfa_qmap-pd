#!/usr/bin/env python
"""Test script for Colab notebook imports."""

import sys
import importlib
from pathlib import Path

def test_import(module_name):
    """Test if a module can be imported."""
    try:
        importlib.import_module(module_name)
        return True, None
    except ImportError as e:
        return False, str(e)

def main():
    """Test all critical imports for the experimental framework."""
    print("=== Testing Experimental Framework Imports ===")
    
    # Add current directory to path
    sys.path.insert(0, str(Path.cwd()))
    
    # Test core modules
    modules_to_test = [
        "experiments.framework",
        "experiments.data_validation", 
        "experiments.method_comparison",
        "experiments.performance_benchmarks",
        "experiments.clinical_validation",
        "analysis.cross_validation",
        "data.preprocessing",
        "models.factory"
    ]
    
    results = {}
    for module in modules_to_test:
        success, error = test_import(module)
        results[module] = (success, error)
        
        if success:
            print(f"✅ {module}")
        else:
            print(f"❌ {module}: {error}")
    
    # Summary
    successful = sum(1 for success, _ in results.values() if success)
    total = len(results)
    
    print(f"\n=== Summary ===")
    print(f"Successful imports: {successful}/{total}")
    
    if successful == total:
        print("🎉 All imports successful! Colab notebook should work.")
        return True
    else:
        print("⚠️  Some imports failed - may need fixes for Colab.")
        return False

if __name__ == "__main__":
    main()
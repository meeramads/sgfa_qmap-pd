#!/usr/bin/env python
"""Test imports and provide fallback for Colab notebook."""

import sys
import os
from pathlib import Path

def test_experimental_framework():
    """Test if experimental framework can be imported."""
    sys.path.insert(0, str(Path.cwd()))
    
    try:
        # Test core framework
        from experiments.framework import ExperimentConfig
        print("✅ ExperimentConfig imported successfully")
        
        # Test individual experiment modules
        from experiments.data_validation import DataValidationExperiments
        print("✅ DataValidationExperiments imported successfully")
        
        from experiments.method_comparison import MethodComparisonExperiments  
        print("✅ MethodComparisonExperiments imported successfully")
        
        from experiments.performance_benchmarks import PerformanceBenchmarkExperiments
        print("✅ PerformanceBenchmarkExperiments imported successfully")
        
        from experiments.clinical_validation import ClinicalValidationExperiments
        print("✅ ClinicalValidationExperiments imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Experimental framework import failed: {e}")
        print("🔄 Falling back to legacy analysis only")
        return False

def run_legacy_analysis_only():
    """Fallback to run only legacy analysis if experimental framework fails."""
    print("\n=== Running Legacy Analysis Only ===")
    print("The experimental framework had import issues, but you can still run:")
    print("1. Direct analysis with run_analysis.py")
    print("2. Cross-validation experiments")
    print("3. Preprocessing pipelines")
    
    # Test basic run_analysis.py functionality
    print("\n🧪 Testing legacy analysis...")
    import subprocess
    
    try:
        result = subprocess.run([
            "python", "run_analysis.py", "--help"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("✅ run_analysis.py is working correctly")
            return True
        else:
            print(f"❌ run_analysis.py failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Could not test run_analysis.py: {e}")
        return False

if __name__ == "__main__":
    print("=== Testing SGFA Framework Imports ===")
    
    framework_works = test_experimental_framework()
    
    if not framework_works:
        legacy_works = run_legacy_analysis_only()
        if not legacy_works:
            print("\n❌ Both experimental framework and legacy analysis failed")
            print("Please check your environment setup")
        else:
            print("\n✅ Legacy analysis is available")
    else:
        print("\n🎉 Full experimental framework is available!")
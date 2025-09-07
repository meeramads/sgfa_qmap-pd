#!/usr/bin/env python
"""Test framework imports with JAX mocking."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

# Mock JAX and related modules that aren't available locally
sys.modules['jax'] = MagicMock()
sys.modules['jax.numpy'] = MagicMock()
sys.modules['jax.random'] = MagicMock()
sys.modules['jax.lax'] = MagicMock()
sys.modules['numpyro'] = MagicMock()
sys.modules['numpyro.distributions'] = MagicMock()
sys.modules['numpyro.infer'] = MagicMock()

# Add current directory to path
sys.path.insert(0, str(Path.cwd()))

def test_imports():
    """Test critical imports with mocked dependencies."""
    
    print("=== Testing Framework Imports (with JAX mocked) ===")
    
    try:
        from experiments.framework import ExperimentConfig, ExperimentFramework
        print("✅ ExperimentFramework imports successful")
    except Exception as e:
        print(f"❌ ExperimentFramework import failed: {e}")
        return False
    
    try:
        from experiments.data_validation import DataValidationExperiments
        print("✅ DataValidationExperiments import successful")
    except Exception as e:
        print(f"❌ DataValidationExperiments import failed: {e}")
        return False
        
    try:
        from experiments.method_comparison import MethodComparisonExperiments
        print("✅ MethodComparisonExperiments import successful")
    except Exception as e:
        print(f"❌ MethodComparisonExperiments import failed: {e}")
        return False
        
    try:
        from experiments.performance_benchmarks import PerformanceBenchmarkExperiments
        print("✅ PerformanceBenchmarkExperiments import successful")
    except Exception as e:
        print(f"❌ PerformanceBenchmarkExperiments import failed: {e}")
        return False
        
    try:
        from experiments.clinical_validation import ClinicalValidationExperiments
        print("✅ ClinicalValidationExperiments import successful")
    except Exception as e:
        print(f"❌ ClinicalValidationExperiments import failed: {e}")
        return False
    
    print("\n🎉 All experimental framework imports successful!")
    return True

if __name__ == "__main__":
    success = test_imports()
    if success:
        print("\n✅ Colab notebook should work correctly with these imports")
    else:
        print("\n❌ There are still import issues that need fixing")
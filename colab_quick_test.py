#!/usr/bin/env python
"""Quick test for Colab - should work now."""

import sys
sys.path.insert(0, '.')

print("=== Testing SGFA Framework ===")

try:
    # Test experimental framework
    from experiments.framework import ExperimentConfig
    from experiments.data_validation import DataValidationExperiments
    print("✅ Experimental framework imports successful")
    
    # Test configuration creation
    config = ExperimentConfig(
        experiment_name="test_config",
        description="Test configuration creation",
        dataset="synthetic"
    )
    print("✅ Configuration creation successful")
    
    # Test legacy analysis
    from analysis.cross_validation import CVRunner
    print("✅ Legacy analysis imports successful")
    
    print("\n🎉 All systems working! You can now run the Colab notebook.")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("There may still be dependency issues in Colab.")
    
print("\n=== Next Steps ===")
print("1. Run the updated Colab notebook")  
print("2. The framework should now import correctly")
print("3. If still issues, use the legacy analysis cells instead")
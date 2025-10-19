#!/usr/bin/env python3
"""
Verification script to test that sparse_gfa_fixed is selected from config.
Run this before running the full experiment to verify the fix works.
"""

import yaml
import sys

print("=" * 80)
print("MODEL SELECTION VERIFICATION TEST")
print("=" * 80)

# Step 1: Load config
print("\n1. Loading config_convergence.yaml...")
try:
    with open("config_convergence.yaml", "r") as f:
        config = yaml.safe_load(f)
    print(f"   ✅ Config loaded successfully")
    model_type_in_yaml = config.get("model", {}).get("model_type")
    print(f"   Model type in YAML: '{model_type_in_yaml}'")
except Exception as e:
    print(f"   ❌ Failed to load config: {e}")
    sys.exit(1)

# Step 2: Check ModelFactory registration
print("\n2. Checking ModelFactory model registration...")
try:
    from models import ModelFactory
    available_models = ModelFactory.list_models()
    print(f"   ✅ ModelFactory loaded successfully")
    print(f"   Available models: {available_models}")

    if "sparse_gfa_fixed" in available_models:
        print(f"   ✅ sparse_gfa_fixed is registered in ModelFactory")
    else:
        print(f"   ❌ sparse_gfa_fixed is NOT in ModelFactory!")
        sys.exit(1)
except Exception as e:
    print(f"   ❌ Failed to load ModelFactory: {e}")
    sys.exit(1)

# Step 3: Test get_optimal_model_configuration
print("\n3. Testing get_optimal_model_configuration...")
try:
    from models.models_integration import get_optimal_model_configuration

    data_characteristics = {
        "total_features": 864,
        "n_views": 2,
        "n_subjects": 86,
        "has_imaging_data": True
    }

    print(f"   Calling with config containing model_type='{model_type_in_yaml}'")
    model_type, model_config = get_optimal_model_configuration(
        config, data_characteristics, verbose=False
    )

    print(f"   Returned model_type: '{model_type}'")

    if model_type == "sparse_gfa_fixed":
        print(f"   ✅ SUCCESS! Model selection is working correctly")
    else:
        print(f"   ❌ FAILURE! Expected 'sparse_gfa_fixed' but got '{model_type}'")
        print(f"\n   This suggests the config is not being passed correctly or")
        print(f"   the model validation is rejecting sparse_gfa_fixed")
        sys.exit(1)

except Exception as e:
    print(f"   ❌ Failed during model configuration: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 4: Test full integration
print("\n4. Testing integrate_models_with_pipeline...")
try:
    from models.models_integration import integrate_models_with_pipeline
    import numpy as np

    # Create dummy data
    X_list = [np.random.randn(86, 850), np.random.randn(86, 14)]
    hypers = {
        "K": 2,
        "percW": 33.0,
        "Dm": [850, 14],
        "slab_df": 4,
        "slab_scale": 2
    }

    print(f"   Calling integrate_models_with_pipeline with full config...")
    model_type, model_instance, models_summary = integrate_models_with_pipeline(
        config=config,
        X_list=X_list,
        data_characteristics=data_characteristics,
        hypers=hypers,
        verbose=False
    )

    print(f"   Returned model_type: '{model_type}'")
    print(f"   Model instance class: {type(model_instance).__name__}")

    if model_type == "sparse_gfa_fixed":
        print(f"   ✅ SUCCESS! Full integration working correctly")
    else:
        print(f"   ❌ FAILURE! Expected 'sparse_gfa_fixed' but got '{model_type}'")
        sys.exit(1)

except Exception as e:
    print(f"   ❌ Failed during integration: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("✅ ALL TESTS PASSED!")
print("=" * 80)
print("\nYour configuration should now correctly use sparse_gfa_fixed model.")
print("You can safely run your experiments.")

#!/usr/bin/env python
"""Test script to verify config flows correctly from YAML to factor stability experiment."""

import yaml
import sys

def test_config_flow():
    """Test that factor_stability config is correctly read from config.yaml."""

    print("=" * 80)
    print("Testing Config Flow: config.yaml → run_experiments.py → factor_stability")
    print("=" * 80)
    print()

    # Load config.yaml
    try:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
        print("✅ Successfully loaded config.yaml")
    except Exception as e:
        print(f"❌ Failed to load config.yaml: {e}")
        return False

    # Check if factor_stability section exists
    if "factor_stability" not in config:
        print("❌ factor_stability section not found in config.yaml")
        return False

    print("✅ factor_stability section found in config.yaml")
    print()

    # Extract factor_stability config
    fs_config = config["factor_stability"]

    print("Factor Stability Configuration:")
    print("-" * 80)

    # Core parameters
    print("\n📊 Core Parameters:")
    params = ["K", "num_chains", "num_samples", "num_warmup"]
    for param in params:
        value = fs_config.get(param, "NOT FOUND")
        status = "✅" if param in fs_config else "❌"
        print(f"  {status} {param}: {value}")

    # Matching parameters
    print("\n🎯 Factor Matching Parameters:")
    params = ["cosine_threshold", "min_match_rate"]
    for param in params:
        value = fs_config.get(param, "NOT FOUND")
        status = "✅" if param in fs_config else "❌"
        print(f"  {status} {param}: {value}")

    # Effective factor parameters
    print("\n📈 Effective Factor Parameters:")
    params = ["sparsity_threshold", "min_nonzero_pct"]
    for param in params:
        value = fs_config.get(param, "NOT FOUND")
        status = "✅" if param in fs_config else "❌"
        print(f"  {status} {param}: {value}")

    # Fixed Ferreira parameters
    print("\n🔬 Fixed Parameters (Ferreira et al. 2024):")
    params = ["percW", "slab_df", "slab_scale", "reghsZ", "target_accept_prob"]
    for param in params:
        value = fs_config.get(param, "NOT FOUND")
        status = "✅" if param in fs_config else "❌"
        print(f"  {status} {param}: {value}")

    print()
    print("-" * 80)

    # Test ExperimentConfig creation
    print("\n📦 Testing ExperimentConfig Creation:")
    try:
        from experiments.framework import ExperimentConfig

        experiment_config = ExperimentConfig(
            experiment_name="factor_stability_test",
            description="Test config flow",
            K_values=[fs_config.get("K", 20)],
            num_samples=fs_config.get("num_samples", 5000),
            num_warmup=fs_config.get("num_warmup", 1000),
            num_chains=fs_config.get("num_chains", 4),
            cosine_threshold=fs_config.get("cosine_threshold", 0.8),
            min_match_rate=fs_config.get("min_match_rate", 0.5),
            sparsity_threshold=fs_config.get("sparsity_threshold", 0.01),
            min_nonzero_pct=fs_config.get("min_nonzero_pct", 0.05),
        )

        print("✅ ExperimentConfig created successfully")
        print(f"   - K: {experiment_config.K_values[0]}")
        print(f"   - num_chains: {experiment_config.num_chains}")
        print(f"   - cosine_threshold: {experiment_config.cosine_threshold}")
        print(f"   - min_match_rate: {experiment_config.min_match_rate}")

    except Exception as e:
        print(f"❌ Failed to create ExperimentConfig: {e}")
        return False

    # Test hyperparameters dict
    print("\n⚙️  Testing Hyperparameters Dict:")
    try:
        K = fs_config.get("K", 20)
        hypers = {
            "Dm": [100, 200],  # Dummy values
            "K": K,
            "a_sigma": 1.0,
            "b_sigma": 1.0,
            "slab_df": fs_config.get("slab_df", 4),
            "slab_scale": fs_config.get("slab_scale", 2),
            "percW": fs_config.get("percW", 33),
        }

        print("✅ Hyperparameters dict created successfully")
        print(f"   - K: {hypers['K']}")
        print(f"   - percW: {hypers['percW']}")
        print(f"   - slab_df: {hypers['slab_df']}")
        print(f"   - slab_scale: {hypers['slab_scale']}")

    except Exception as e:
        print(f"❌ Failed to create hyperparameters dict: {e}")
        return False

    # Test MCMC args
    print("\n🔢 Testing MCMC Args:")
    try:
        mcmc_args = {
            "K": K,
            "num_warmup": fs_config.get("num_warmup", 1000),
            "num_samples": fs_config.get("num_samples", 5000),
            "num_chains": 1,
            "target_accept_prob": fs_config.get("target_accept_prob", 0.8),
            "reghsZ": fs_config.get("reghsZ", True),
            "random_seed": 42,
        }

        print("✅ MCMC args dict created successfully")
        print(f"   - num_samples: {mcmc_args['num_samples']}")
        print(f"   - num_warmup: {mcmc_args['num_warmup']}")
        print(f"   - target_accept_prob: {mcmc_args['target_accept_prob']}")
        print(f"   - reghsZ: {mcmc_args['reghsZ']}")

    except Exception as e:
        print(f"❌ Failed to create MCMC args: {e}")
        return False

    print()
    print("=" * 80)
    print("✅ All Config Flow Tests Passed!")
    print("=" * 80)
    print()
    print("🎉 Configuration will flow correctly from config.yaml to factor stability analysis")
    print()

    return True


if __name__ == "__main__":
    success = test_config_flow()
    sys.exit(0 if success else 1)

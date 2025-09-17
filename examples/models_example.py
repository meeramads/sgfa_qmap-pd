"""Example usage of SGFA models and the model factory."""

import logging

import numpy as np

from core.io_utils import save_json, save_numpy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_basic_model_creation():
    """Basic model creation and configuration example."""
    print("=" * 60)
    print("BASIC MODEL CREATION EXAMPLE")
    print("=" * 60)

    from data import generate_synthetic_data
    from models import create_model

    # Generate synthetic data for testing
    data = generate_synthetic_data(num_sources=3, K=5, num_subjects=100)
    X_list = data["X_list"]

    print(f"Generated data: {len(X_list)} views")
    for i, X in enumerate(X_list):
        print(f"  View {i}: {X.shape}")

    # Create a basic configuration
    class MockConfig:
        def __init__(self):
            self.K = 5
            self.num_sources = len(X_list)
            self.reghsZ = True

    config = MockConfig()

    # Basic hyperparameters
    hypers = {
        "Dm": [X.shape[1] for X in X_list],  # Dimensions per view
        "percW": 33.0,  # Sparsity percentage
        "a_sigma": 1.0,
        "b_sigma": 1.0,
        "slab_df": 4.0,
        "slab_scale": 2.0,
    }

    print(f"\nConfiguration: K={config.K}, num_sources={config.num_sources}")
    print(f"Hyperparameters: {hypers}")

    # Create different model types
    model_types = ["sparseGFA", "standardGFA"]
    models = {}

    for model_type in model_types:
        try:
            model = create_model(model_type, config, hypers)
            models[model_type] = model
            print(f"✅ Created {model_type} model: {type(model).__name__}")
        except Exception as e:
            print(f"❌ Failed to create {model_type}: {e}")

    return models, X_list, config, hypers


def example_sparse_gfa_training():
    """Sparse GFA training with full MCMC example."""
    print("\n" + "=" * 60)
    print("SPARSE GFA TRAINING EXAMPLE")
    print("=" * 60)

    import jax
    import jax.numpy as jnp
    from numpyro.infer import MCMC, NUTS

    from data import generate_synthetic_data
    from models import create_model

    # Generate training data
    data = generate_synthetic_data(num_sources=2, K=3, num_subjects=50)
    X_list = [jnp.array(X) for X in data["X_list"]]

    print(f"Training data: {len(X_list)} views")
    for i, X in enumerate(X_list):
        print(f"  View {i}: {X.shape}")

    # Configuration for sparse GFA
    class Config:
        def __init__(self):
            self.K = 3
            self.num_sources = len(X_list)
            self.reghsZ = True  # Enable regularized horseshoe

    config = Config()

    # Hyperparameters optimized for small example
    hypers = {
        "Dm": jnp.array([X.shape[1] for X in X_list]),
        "percW": 25.0,  # More sparsity for small example
        "a_sigma": 2.0,
        "b_sigma": 1.0,
        "slab_df": 3.0,
        "slab_scale": 1.5,
    }

    print(f"Model config: K={config.K}, reghsZ={config.reghsZ}")
    print(f"Sparsity: {hypers['percW']}%")

    # Create and run sparse GFA model
    model = create_model("sparseGFA", config, hypers)

    # Set up MCMC
    key = jax.random.PRNGKey(42)
    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel, num_warmup=100, num_samples=200, num_chains=2)

    print("\n🔄 Running MCMC sampling...")
    mcmc.run(key, X_list)

    # Extract results
    samples = mcmc.get_samples()
    print(f"✅ MCMC completed!")
    print(f"   Samples collected: {list(samples.keys())}")

    # Compute posterior summaries
    W_mean = jnp.mean(samples["W"], axis=0)
    Z_mean = jnp.mean(samples["Z"], axis=0)
    sigma_mean = jnp.mean(samples["sigma"], axis=0)

    print(f"\nPosterior summaries:")
    print(f"  W shape: {W_mean.shape}")
    print(f"  Z shape: {Z_mean.shape}")
    print(f"  sigma: {sigma_mean}")

    # Analyze sparsity
    sparsity_threshold = 0.01
    sparse_elements = jnp.sum(jnp.abs(W_mean) < sparsity_threshold)
    total_elements = W_mean.size
    sparsity_achieved = 100 * sparse_elements / total_elements

    print(f"\nSparsity analysis:")
    print(f"  Elements < {sparsity_threshold}: {sparse_elements}/{total_elements}")
    print(f"  Achieved sparsity: {sparsity_achieved:.1f}%")
    print(f"  Target sparsity: {hypers['percW']:.1f}%")

    return {
        "samples": samples,
        "W_mean": W_mean,
        "Z_mean": Z_mean,
        "config": config,
        "hypers": hypers,
    }


def example_model_comparison():
    """Compare different model types on the same data."""
    print("\n" + "=" * 60)
    print("MODEL COMPARISON EXAMPLE")
    print("=" * 60)

    import jax
    import jax.numpy as jnp
    from numpyro.infer import MCMC, NUTS

    from data import generate_synthetic_data
    from models import create_model

    # Generate comparison data
    data = generate_synthetic_data(num_sources=2, K=4, num_subjects=60)
    X_list = [jnp.array(X) for X in data["X_list"]]

    print(f"Comparison data: {len(X_list)} views")

    # Base configuration
    class Config:
        def __init__(self, reg_horseshoe=True):
            self.K = 4
            self.num_sources = len(X_list)
            self.reghsZ = reg_horseshoe

    # Model configurations to compare
    model_configs = {
        "sparse_gfa_reg": {
            "type": "sparseGFA",
            "config": Config(reg_horseshoe=True),
            "hypers": {
                "Dm": jnp.array([X.shape[1] for X in X_list]),
                "percW": 30.0,
                "a_sigma": 2.0,
                "b_sigma": 1.0,
                "slab_df": 4.0,
                "slab_scale": 2.0,
            },
        },
        "sparse_gfa_noreg": {
            "type": "sparseGFA",
            "config": Config(reg_horseshoe=False),
            "hypers": {
                "Dm": jnp.array([X.shape[1] for X in X_list]),
                "percW": 30.0,
                "a_sigma": 2.0,
                "b_sigma": 1.0,
                "slab_df": 4.0,
                "slab_scale": 2.0,
            },
        },
        "standard_gfa": {
            "type": "standardGFA",
            "config": Config(reg_horseshoe=False),
            "hypers": {
                "Dm": jnp.array([X.shape[1] for X in X_list]),
                "a_sigma": 2.0,
                "b_sigma": 1.0,
            },
        },
    }

    results = {}
    key = jax.random.PRNGKey(123)

    for model_name, model_info in model_configs.items():
        print(f"\n🔄 Training {model_name}...")

        try:
            # Create model
            model = create_model(
                model_info["type"], model_info["config"], model_info["hypers"]
            )

            # Run MCMC (shorter for comparison)
            nuts_kernel = NUTS(model)
            mcmc = MCMC(nuts_kernel, num_warmup=50, num_samples=100, num_chains=1)

            key, subkey = jax.random.split(key)
            mcmc.run(subkey, X_list)

            samples = mcmc.get_samples()

            # Compute metrics
            W_mean = jnp.mean(samples["W"], axis=0)
            Z_mean = jnp.mean(samples["Z"], axis=0)

            # Reconstruction error
            X_recon = jnp.dot(Z_mean, W_mean.T)
            X_concat = jnp.concatenate(X_list, axis=1)
            recon_error = jnp.mean((X_concat - X_recon) ** 2)

            # Sparsity (for sparse models)
            sparsity = 0.0
            if "sparse" in model_name:
                sparsity = jnp.mean(jnp.abs(W_mean) < 0.01) * 100

            results[model_name] = {
                "samples": samples,
                "W_mean": W_mean,
                "Z_mean": Z_mean,
                "reconstruction_error": float(recon_error),
                "sparsity_percent": float(sparsity),
                "log_likelihood": float(
                    jnp.mean(mcmc.get_extra_fields()["potential_energy"])
                ),
            }

            print(f"✅ {model_name} completed")
            print(f"   Reconstruction error: {recon_error:.4f}")
            if sparsity > 0:
                print(f"   Sparsity achieved: {sparsity:.1f}%")

        except Exception as e:
            print(f"❌ {model_name} failed: {e}")
            results[model_name] = None

    # Compare results
    print(f"\n📊 MODEL COMPARISON SUMMARY")
    print("-" * 60)
    print(f"{'Model':<20} {'Recon Error':<12} {'Sparsity %':<12} {'Log Like':<12}")
    print("-" * 60)

    for model_name, result in results.items():
        if result:
            print(
                f"{model_name:<20} {result['reconstruction_error']:<12.4f} "
                f"{result['sparsity_percent']:<12.1f} {result['log_likelihood']:<12.1f}"
            )
        else:
            print(f"{model_name:<20} {'FAILED':<12} {'N/A':<12} {'N/A':<12}")

    return results


def example_hyperparameter_optimization():
    """Hyperparameter optimization and selection example."""
    print("\n" + "=" * 60)
    print("HYPERPARAMETER OPTIMIZATION EXAMPLE")
    print("=" * 60)

    import jax
    import jax.numpy as jnp
    from numpyro.infer import MCMC, NUTS

    from data import generate_synthetic_data
    from models import create_model

    # Generate optimization data
    data = generate_synthetic_data(num_sources=2, K=3, num_subjects=40)
    X_list = [jnp.array(X) for X in data["X_list"]]

    print(f"Optimization data: {len(X_list)} views")

    # Configuration
    class Config:
        def __init__(self):
            self.K = 3
            self.num_sources = len(X_list)
            self.reghsZ = True

    config = Config()

    # Hyperparameter grid to search
    param_grid = {
        "percW": [20.0, 30.0, 40.0],
        "slab_scale": [1.0, 1.5, 2.0],
        "a_sigma": [1.0, 2.0],
    }

    print(f"Parameter grid: {param_grid}")
    print(f"Total combinations: {np.prod([len(v) for v in param_grid.values()])}")

    best_score = float("inf")
    best_params = None
    results = []

    key = jax.random.PRNGKey(456)

    for percW in param_grid["percW"]:
        for slab_scale in param_grid["slab_scale"]:
            for a_sigma in param_grid["a_sigma"]:

                hypers = {
                    "Dm": jnp.array([X.shape[1] for X in X_list]),
                    "percW": percW,
                    "a_sigma": a_sigma,
                    "b_sigma": 1.0,
                    "slab_df": 4.0,
                    "slab_scale": slab_scale,
                }

                try:
                    print(
                        f"\n🔄 Testing percW={percW}, slab_scale={slab_scale}, a_sigma={a_sigma}"
                    )

                    # Create and run model
                    model = create_model("sparseGFA", config, hypers)
                    nuts_kernel = NUTS(model)
                    mcmc = MCMC(
                        nuts_kernel, num_warmup=30, num_samples=60, num_chains=1
                    )

                    key, subkey = jax.random.split(key)
                    mcmc.run(subkey, X_list)

                    # Evaluate model
                    samples = mcmc.get_samples()
                    W_mean = jnp.mean(samples["W"], axis=0)
                    Z_mean = jnp.mean(samples["Z"], axis=0)

                    # Reconstruction error as score
                    X_recon = jnp.dot(Z_mean, W_mean.T)
                    X_concat = jnp.concatenate(X_list, axis=1)
                    score = jnp.mean((X_concat - X_recon) ** 2)

                    # Sparsity achieved
                    sparsity = jnp.mean(jnp.abs(W_mean) < 0.01) * 100

                    result = {
                        "params": hypers.copy(),
                        "score": float(score),
                        "sparsity": float(sparsity),
                    }
                    results.append(result)

                    print(f"   Score: {score:.4f}, Sparsity: {sparsity:.1f}%")

                    if score < best_score:
                        best_score = score
                        best_params = hypers.copy()
                        print(f"   ⭐ New best score!")

                except Exception as e:
                    print(f"   ❌ Failed: {e}")

    # Summary
    print(f"\n🏆 HYPERPARAMETER OPTIMIZATION RESULTS")
    print("-" * 60)
    print(f"Best score: {best_score:.4f}")
    print(f"Best parameters:")
    for key, value in best_params.items():
        if key != "Dm":  # Skip array parameter
            print(f"  {key}: {value}")

    # Show top 3 results
    results.sort(key=lambda x: x["score"])
    print(f"\nTop 3 parameter combinations:")
    for i, result in enumerate(results[:3]):
        print(
            f"{i + 1}. Score: {result['score']:.4f}, "
            f"percW: {result['params']['percW']}, "
            f"slab_scale: {result['params']['slab_scale']}, "
            f"a_sigma: {result['params']['a_sigma']}"
        )

    return best_params, results


def example_model_diagnostics():
    """Model diagnostics and convergence checking example."""
    print("\n" + "=" * 60)
    print("MODEL DIAGNOSTICS EXAMPLE")
    print("=" * 60)

    import jax
    import jax.numpy as jnp
    from numpyro.diagnostics import effective_sample_size, gelman_rubin
    from numpyro.infer import MCMC, NUTS

    from data import generate_synthetic_data
    from models import create_model

    # Generate data
    data = generate_synthetic_data(num_sources=2, K=3, num_subjects=50)
    X_list = [jnp.array(X) for X in data["X_list"]]

    print(f"Diagnostic data: {len(X_list)} views")

    # Configuration
    class Config:
        def __init__(self):
            self.K = 3
            self.num_sources = len(X_list)
            self.reghsZ = True

    config = Config()
    hypers = {
        "Dm": jnp.array([X.shape[1] for X in X_list]),
        "percW": 30.0,
        "a_sigma": 2.0,
        "b_sigma": 1.0,
        "slab_df": 4.0,
        "slab_scale": 2.0,
    }

    # Run MCMC with multiple chains for diagnostics
    model = create_model("sparseGFA", config, hypers)
    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel, num_warmup=200, num_samples=500, num_chains=4)

    print("\n🔄 Running MCMC with 4 chains for diagnostics...")
    key = jax.random.PRNGKey(789)
    mcmc.run(key, X_list)

    samples = mcmc.get_samples()
    print(f"✅ MCMC completed with {len(samples)} parameter types")

    # Compute diagnostics
    print(f"\n📊 CONVERGENCE DIAGNOSTICS")
    print("-" * 60)

    # Effective sample size
    ess = effective_sample_size(samples)
    print(f"Effective Sample Sizes:")
    for param_name, ess_value in ess.items():
        if hasattr(ess_value, "shape") and ess_value.size > 1:
            print(
                f" {param_name}: mean={ jnp.mean(ess_value):.1f}, min={ jnp.min(ess_value):.1f}"
            )
        else:
            print(f"  {param_name}: {ess_value:.1f}")

    # Gelman-Rubin statistic (R-hat)
    rhat = gelman_rubin(samples)
    print(f"\nGelman-Rubin Statistics (R-hat):")
    for param_name, rhat_value in rhat.items():
        if hasattr(rhat_value, "shape") and rhat_value.size > 1:
            max_rhat = jnp.max(rhat_value)
            print(f"  {param_name}: max={max_rhat:.3f}")
            if max_rhat > 1.1:
                print(f"    ⚠️  Warning: R-hat > 1.1 indicates poor convergence")
        else:
            print(f"  {param_name}: {rhat_value:.3f}")
            if rhat_value > 1.1:
                print(f"    ⚠️  Warning: R-hat > 1.1 indicates poor convergence")

    # MCMC diagnostics from the sampler
    extra_fields = mcmc.get_extra_fields()

    print(f"\nMCMC Sampler Diagnostics:")
    if "num_steps" in extra_fields:
        print(f"  Average steps per sample: {jnp.mean(extra_fields['num_steps']):.1f}")
    if "accept_prob" in extra_fields:
        print(
            f" Average acceptance probability: { jnp.mean( extra_fields['accept_prob']):.3f}"
        )
    if "diverging" in extra_fields:
        divergent_count = jnp.sum(extra_fields["diverging"])
        total_samples = extra_fields["diverging"].size
        print(
            f" Divergent transitions: {divergent_count}/{total_samples} ({ 100 * divergent_count / total_samples:.1f}%)"
        )

    # Posterior summaries
    print(f"\nPOSTERIOR SUMMARIES")
    print("-" * 60)

    W_mean = jnp.mean(samples["W"], axis=0)
    W_std = jnp.std(samples["W"], axis=0)

    print(f"Factor loadings (W):")
    print(f"  Shape: {W_mean.shape}")
    print(f"  Mean magnitude: {jnp.mean(jnp.abs(W_mean)):.3f}")
    print(f"  Average uncertainty (std): {jnp.mean(W_std):.3f}")

    Z_mean = jnp.mean(samples["Z"], axis=0)
    Z_std = jnp.std(samples["Z"], axis=0)

    print(f"Factor scores (Z):")
    print(f"  Shape: {Z_mean.shape}")
    print(f"  Mean magnitude: {jnp.mean(jnp.abs(Z_mean)):.3f}")
    print(f"  Average uncertainty (std): {jnp.mean(Z_std):.3f}")

    return {
        "samples": samples,
        "diagnostics": {"ess": ess, "rhat": rhat, "extra_fields": extra_fields},
        "summaries": {
            "W_mean": W_mean,
            "W_std": W_std,
            "Z_mean": Z_mean,
            "Z_std": Z_std,
        },
    }


def example_saving_and_loading():
    """Model saving and loading example."""
    print("\n" + "=" * 60)
    print("MODEL SAVING AND LOADING EXAMPLE")
    print("=" * 60)

    import json
    import pickle
    from pathlib import Path

    # Use results from previous example
    print("🔄 Running a quick model training...")
    result = example_sparse_gfa_training()

    # Create results directory
    save_dir = Path("results/model_example")
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n💾 Saving model results to {save_dir}")

    # Save different components

    # 1. Save full samples (pickle)
    samples_file = save_dir / "mcmc_samples.pkl"
    with open(samples_file, "wb") as f:
        pickle.dump(result["samples"], f)
    print(f"✅ Saved MCMC samples: {samples_file}")

    # 2. Save posterior means (numpy)
    import numpy as np

    save_numpy(result["W_mean"], save_dir / "W_mean.npy")
    save_numpy(result["Z_mean"], save_dir / "Z_mean.npy")
    print(f"✅ Saved posterior means: W_mean.npy, Z_mean.npy")

    # 3. Save configuration and hyperparameters (JSON)
    config_dict = {
        "K": result["config"].K,
        "num_sources": result["config"].num_sources,
        "reghsZ": result["config"].reghsZ,
        "hyperparameters": {
            k: (v.tolist() if hasattr(v, "tolist") else v)
            for k, v in result["hypers"].items()
        },
    }

    config_file = save_dir / "model_config.json"
    save_json(config_dict, config_file)
    print(f"✅ Saved configuration: {config_file}")

    # 4. Create a summary report
    summary = {
        "model_type": "sparseGFA",
        "training_date": str(Path.ctime(Path.cwd())),
        "data_shape": {
            "num_subjects": int(result["Z_mean"].shape[0]),
            "num_factors": int(result["Z_mean"].shape[1]),
            "num_features": int(result["W_mean"].shape[0]),
        },
        "sparsity_achieved": float(np.mean(np.abs(result["W_mean"]) < 0.01) * 100),
        "files_saved": [
            "mcmc_samples.pkl",
            "W_mean.npy",
            "Z_mean.npy",
            "model_config.json",
        ],
    }

    summary_file = save_dir / "model_summary.json"
    save_json(summary, summary_file)
    print(f"✅ Saved summary: {summary_file}")

    # Demonstrate loading
    print(f"\n📂 Loading saved model results...")

    # Load samples
    with open(samples_file, "rb") as f:
        loaded_samples = pickle.load(f)
    print(f"✅ Loaded MCMC samples: {list(loaded_samples.keys())}")

    # Load posterior means
    loaded_W = np.load(save_dir / "W_mean.npy")
    loaded_Z = np.load(save_dir / "Z_mean.npy")
    print(f"✅ Loaded posterior means: W{loaded_W.shape}, Z{loaded_Z.shape}")

    # Load configuration
    with open(config_file, "r") as f:
        loaded_config = json.load(f)
    print(f"✅ Loaded configuration: K={loaded_config['K']}")

    # Verify integrity
    print(f"\n🔍 Verifying data integrity...")
    original_W_mean = np.array(result["W_mean"])
    if np.allclose(original_W_mean, loaded_W):
        print("✅ W matrices match perfectly")
    else:
        print("❌ W matrices don't match")

    original_Z_mean = np.array(result["Z_mean"])
    if np.allclose(original_Z_mean, loaded_Z):
        print("✅ Z matrices match perfectly")
    else:
        print("❌ Z matrices don't match")

    print(f"\nSaved files in {save_dir}:")
    for file in save_dir.iterdir():
        print(f"  {file.name}")

    return save_dir


if __name__ == "__main__":
    print("SGFA Models Usage Examples")
    print("=" * 60)

    # Run all examples
    try:
        # Basic examples
        models, X_list, config, hypers = example_basic_model_creation()

        # Advanced examples
        sparse_result = example_sparse_gfa_training()
        comparison_results = example_model_comparison()
        best_params, opt_results = example_hyperparameter_optimization()
        diagnostic_results = example_model_diagnostics()
        save_dir = example_saving_and_loading()

        print("\n" + "=" * 60)
        print("ALL MODEL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 60)

        print(f"\nKey takeaways:")
        print(f"• Created models: {list(models.keys())}")
        print(f"• Best hyperparameters found: percW={best_params['percW']}")
        print(f"• Results saved to: {save_dir}")
        print(f"• Multiple model types and configurations demonstrated")

    except Exception as e:
        logger.error(f"Example failed: {e}")
        raise

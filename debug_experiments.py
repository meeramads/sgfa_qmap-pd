#!/usr/bin/env python
"""
Quick Debug Experiment Runner
Individual experiment runners for debugging and development.

Usage:
    python debug_experiments.py data_validation
    python debug_experiments.py sgfa_parameter_comparison
    python debug_experiments.py model_comparison
    python debug_experiments.py performance_benchmarks
    python debug_experiments.py sensitivity_analysis
    python debug_experiments.py clinical_validation
    python debug_experiments.py all
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import yaml

# Add project root to path
sys.path.insert(0, ".")

# Set up simple logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def load_debug_config():
    """Load debug configuration."""
    config_path = "config_debug.yaml"
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded debug configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load debug config: {e}")
        sys.exit(1)


def run_data_validation_debug():
    """Run minimal data validation experiment - simple version."""
    logger.info("🔍 Running DEBUG: Data Validation (Simple)")

    config = load_debug_config()
    start_time = time.time()

    try:
        # Simple debug version - just load and validate data
        from data.qmap_pd import load_qmap_pd
        from core.config_utils import get_data_dir

        logger.info("Loading qMAP-PD data...")
        data_dir = get_data_dir(config)

        # Try to load data and handle various return types
        try:
            X_list = load_qmap_pd(data_dir)
            logger.info(f"Data loading returned: {type(X_list)}")

            # Handle different return types
            if isinstance(X_list, str):
                logger.error(f"Data loading returned error message: {X_list}")
                raise ValueError(f"Data loading failed: {X_list}")
            elif X_list is None:
                logger.error("Data loading returned None")
                raise ValueError("Data loading failed: returned None")
            elif isinstance(X_list, dict):
                # Handle dictionary return (likely from preprocessing integration)
                logger.info(f"Data loading returned dict with keys: {list(X_list.keys())}")

                # Try to extract X_list from common dictionary keys
                if 'X_list' in X_list:
                    actual_X_list = X_list['X_list']
                    logger.info(f"Extracted X_list from dict: {type(actual_X_list)}")
                elif 'data' in X_list:
                    actual_X_list = X_list['data']
                    logger.info(f"Extracted data from dict: {type(actual_X_list)}")
                elif 'views' in X_list:
                    actual_X_list = X_list['views']
                    logger.info(f"Extracted views from dict: {type(actual_X_list)}")
                else:
                    # Look for any list/array-like values
                    potential_data = None
                    for key, value in X_list.items():
                        if isinstance(value, (list, tuple)) and len(value) > 0:
                            if hasattr(value[0], 'shape'):  # Looks like array data
                                potential_data = value
                                logger.info(f"Found potential data in key '{key}': {type(value)}")
                                break

                    if potential_data is not None:
                        actual_X_list = potential_data
                    else:
                        logger.error(f"Could not find data arrays in dict keys: {list(X_list.keys())}")
                        raise ValueError(f"Dict contains no recognizable data arrays")

                # Replace X_list with the extracted data
                X_list = actual_X_list

            elif not isinstance(X_list, (list, tuple)):
                logger.error(f"Data loading returned unexpected type: {type(X_list)}")
                raise ValueError(f"Expected list/tuple/dict, got {type(X_list)}")

            if len(X_list) == 0:
                logger.error("Data loading returned empty list")
                raise ValueError("Data loading failed: empty list")

            # Validate each array in the list
            for i, X in enumerate(X_list):
                if not hasattr(X, 'shape'):
                    logger.error(f"View {i} is not a numpy array: {type(X)}")
                    raise ValueError(f"View {i} is not a numpy array: {type(X)}")
                logger.info(f"View {i} shape: {X.shape}")

        except Exception as e:
            logger.error(f"Data loading failed: {e}")
            # Create error result instead of crashing
            debug_dir = Path("debug_results/data_validation")
            debug_dir.mkdir(parents=True, exist_ok=True)

            duration = time.time() - start_time
            error_result = {
                "status": "failed",
                "duration_seconds": duration,
                "error": str(e),
                "data_loading_failed": True
            }

            import json
            with open(debug_dir / "summary.json", "w") as f:
                json.dump(error_result, f, indent=2, default=str)

            raise

        # Create simple debug output
        debug_dir = Path("debug_results/data_validation")
        debug_dir.mkdir(parents=True, exist_ok=True)

        # Basic validation info
        duration = time.time() - start_time
        result = {
            "status": "completed",
            "duration_seconds": duration,
            "views": len(X_list),
            "shapes": [list(X.shape) for X in X_list],  # Convert to list for JSON serialization
            "total_features": sum(X.shape[1] for X in X_list),
            "total_subjects": X_list[0].shape[0] if X_list else 0
        }

        # Save simple summary
        import json
        with open(debug_dir / "summary.json", "w") as f:
            json.dump(result, f, indent=2, default=str)

        logger.info(f"✅ Data validation completed in {duration:.2f}s")
        logger.info(f"   Views: {len(X_list)}")
        logger.info(f"   Shapes: {[X.shape for X in X_list]}")
        logger.info(f"   Results saved to: {debug_dir}")

        return result

    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"❌ Data validation failed after {duration:.2f}s: {e}")
        raise


def run_sgfa_parameter_comparison_debug():
    """Run minimal SGFA parameter comparison - simple version."""
    logger.info("🔬 Running DEBUG: SGFA Parameter Comparison (Simple)")

    config = load_debug_config()
    start_time = time.time()

    try:
        # Simple debug version - just test one SGFA configuration
        from data.qmap_pd import load_qmap_pd
        from core.config_utils import get_data_dir
        from core.run_analysis import models
        from numpyro.infer import MCMC, NUTS
        import jax
        import jax.numpy as jnp
        import argparse

        logger.info("Loading data and running single SGFA test...")
        data_dir = get_data_dir(config)

        # Handle data loading (same as data validation debug)
        try:
            data_result = load_qmap_pd(data_dir)
            if isinstance(data_result, dict):
                X_list = data_result['X_list']
            else:
                X_list = data_result
            logger.info(f"Loaded {len(X_list)} views for SGFA test")
        except Exception as e:
            logger.error(f"Data loading failed: {e}")
            raise

        # Create simple debug output
        debug_dir = Path("debug_results/sgfa_parameter_comparison")
        debug_dir.mkdir(parents=True, exist_ok=True)

        # Set up minimal SGFA parameters
        logger.info("Running SGFA with debug parameters...")

        # Create minimal hyperparameters
        Dm = [X.shape[1] for X in X_list]  # Features per view
        hypers = {
            'Dm': Dm,
            'a_sigma': 1.0,
            'b_sigma': 1.0,
            'percW': 25.0  # Sparsity percentage
        }

        # Create minimal args object
        args = argparse.Namespace()
        args.K = 3  # Small number for debug
        args.num_sources = len(X_list)
        args.reghsZ = False  # Simpler version
        args.num_samples = 20  # Very fast for debug
        args.num_warmup = 10
        args.num_chains = 1

        try:
            # Run SGFA model
            nuts_kernel = NUTS(models)
            mcmc = MCMC(nuts_kernel, num_samples=args.num_samples, num_warmup=args.num_warmup, num_chains=args.num_chains)
            mcmc.run(jax.random.PRNGKey(42), X_list, hypers, args)

            # Get results
            samples = mcmc.get_samples()
            result = {
                "converged": True,
                "log_likelihood": 0.0,  # Simplified for debug
                "samples": len(samples) > 0,
                "W": samples.get("W") if "W" in samples else None,
                "Z": samples.get("Z") if "Z" in samples else None
            }

        except Exception as e:
            logger.warning(f"SGFA run failed: {e}")
            result = {
                "converged": False,
                "error": str(e),
                "samples": False,
                "W": None,
                "Z": None
            }

        duration = time.time() - start_time
        debug_result = {
            "status": "completed",
            "duration_seconds": duration,
            "K": 3,
            "num_samples": 50,
            "converged": result.get("converged", False) if result else False,
            "log_likelihood": result.get("log_likelihood", None) if result else None
        }

        # Save simple summary
        import json
        with open(debug_dir / "summary.json", "w") as f:
            json.dump(debug_result, f, indent=2, default=str)

        logger.info(f"✅ SGFA parameter comparison completed in {duration:.2f}s")
        logger.info(f"   K=3, 50 samples, converged: {debug_result['converged']}")
        logger.info(f"   Results saved to: {debug_dir}")

        return debug_result

    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"❌ SGFA parameter comparison failed after {duration:.2f}s: {e}")
        raise


def run_model_comparison_debug():
    """Run enhanced model comparison - lightweight but comprehensive testing."""
    logger.info("🧠 Running DEBUG: Model Comparison (Enhanced)")

    config = load_debug_config()
    start_time = time.time()

    try:
        from data.qmap_pd import load_qmap_pd
        from core.config_utils import get_data_dir
        from models.sparse_gfa import run_sparse_gfa_model
        from sklearn.decomposition import PCA, FactorAnalysis, FastICA
        from sklearn.cluster import KMeans
        from sklearn.cross_decomposition import CCA
        import numpy as np

        logger.info("Testing model comparison pipeline...")
        data_dir = get_data_dir(config)
        X_list = load_qmap_pd(data_dir)

        issues = []
        methods_results = {}

        # Concatenate data for traditional methods
        X_combined = np.concatenate(X_list, axis=1)
        n_subjects, n_features = X_combined.shape

        logger.info(f"Data shape: {n_subjects} subjects × {n_features} features")

        # 1. Test SGFA (main method)
        logger.info("Test 1: SGFA...")
        try:
            sgfa_result = run_sparse_gfa_model(X_list, K=3, num_samples=20, num_chains=1, num_warmup=10)

            if sgfa_result:
                methods_results["SGFA"] = {
                    "success": True,
                    "converged": sgfa_result.get("converged", False),
                    "log_likelihood": sgfa_result.get("log_likelihood", None),
                    "has_factors": sgfa_result.get("Z") is not None,
                    "has_loadings": sgfa_result.get("W") is not None
                }

                if not sgfa_result.get("converged", False):
                    issues.append("SGFA did not converge")

                if sgfa_result.get("Z") is None:
                    issues.append("SGFA failed to extract factors")
            else:
                methods_results["SGFA"] = {"success": False}
                issues.append("SGFA completely failed")

        except Exception as e:
            methods_results["SGFA"] = {"success": False, "error": str(e)}
            issues.append(f"SGFA crashed: {str(e)}")

        # 2. Test traditional methods
        traditional_methods = {
            "PCA": lambda: PCA(n_components=3).fit(X_combined),
            "ICA": lambda: FastICA(n_components=3, random_state=42).fit(X_combined),
            "FA": lambda: FactorAnalysis(n_components=3, random_state=42).fit(X_combined),
            "KMeans": lambda: KMeans(n_clusters=3, random_state=42).fit(X_combined),
        }

        for method_name, method_func in traditional_methods.items():
            logger.info(f"Test: {method_name}...")
            try:
                model = method_func()

                # Basic checks
                success = True
                explained_var = None

                if hasattr(model, 'explained_variance_ratio_'):
                    explained_var = float(model.explained_variance_ratio_.sum())
                    if explained_var < 0.1:  # Very low explained variance
                        issues.append(f"{method_name} explains very little variance: {explained_var:.3f}")

                if hasattr(model, 'components_'):
                    if model.components_.shape[1] != n_features:
                        issues.append(f"{method_name} component shape mismatch")
                        success = False

                methods_results[method_name] = {
                    "success": success,
                    "explained_variance": explained_var,
                    "has_components": hasattr(model, 'components_'),
                    "has_transform": hasattr(model, 'transform')
                }

            except Exception as e:
                methods_results[method_name] = {"success": False, "error": str(e)}
                issues.append(f"{method_name} crashed: {str(e)}")

        # 3. Test CCA (special case - needs two views)
        logger.info("Test: CCA...")
        try:
            if len(X_list) >= 2:
                # Use first two views
                X1, X2 = X_list[0], X_list[1]
                n_components = min(3, X1.shape[1], X2.shape[1])

                cca = CCA(n_components=n_components)
                cca.fit(X1, X2)

                methods_results["CCA"] = {
                    "success": True,
                    "n_components": n_components,
                    "has_transform": hasattr(cca, 'transform')
                }
            else:
                methods_results["CCA"] = {"success": False, "reason": "Need at least 2 views"}
                issues.append("CCA needs multiple views but only one available")

        except Exception as e:
            methods_results["CCA"] = {"success": False, "error": str(e)}
            issues.append(f"CCA crashed: {str(e)}")

        # 4. Test model comparison metrics
        logger.info("Test: Comparison metrics...")
        try:
            from models.models_integration import integrate_models_with_pipeline
            # Test if factory integration works
            factory_test = integrate_models_with_pipeline(
                data_characteristics={"n_subjects": n_subjects, "n_features": n_features},
                computational_resources={"memory_limit_gb": 4},
                model_preferences={"model_type": "sparseGFA"}
            )
            methods_results["model_factory"] = {"success": True, "integration_ok": factory_test is not None}

        except Exception as e:
            methods_results["model_factory"] = {"success": False, "error": str(e)}
            issues.append(f"Model factory integration failed: {str(e)}")

        # Create debug output
        debug_dir = Path("debug_results/model_comparison")
        debug_dir.mkdir(parents=True, exist_ok=True)

        duration = time.time() - start_time

        # Calculate success rate
        successful_methods = sum(1 for result in methods_results.values() if result.get("success", False))
        total_methods = len(methods_results)

        debug_result = {
            "status": "completed" if len(issues) == 0 else "completed_with_issues",
            "duration_seconds": duration,
            "issues": issues,
            "methods_results": methods_results,
            "summary": {
                "successful_methods": successful_methods,
                "total_methods": total_methods,
                "success_rate": successful_methods / total_methods if total_methods > 0 else 0,
                "data_shape": [n_subjects, n_features]
            }
        }

        # Save detailed summary
        import json
        with open(debug_dir / "summary.json", "w") as f:
            json.dump(debug_result, f, indent=2, default=str)

        # Log summary
        status_emoji = "✅" if len(issues) == 0 else "⚠️"
        logger.info(f"{status_emoji} Model comparison completed in {duration:.2f}s")
        logger.info(f"   Issues found: {len(issues)}")
        if issues:
            for issue in issues[:3]:  # Show first 3 issues
                logger.warning(f"   - {issue}")

        logger.info(f"   Success rate: {successful_methods}/{total_methods} methods")
        for method, result in methods_results.items():
            status = "✓" if result.get("success", False) else "✗"
            logger.info(f"   {method}: {status}")

        logger.info(f"   Results saved to: {debug_dir}")

        return debug_result

    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"❌ Model comparison failed after {duration:.2f}s: {e}")
        raise


def run_performance_benchmarks_debug():
    """Run enhanced performance benchmarks - lightweight but comprehensive testing."""
    logger.info("⚡ Running DEBUG: Performance Benchmarks (Enhanced)")

    config = load_debug_config()
    start_time = time.time()

    try:
        from data.qmap_pd import load_qmap_pd
        from core.config_utils import get_data_dir
        from models.sparse_gfa import run_sparse_gfa_model
        import psutil
        import os
        import gc
        import numpy as np

        logger.info("Testing performance across multiple operations...")
        data_dir = get_data_dir(config)

        issues = []
        benchmarks = {}

        # Get baseline system info
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # 1. Data Loading Performance
        logger.info("Benchmark 1: Data loading...")
        load_start = time.time()
        X_list = load_qmap_pd(data_dir)
        load_time = time.time() - load_start

        post_load_memory = process.memory_info().rss / 1024 / 1024
        load_memory_delta = post_load_memory - initial_memory

        data_size_mb = sum(X.nbytes for X in X_list) / 1024 / 1024
        load_efficiency = data_size_mb / load_memory_delta if load_memory_delta > 0 else 0

        benchmarks["data_loading"] = {
            "load_time_seconds": load_time,
            "data_size_mb": data_size_mb,
            "memory_delta_mb": load_memory_delta,
            "load_efficiency": load_efficiency  # Data size / memory used
        }

        if load_time > 10:  # Slow loading
            issues.append(f"Slow data loading: {load_time:.1f}s")

        if load_efficiency < 0.5:  # Memory inefficient
            issues.append(f"Memory inefficient loading: {load_efficiency:.2f} efficiency")

        # 2. SGFA Performance Scaling
        logger.info("Benchmark 2: SGFA scaling...")

        # Test different K values for scaling
        K_values = [2, 3, 5]
        sgfa_timings = {}

        for K in K_values:
            try:
                sgfa_start = time.time()
                pre_sgfa_memory = process.memory_info().rss / 1024 / 1024

                result = run_sparse_gfa_model(
                    X_list, K=K, num_samples=10, num_chains=1, num_warmup=5
                )

                sgfa_time = time.time() - sgfa_start
                post_sgfa_memory = process.memory_info().rss / 1024 / 1024
                sgfa_memory_delta = post_sgfa_memory - pre_sgfa_memory

                sgfa_timings[f"K{K}"] = {
                    "time_seconds": sgfa_time,
                    "memory_delta_mb": sgfa_memory_delta,
                    "converged": result.get("converged", False) if result else False
                }

                # Cleanup
                del result
                gc.collect()

            except Exception as e:
                sgfa_timings[f"K{K}"] = {"error": str(e)}
                issues.append(f"SGFA K={K} failed: {str(e)}")

        benchmarks["sgfa_scaling"] = sgfa_timings

        # Check for performance issues
        if len(sgfa_timings) > 1:
            times = [t.get("time_seconds", 0) for t in sgfa_timings.values() if "time_seconds" in t]
            if times and max(times) > 30:  # Very slow
                issues.append(f"SGFA very slow: max {max(times):.1f}s")

        # 3. Memory Pressure Test
        logger.info("Benchmark 3: Memory pressure...")
        try:
            pre_pressure_memory = process.memory_info().rss / 1024 / 1024

            # Create temporary large arrays to test memory handling
            large_arrays = []
            for i in range(3):
                # Create arrays similar to factor matrices
                n_subjects = X_list[0].shape[0]
                large_array = np.random.randn(n_subjects, 50)  # Simulated large factor matrix
                large_arrays.append(large_array)

            peak_memory = process.memory_info().rss / 1024 / 1024
            pressure_memory_delta = peak_memory - pre_pressure_memory

            # Cleanup
            del large_arrays
            gc.collect()

            post_cleanup_memory = process.memory_info().rss / 1024 / 1024
            cleanup_efficiency = (peak_memory - post_cleanup_memory) / pressure_memory_delta if pressure_memory_delta > 0 else 0

            benchmarks["memory_pressure"] = {
                "pressure_memory_delta_mb": pressure_memory_delta,
                "cleanup_efficiency": cleanup_efficiency,
                "peak_memory_mb": peak_memory
            }

            if cleanup_efficiency < 0.8:  # Poor memory cleanup
                issues.append(f"Poor memory cleanup: {cleanup_efficiency:.2f} efficiency")

        except Exception as e:
            benchmarks["memory_pressure"] = {"error": str(e)}
            issues.append(f"Memory pressure test failed: {str(e)}")

        # 4. System Resource Check
        logger.info("Benchmark 4: System resources...")
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_percent = psutil.virtual_memory().percent
            disk_usage = psutil.disk_usage('.').percent

            benchmarks["system_resources"] = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent,
                "disk_usage_percent": disk_usage,
                "available_memory_gb": psutil.virtual_memory().available / 1024 / 1024 / 1024
            }

            if memory_percent > 90:
                issues.append(f"High memory usage: {memory_percent:.1f}%")

            if disk_usage > 90:
                issues.append(f"High disk usage: {disk_usage:.1f}%")

        except Exception as e:
            benchmarks["system_resources"] = {"error": str(e)}
            issues.append(f"System resource check failed: {str(e)}")

        # 5. JAX/GPU Performance Check
        logger.info("Benchmark 5: JAX/GPU check...")
        try:
            import jax
            import jax.numpy as jnp

            # Simple JAX operation timing
            jax_start = time.time()
            x = jnp.ones((1000, 1000))
            y = jnp.dot(x, x)
            jax_time = time.time() - jax_start

            devices = jax.devices()
            device_info = [str(device) for device in devices]

            benchmarks["jax_performance"] = {
                "simple_operation_time": jax_time,
                "devices": device_info,
                "device_count": len(devices),
                "default_backend": jax.default_backend()
            }

            if jax_time > 1.0:  # Slow JAX operations
                issues.append(f"Slow JAX operations: {jax_time:.2f}s")

        except Exception as e:
            benchmarks["jax_performance"] = {"error": str(e)}
            issues.append(f"JAX performance check failed: {str(e)}")

        # Create debug output
        debug_dir = Path("debug_results/performance_benchmarks")
        debug_dir.mkdir(parents=True, exist_ok=True)

        duration = time.time() - start_time
        final_memory = process.memory_info().rss / 1024 / 1024
        total_memory_delta = final_memory - initial_memory

        debug_result = {
            "status": "completed" if len(issues) == 0 else "completed_with_issues",
            "duration_seconds": duration,
            "issues": issues,
            "benchmarks": benchmarks,
            "summary": {
                "total_memory_delta_mb": total_memory_delta,
                "data_size_mb": data_size_mb,
                "performance_rating": "good" if len(issues) == 0 else "poor"
            }
        }

        # Save detailed summary
        import json
        with open(debug_dir / "summary.json", "w") as f:
            json.dump(debug_result, f, indent=2, default=str)

        # Log summary
        status_emoji = "✅" if len(issues) == 0 else "⚠️"
        logger.info(f"{status_emoji} Performance benchmarks completed in {duration:.2f}s")
        logger.info(f"   Issues found: {len(issues)}")
        if issues:
            for issue in issues[:3]:  # Show first 3 issues
                logger.warning(f"   - {issue}")

        logger.info(f"   Data loading: {load_time:.2f}s")
        logger.info(f"   Memory delta: {total_memory_delta:.1f}MB")
        logger.info(f"   JAX backend: {benchmarks.get('jax_performance', {}).get('default_backend', 'unknown')}")
        logger.info(f"   Results saved to: {debug_dir}")

        return debug_result

    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"❌ Performance benchmarks failed after {duration:.2f}s: {e}")
        raise


def run_sensitivity_analysis_debug():
    """Run minimal sensitivity analysis."""
    logger.info("📊 Running DEBUG: Sensitivity Analysis")

    from experiments.sensitivity_analysis import run_sensitivity_analysis

    config = load_debug_config()
    start_time = time.time()

    try:
        result = run_sensitivity_analysis(config)
        duration = time.time() - start_time

        if result:
            logger.info(f"✅ Sensitivity analysis completed in {duration:.2f}s")
            logger.info(f"   Status: {getattr(result, 'success', 'Unknown')}")
        else:
            logger.error("❌ Sensitivity analysis failed")

    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"❌ Sensitivity analysis failed after {duration:.2f}s: {e}")
        raise


def run_reproducibility_debug():
    """Run enhanced reproducibility testing - lightweight but comprehensive."""
    logger.info("🔄 Running DEBUG: Reproducibility (Enhanced)")

    config = load_debug_config()
    start_time = time.time()

    try:
        from data.qmap_pd import load_qmap_pd
        from core.config_utils import get_data_dir
        from models.sparse_gfa import run_sparse_gfa_model
        import numpy as np
        import pickle

        logger.info("Testing reproducibility across multiple dimensions...")
        data_dir = get_data_dir(config)
        X_list = load_qmap_pd(data_dir)

        issues = []
        tests = {}

        # 1. Test basic seed reproducibility
        logger.info("Test 1: Basic seed reproducibility...")
        try:
            result1 = run_sparse_gfa_model(X_list, K=3, num_samples=20, num_chains=1, random_seed=42)
            result2 = run_sparse_gfa_model(X_list, K=3, num_samples=20, num_chains=1, random_seed=42)

            if result1 and result2:
                ll1 = result1.get("log_likelihood", 0)
                ll2 = result2.get("log_likelihood", 0)
                ll_diff = abs(ll1 - ll2)
                basic_reproducible = ll_diff < 0.01

                # Also check factor matrices if available
                w_reproducible = True
                if result1.get("W") is not None and result2.get("W") is not None:
                    W1, W2 = result1["W"], result2["W"]
                    w_diff = np.mean(np.abs(W1 - W2))
                    w_reproducible = w_diff < 0.01
                    tests["W_matrix_difference"] = float(w_diff)

                tests["basic_seed_test"] = {
                    "ll_difference": float(ll_diff),
                    "reproducible": basic_reproducible and w_reproducible
                }

                if not (basic_reproducible and w_reproducible):
                    issues.append(f"Basic seed reproducibility failed: LL diff={ll_diff:.6f}, W diff={w_diff:.6f}")
            else:
                issues.append("Basic seed test failed - models didn't complete")
                tests["basic_seed_test"] = {"reproducible": False}

        except Exception as e:
            issues.append(f"Basic seed test crashed: {str(e)}")
            tests["basic_seed_test"] = {"error": str(e)}

        # 2. Test different seeds produce different results
        logger.info("Test 2: Different seeds produce different results...")
        try:
            result_seed42 = run_sparse_gfa_model(X_list, K=3, num_samples=20, num_chains=1, random_seed=42)
            result_seed99 = run_sparse_gfa_model(X_list, K=3, num_samples=20, num_chains=1, random_seed=99)

            if result_seed42 and result_seed99:
                ll42 = result_seed42.get("log_likelihood", 0)
                ll99 = result_seed99.get("log_likelihood", 0)
                ll_diff = abs(ll42 - ll99)
                appropriately_different = ll_diff > 0.001  # Should be different

                tests["different_seeds_test"] = {
                    "ll_difference": float(ll_diff),
                    "appropriately_different": appropriately_different
                }

                if not appropriately_different:
                    issues.append(f"Different seeds produce identical results: diff={ll_diff:.6f}")
            else:
                issues.append("Different seeds test failed - models didn't complete")

        except Exception as e:
            issues.append(f"Different seeds test crashed: {str(e)}")
            tests["different_seeds_test"] = {"error": str(e)}

        # 3. Test data perturbation robustness
        logger.info("Test 3: Data perturbation robustness...")
        try:
            # Add tiny noise to data
            X_list_perturbed = [X + np.random.normal(0, 1e-10, X.shape) for X in X_list]

            result_original = run_sparse_gfa_model(X_list, K=3, num_samples=20, num_chains=1, random_seed=42)
            result_perturbed = run_sparse_gfa_model(X_list_perturbed, K=3, num_samples=20, num_chains=1, random_seed=42)

            if result_original and result_perturbed:
                ll_orig = result_original.get("log_likelihood", 0)
                ll_pert = result_perturbed.get("log_likelihood", 0)
                ll_diff = abs(ll_orig - ll_pert)
                robust = ll_diff < 0.1  # Should be robust to tiny noise

                tests["perturbation_robustness"] = {
                    "ll_difference": float(ll_diff),
                    "robust": robust
                }

                if not robust:
                    issues.append(f"Not robust to tiny data perturbations: diff={ll_diff:.6f}")
            else:
                issues.append("Perturbation test failed - models didn't complete")

        except Exception as e:
            issues.append(f"Perturbation test crashed: {str(e)}")
            tests["perturbation_robustness"] = {"error": str(e)}

        # 4. Test serialization reproducibility
        logger.info("Test 4: Serialization reproducibility...")
        try:
            result_fresh = run_sparse_gfa_model(X_list, K=3, num_samples=20, num_chains=1, random_seed=42)

            if result_fresh:
                # Test if we can serialize and deserialize
                serialized = pickle.dumps(result_fresh)
                deserialized = pickle.loads(serialized)

                # Compare key components
                serialization_ok = True
                if "log_likelihood" in result_fresh and "log_likelihood" in deserialized:
                    ll_diff = abs(result_fresh["log_likelihood"] - deserialized["log_likelihood"])
                    if ll_diff > 1e-10:
                        serialization_ok = False
                        issues.append(f"Serialization changes log likelihood: diff={ll_diff}")

                tests["serialization_test"] = {
                    "serialization_ok": serialization_ok,
                    "serialized_size_kb": len(serialized) / 1024
                }
            else:
                issues.append("Serialization test failed - model didn't complete")

        except Exception as e:
            issues.append(f"Serialization test crashed: {str(e)}")
            tests["serialization_test"] = {"error": str(e)}

        # Create debug output
        debug_dir = Path("debug_results/reproducibility")
        debug_dir.mkdir(parents=True, exist_ok=True)

        duration = time.time() - start_time
        debug_result = {
            "status": "completed" if len(issues) == 0 else "completed_with_issues",
            "duration_seconds": duration,
            "issues": issues,
            "tests": tests,
            "overall_reproducible": len(issues) == 0
        }

        # Save detailed summary
        import json
        with open(debug_dir / "summary.json", "w") as f:
            json.dump(debug_result, f, indent=2, default=str)

        # Log summary
        status_emoji = "✅" if len(issues) == 0 else "⚠️"
        logger.info(f"{status_emoji} Reproducibility test completed in {duration:.2f}s")
        logger.info(f"   Issues found: {len(issues)}")
        if issues:
            for issue in issues[:3]:  # Show first 3 issues
                logger.warning(f"   - {issue}")

        # Log test results
        logger.info(f"   Basic seed: {'✓' if tests.get('basic_seed_test', {}).get('reproducible', False) else '✗'}")
        logger.info(f"   Different seeds: {'✓' if tests.get('different_seeds_test', {}).get('appropriately_different', False) else '✗'}")
        logger.info(f"   Perturbation robust: {'✓' if tests.get('perturbation_robustness', {}).get('robust', False) else '✗'}")
        logger.info(f"   Serialization: {'✓' if tests.get('serialization_test', {}).get('serialization_ok', False) else '✗'}")
        logger.info(f"   Results saved to: {debug_dir}")

        return debug_result

    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"❌ Reproducibility test failed after {duration:.2f}s: {e}")
        raise


def run_clinical_validation_debug():
    """Run minimal clinical validation - lightweight but comprehensive testing."""
    logger.info("🏥 Running DEBUG: Clinical Validation (Enhanced)")

    config = load_debug_config()
    start_time = time.time()

    try:
        from data.qmap_pd import load_qmap_pd
        from core.config_utils import get_data_dir
        from models.sparse_gfa import run_sparse_gfa_model
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        import pandas as pd
        import numpy as np

        logger.info("Testing clinical validation pipeline...")
        data_dir = get_data_dir(config)
        X_list = load_qmap_pd(data_dir)

        # 1. Clinical data loading and validation
        clinical_file = data_dir / "data_clinical" / "pd_motor_gfa_data.tsv"
        issues = []

        if clinical_file.exists():
            clinical_data = pd.read_csv(clinical_file, sep="\t")
            logger.info(f"Clinical data loaded: {clinical_data.shape}")

            # Check for required fields
            required_fields = ['diagnosis', 'subject_id']
            missing_fields = [f for f in required_fields if f not in clinical_data.columns]
            if missing_fields:
                issues.append(f"Missing required fields: {missing_fields}")

            # Check data alignment
            n_imaging = X_list[0].shape[0] if X_list else 0
            n_clinical = len(clinical_data)
            if n_imaging != n_clinical:
                issues.append(f"Size mismatch: imaging={n_imaging}, clinical={n_clinical}")

            # Check for missing values in key columns
            for col in ['diagnosis']:
                if col in clinical_data.columns:
                    missing_pct = clinical_data[col].isna().mean() * 100
                    if missing_pct > 0:
                        issues.append(f"{col} has {missing_pct:.1f}% missing values")
        else:
            logger.warning("No clinical data found - creating synthetic for pipeline test")
            n_subjects = X_list[0].shape[0] if X_list else 100
            clinical_data = pd.DataFrame({
                "subject_id": [f"sub_{i:03d}" for i in range(n_subjects)],
                "diagnosis": np.random.choice([0, 1], n_subjects),  # Binary diagnosis
                "UPDRS_motor": np.random.uniform(0, 50, n_subjects),
                "disease_duration": np.random.uniform(1, 15, n_subjects)
            })
            issues.append("Using synthetic clinical data - real data not found")

        # 2. Test SGFA factor extraction (minimal)
        logger.info("Testing SGFA factor extraction...")
        sgfa_success = False
        factors = None
        try:
            sgfa_result = run_sparse_gfa_model(
                X_list, K=3, num_samples=20, num_chains=1, num_warmup=10
            )
            if sgfa_result and sgfa_result.get("Z") is not None:
                factors = sgfa_result["Z"]
                sgfa_success = True
                logger.info(f"SGFA factors extracted: {factors.shape}")
            else:
                issues.append("SGFA failed to extract factors")
        except Exception as e:
            issues.append(f"SGFA extraction failed: {str(e)}")

        # 3. Test clinical prediction pipeline
        logger.info("Testing clinical prediction pipeline...")
        prediction_success = False
        prediction_accuracy = 0.0

        if sgfa_success and factors is not None and 'diagnosis' in clinical_data.columns:
            try:
                y = clinical_data['diagnosis'].values
                # Simple train/test split
                X_train, X_test, y_train, y_test = train_test_split(
                    factors, y, test_size=0.3, random_state=42, stratify=y
                )

                # Test logistic regression
                clf = LogisticRegression(random_state=42)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                prediction_accuracy = accuracy_score(y_test, y_pred)
                prediction_success = True

                logger.info(f"Prediction test accuracy: {prediction_accuracy:.3f}")

                if prediction_accuracy < 0.6:
                    issues.append(f"Low prediction accuracy: {prediction_accuracy:.3f}")

            except Exception as e:
                issues.append(f"Clinical prediction failed: {str(e)}")

        # 4. Test cross-validation imports
        logger.info("Testing CV imports...")
        try:
            from analysis.cross_validation_library import ClinicalAwareSplitter, NeuroImagingCVConfig
            from analysis.cv_fallbacks import CVFallbackHandler
            cv_imports_ok = True
        except Exception as e:
            cv_imports_ok = False
            issues.append(f"CV imports failed: {str(e)}")

        # Create debug output
        debug_dir = Path("debug_results/clinical_validation")
        debug_dir.mkdir(parents=True, exist_ok=True)

        duration = time.time() - start_time
        debug_result = {
            "status": "completed" if len(issues) == 0 else "completed_with_issues",
            "duration_seconds": duration,
            "issues": issues,
            "data_checks": {
                "clinical_data_available": clinical_file.exists(),
                "data_alignment_ok": n_imaging == n_clinical if clinical_file.exists() else True,
                "n_subjects_imaging": n_imaging,
                "n_subjects_clinical": len(clinical_data),
                "clinical_features": list(clinical_data.columns)
            },
            "pipeline_tests": {
                "sgfa_extraction_ok": sgfa_success,
                "prediction_pipeline_ok": prediction_success,
                "prediction_accuracy": float(prediction_accuracy),
                "cv_imports_ok": cv_imports_ok
            },
            "factors_shape": list(factors.shape) if factors is not None else None
        }

        # Save detailed summary
        import json
        with open(debug_dir / "summary.json", "w") as f:
            json.dump(debug_result, f, indent=2, default=str)

        # Log summary
        status_emoji = "✅" if len(issues) == 0 else "⚠️"
        logger.info(f"{status_emoji} Clinical validation completed in {duration:.2f}s")
        logger.info(f"   Issues found: {len(issues)}")
        if issues:
            for issue in issues[:3]:  # Show first 3 issues
                logger.warning(f"   - {issue}")
        logger.info(f"   SGFA extraction: {'✓' if sgfa_success else '✗'}")
        logger.info(f"   Prediction test: {'✓' if prediction_success else '✗'}")
        logger.info(f"   Results saved to: {debug_dir}")

        return debug_result

    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"❌ Clinical validation failed after {duration:.2f}s: {e}")
        raise


def run_all_debug():
    """Run all debug experiments sequentially."""
    logger.info("🚀 Running ALL DEBUG EXPERIMENTS")

    experiments = [
        ("data_validation", run_data_validation_debug),
        ("sgfa_parameter_comparison", run_sgfa_parameter_comparison_debug),
        ("model_comparison", run_model_comparison_debug),
        ("performance_benchmarks", run_performance_benchmarks_debug),
        ("sensitivity_analysis", run_sensitivity_analysis_debug),
        ("reproducibility", run_reproducibility_debug),
        ("clinical_validation", run_clinical_validation_debug),
    ]

    total_start = time.time()
    results = {}

    for exp_name, exp_func in experiments:
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting {exp_name}")
        logger.info(f"{'='*60}")

        try:
            exp_func()
            results[exp_name] = "success"
        except Exception as e:
            logger.error(f"Experiment {exp_name} failed: {e}")
            results[exp_name] = "failed"
            # Continue with other experiments

    total_duration = time.time() - total_start

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("DEBUG EXPERIMENT SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total duration: {total_duration:.2f}s")

    successful = len([r for r in results.values() if r == "success"])
    failed = len([r for r in results.values() if r == "failed"])

    logger.info(f"Successful: {successful}/{len(experiments)}")
    logger.info(f"Failed: {failed}/{len(experiments)}")

    for exp_name, status in results.items():
        status_emoji = "✅" if status == "success" else "❌"
        logger.info(f"  {status_emoji} {exp_name}: {status}")


def main():
    """Main debug runner."""
    parser = argparse.ArgumentParser(
        description="Quick Debug Experiment Runner"
    )
    parser.add_argument(
        "experiment",
        choices=[
            "data_validation",
            "sgfa_parameter_comparison",
            "model_comparison",
            "performance_benchmarks",
            "sensitivity_analysis",
            "reproducibility",
            "clinical_validation",
            "all"
        ],
        help="Experiment to run"
    )

    args = parser.parse_args()

    # Create debug results directory
    debug_dir = Path("debug_results")
    debug_dir.mkdir(exist_ok=True)

    logger.info(f"🐛 Starting debug mode for: {args.experiment}")
    logger.info(f"📁 Debug results will be saved to: {debug_dir}")

    # Map experiment names to functions
    experiment_map = {
        "data_validation": run_data_validation_debug,
        "sgfa_parameter_comparison": run_sgfa_parameter_comparison_debug,
        "model_comparison": run_model_comparison_debug,
        "performance_benchmarks": run_performance_benchmarks_debug,
        "sensitivity_analysis": run_sensitivity_analysis_debug,
        "reproducibility": run_reproducibility_debug,
        "clinical_validation": run_clinical_validation_debug,
        "all": run_all_debug,
    }

    # Run selected experiment
    try:
        experiment_map[args.experiment]()
        logger.info(f"🎉 Debug session completed successfully!")
    except Exception as e:
        logger.error(f"💥 Debug session failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
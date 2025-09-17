import contextlib
import gc
import logging
import os
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO)


def cleanup_memory(aggressive=False):
    """
    Enhanced memory cleanup for long-running analyses.

    Parameters
    ----------
    aggressive : bool
        If True, perform more aggressive cleanup (slower but more thorough)
    """
    import gc

    # Close all matplotlib figures
    try:
        import matplotlib.pyplot as plt

        plt.close("all")
        logging.debug("Closed all matplotlib figures")
    except ImportError:
        pass

    # Clear JAX compilation cache if needed
    try:
        import jax

        if aggressive:
            # Clear compiled functions (expensive but frees more memory)
            jax.clear_backends()
            logging.debug("Cleared JAX backends")
    except ImportError:
        pass

    # Force garbage collection multiple times for better cleanup
    collected_total = 0
    for _ in range(3 if aggressive else 1):
        collected = gc.collect()
        collected_total += collected

    if collected_total > 0:
        logging.debug(f"Garbage collected {collected_total} objects")

    # Log current memory usage if possible
    try:
        import os

        import psutil

        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        logging.debug(f"Current memory usage: {memory_mb:.1f} MB")
    except ImportError:
        pass


def cleanup_mcmc_samples(samples, keep_keys=None):
    """
    Clean up MCMC samples to keep only essential results.
    Use this after saving full samples to disk.
    """
    if keep_keys is None:
        keep_keys = ["W", "Z", "exp_logdensity", "time_elapsed"]

    cleaned = {}
    for key in keep_keys:
        if key in samples:
            cleaned[key] = samples[key]

    # Clear original samples dict
    samples.clear()

    # Force cleanup
    cleanup_memory(aggressive=True)

    return cleaned


@contextlib.contextmanager
def safe_plotting_context():
    """
    Context manager for safe plotting that ensures cleanup.

    Usage:
        with safe_plotting_context() as plt:
            plt.figure()
            plt.plot(data)
            plt.savefig('output.png')
            # Automatic cleanup on exit
    """
    import matplotlib.pyplot as plt

    try:
        yield plt
    except Exception as e:
        logging.error(f"Plotting error: {e}")
        raise
    finally:
        plt.close("all")
        # Force garbage collection for large plots
        gc.collect()


@contextlib.contextmanager
def memory_monitoring_context(operation_name="operation"):
    """
    Context manager that monitors memory usage during an operation.

    Parameters
    ----------
    operation_name : str
        Name of the operation for logging

    Usage:
        with memory_monitoring_context("MCMC inference"):
            # memory-intensive operation
            mcmc.run(...)
    """
    import os

    import psutil

    # Get initial memory
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB

    logging.info(f"Starting {operation_name} - Memory: {initial_memory:.1f} MB")

    try:
        yield
    finally:
        # Get final memory
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_diff = final_memory - initial_memory

        if memory_diff > 100:  # Log if significant memory increase
            logging.info(
                f"Completed {operation_name} - Memory: {final_memory:.1f} MB (+{memory_diff:.1f} MB)"
            )
        else:
            logging.debug(
                f"Completed {operation_name} - Memory: {final_memory:.1f} MB (+{memory_diff:.1f} MB)"
            )

        # Cleanup if memory usage is high
        if final_memory > 4000:  # > 4GB
            logging.info("High memory usage detected, running cleanup...")
            cleanup_memory()


def check_available_memory():
    """
    Check available system memory and warn if low.

    Returns
    -------
    available_gb : float
        Available memory in GB
    """
    try:
        import psutil

        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        total_gb = memory.total / (1024**3)
        used_percent = memory.percent

        logging.info(
            f"Memory status: { used_percent:.1f}% used, { available_gb:.1f}GB/{ total_gb:.1f}GB available"
        )

        if available_gb < 2:
            logging.warning(
                f"Low memory warning: Only {available_gb:.1f}GB available. "
                "Consider reducing batch sizes or using fewer chains."
            )
        elif available_gb < 4:
            logging.info(
                f"Memory note: { available_gb:.1f}GB available. Monitor memory usage during analysis."
            )

        return available_gb

    except ImportError:
        logging.debug("psutil not available for memory monitoring")
        return float("inf")  # Assume sufficient memory
    except Exception as e:
        logging.debug(f"Memory check failed: {e}")
        return float("inf")


def estimate_memory_requirements(
    n_subjects,
    n_features,
    n_factors,
    n_chains,
    n_samples,
    safety_factor=2.0,
    max_memory_gb=32,
):
    """
    Enhanced memory estimation with safety checks and limits.

    Parameters
    ----------
    safety_factor : float
        Multiply estimate by this factor for safety margin
    max_memory_gb : float
        Maximum allowed memory usage in GB
    """

    # Data storage: X matrices
    data_memory = n_subjects * n_features * 8  # float64

    # MCMC samples: W, Z, and other parameters
    W_memory = n_features * n_factors * n_chains * n_samples * 8
    Z_memory = n_subjects * n_factors * n_chains * n_samples * 8
    other_params_memory = (
        (n_features + n_subjects) * n_factors * n_chains * n_samples * 8 * 0.5
    )

    # Working memory (JAX compilation, intermediate calculations)
    working_memory = (data_memory + W_memory + Z_memory) * 1.5

    total_bytes = (
        data_memory + W_memory + Z_memory + other_params_memory + working_memory
    ) * safety_factor
    estimated_gb = total_bytes / (1024**3)

    logging.info(f"Enhanced memory estimation:")
    logging.info(f"  Data: {data_memory / (1024**3):.2f} GB")
    logging.info(
        f"  MCMC samples: {(W_memory + Z_memory + other_params_memory) / (1024**3):.2f} GB"
    )
    logging.info(f"  Working memory: {working_memory / (1024**3):.2f} GB")
    logging.info(f"  Total (with {safety_factor}x safety): {estimated_gb:.2f} GB")

    # CRITICAL CHECK - Raise error if too much memory
    if estimated_gb > max_memory_gb:
        raise MemoryError(
            f"Estimated memory requirement ({estimated_gb:.1f} GB) exceeds limit ({max_memory_gb} GB).\n"
            f"Suggestions:\n"
            f"  - Reduce --num-samples (current: {n_samples})\n"
            f"  - Reduce --num-chains (current: {n_chains})\n"
            f"  - Reduce --K factors (current: {n_factors})\n"
            f"  - Use feature selection to reduce features (current: {n_features})\n"
            f"  - Run on a high-memory system or use --quick_cv mode"
        )

    # Warnings for high memory usage
    if estimated_gb > 16:
        logging.warning(
            f"High memory requirement ({ estimated_gb:.1f} GB). Monitor system resources closely."
        )

    return estimated_gb


def check_memory_before_analysis(args, X_list):
    """
    Check memory requirements before starting analysis.
    Call this early in main() function.
    """
    if not X_list:
        return

    n_subjects = X_list[0].shape[0]
    n_features = sum(X.shape[1] for X in X_list)

    # Check different scenarios
    scenarios = [
        ("Standard Analysis", args.num_samples, args.num_chains),
        ("CV Analysis", 1000, 2),  # Reduced for CV
    ]

    for scenario_name, samples, chains in scenarios:
        try:
            estimated = estimate_memory_requirements(
                n_subjects,
                n_features,
                args.K,
                chains,
                samples,
                max_memory_gb=64,  # Higher limit for check, will warn
            )
            logging.info(f"{scenario_name} memory estimate: {estimated:.1f} GB")
        except MemoryError as e:
            logging.error(f"{scenario_name} will likely fail: {e}")
            if not getattr(args, "force_run", False):
                raise


def ensure_directory(path):
    """
    Ensure directory exists, create if necessary.

    Parameters
    ----------
    path : str or Path
        Directory path to create

    Returns
    -------
    path : Path
        Resolved Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_file_path(directory, filename):
    """
    Create safe file path with proper cross-platform handling.

    Parameters
    ----------
    directory : str or Path
        Directory path
    filename : str
        Filename

    Returns
    -------
    path : Path
        Safe file path
    """
    return Path(directory) / filename


def validate_file_exists(filepath, description="File"):
    """
    Validate that a file exists with informative error message.

    Parameters
    ----------
    filepath : str or Path
        Path to file
    description : str
        Description of file for error message

    Raises
    ------
    FileNotFoundError
        If file does not exist
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(
            f"{description} not found: {filepath}\n"
            f"Please check the path and ensure the file exists."
        )


def create_results_structure(base_dir, dataset, model, flag, flag_regZ):
    """
    Create standardized results directory structure.

    Parameters
    ----------
    base_dir : str
        Base results directory (e.g., '../results')
    dataset : str
        Dataset name
    model : str
        Model name
    flag : str
        Parameter flag string
    flag_regZ : str
        Regularization flag string

    Returns
    -------
    results_dir : Path
        Created results directory
    """
    base_path = Path(base_dir)
    results_dir = base_path / dataset / f"{model}_{flag}{flag_regZ}"
    ensure_directory(results_dir)
    return results_dir


def get_model_files(results_dir, run_id):
    """
    Get standardized model file paths for a given run.

    Parameters
    ----------
    results_dir : str or Path
        Results directory
    run_id : int
        Run identifier

    Returns
    -------
    files : dict
        Dictionary of file paths
    """
    results_dir = Path(results_dir)

    files = {
        "model_params": results_dir / f"[{run_id}]Model_params.dictionary",
        "robust_params": results_dir / f"[{run_id}]Robust_params.dictionary",
        "hyperparameters": results_dir / "hyperparameters.dictionary",
        "results_txt": results_dir / "results.txt",
        "plots_dir": results_dir / f"plots_{run_id}",
        "publication_dir": results_dir / f"plots_{run_id}" / "publication",
        "preprocessing_dir": results_dir / f"plots_{run_id}" / "preprocessing",
        "factor_maps_dir": results_dir / f"plots_{run_id}" / "factor_maps",
    }

    return files


def safe_pickle_load(filepath, max_retries=3, description="File"):
    """
    Safely load pickle file with error handling and backup support.

    Parameters
    ----------
    filepath : str or Path
        Path to pickle file
    max_retries : int
        Maximum number of retry attempts
    description : str
        Description for error messages

    Returns
    -------
    data : object
        Loaded data or None if failed
    """
    import pickle

    filepath = Path(filepath)
    backup_suffix = ".backup"  # Define backup suffix

    if not filepath.exists():
        logging.error(f"{description} not found: {filepath}")
        return None

    if filepath.stat().st_size <= 5:
        logging.error(f"{description} is empty or corrupted: {filepath}")
        return None

    # Try loading main file
    for attempt in range(max_retries):
        try:
            with open(filepath, "rb") as f:
                data = pickle.load(f)

            # Validate the loaded data
            if data is None:
                raise ValueError("Loaded data is None")

            logging.debug(f"Successfully loaded {description}: {filepath}")
            return data

        except (pickle.PickleError, EOFError, ValueError) as e:
            logging.warning(f"Attempt {attempt + 1} failed to load {description}: {e}")

            if attempt < max_retries - 1:
                # Try backup file if it exists
                backup_path = filepath.with_suffix(filepath.suffix + backup_suffix)
                if backup_path.exists():
                    logging.info(f"Trying backup file: {backup_path}")
                    filepath = backup_path
                    continue
            else:
                logging.error(
                    f"All {max_retries} attempts failed to load {description}"
                )

        except Exception as e:
            logging.error(f"Unexpected error loading {description}: {e}")
            break

    return None


def safe_pickle_save_with_backup(data, filepath, description="File"):
    """
    Save pickle with atomic write and backup creation.

    Parameters
    ----------
    data : object
        Data to save
    filepath : str or Path
        Path to save file
    description : str
        Description for logging

    Returns
    -------
    success : bool
        True if saved successfully
    """
    import pickle
    import shutil

    filepath = Path(filepath)
    temp_path = filepath.with_suffix(filepath.suffix + ".tmp")
    backup_path = filepath.with_suffix(filepath.suffix + ".backup")

    # Ensure directory exists
    ensure_directory(filepath.parent)

    try:
        # Create backup if original exists
        if filepath.exists() and filepath.stat().st_size > 5:
            shutil.copy2(filepath, backup_path)
            logging.debug(f"Created backup: {backup_path}")

        # Write to temporary file first (atomic write)
        with open(temp_path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Verify the written file
        with open(temp_path, "rb") as f:
            pickle.load(f)

        # If verification passed, move temp to final location
        temp_path.rename(filepath)

        logging.debug(f"Successfully saved {description}: {filepath}")
        return True

    except Exception as e:
        logging.error(f"Failed to save {description} to {filepath}: {e}")

        # Cleanup failed temp file
        if temp_path.exists():
            try:
                temp_path.unlink()
            except BaseException:
                pass

        return False


def safe_pickle_save(data, filepath, description="File"):
    """
    Safely save pickle file with error handling.

    Parameters
    ----------
    data : object
        Data to save
    filepath : str or Path
        Path to save file
    description : str
        Description for logging

    Returns
    -------
    success : bool
        True if saved successfully
    """
    import pickle

    filepath = Path(filepath)

    # Ensure directory exists
    ensure_directory(filepath.parent)

    try:
        with open(filepath, "wb") as f:
            pickle.dump(data, f)

        # Verify file was saved and is not empty
        if filepath.stat().st_size > 5:
            logging.debug(f"Successfully saved {description}: {filepath}")
            return True
        else:
            logging.error(f"Failed to save {description}: file is empty after save")
            return False

    except Exception as e:
        logging.error(f"Failed to save {description} to {filepath}: {e}")
        return False


def backup_file(filepath, max_backups=5):
    """
    Create backup of existing file before overwriting.

    Parameters
    ----------
    filepath : str or Path
        Path to file to backup
    max_backups : int
        Maximum number of backups to keep
    """
    filepath = Path(filepath)

    if not filepath.exists():
        return

    # Find next backup number
    backup_num = 1
    while backup_num <= max_backups:
        backup_path = filepath.with_suffix(f"{filepath.suffix}.bak{backup_num}")
        if not backup_path.exists():
            break
        backup_num += 1

    if backup_num <= max_backups:
        try:
            import shutil

            shutil.copy2(filepath, backup_path)
            logging.debug(f"Created backup: {backup_path}")
        except Exception as e:
            logging.warning(f"Failed to create backup of {filepath}: {e}")
    else:
        logging.warning(f"Maximum backups ({max_backups}) reached for {filepath}")


def clean_filename(filename):
    """
    Clean filename for cross-platform compatibility.

    Parameters
    ----------
    filename : str
        Original filename

    Returns
    -------
    clean_name : str
        Cross-platform safe filename
    """
    import re

    # Replace problematic characters
    clean_name = re.sub(r'[<>:"/\\|?*]', "_", filename)

    # Remove multiple underscores
    clean_name = re.sub(r"_+", "_", clean_name)

    # Remove leading/trailing underscores and dots
    clean_name = clean_name.strip("_.")

    # Ensure not too long (max 255 chars for most filesystems)
    if len(clean_name) > 200:
        name, ext = os.path.splitext(clean_name)
        clean_name = name[: 200 - len(ext)] + ext

    return clean_name


def get_relative_path(filepath, base_path):
    """
    Get relative path from base directory.

    Parameters
    ----------
    filepath : str or Path
        Full path to file
    base_path : str or Path
        Base directory path

    Returns
    -------
    rel_path : str
        Relative path string
    """
    try:
        return str(Path(filepath).relative_to(Path(base_path)))
    except ValueError:
        # If files are on different drives (Windows), return absolute path
        return str(Path(filepath).resolve())


def get_robustK(thrs, args, params, d_comps):
    """
    Identify robust components across multiple MCMC chains.

    Parameters
    ----------
    thrs : dict
        Thresholds for component matching
    args : object
        Arguments containing model parameters
    params : dict
        MCMC parameter samples
    d_comps : list
        Components in data space

    Returns
    -------
    rob_params : dict
        Robust parameters
    X_rob : list
        Robust components
    success : bool
        Whether robust components were found
    """
    logging.info("Running get_robustK")
    # Initialize parameters
    ncomps = args.K
    nchs = args.num_chains
    nsamples = args.num_samples
    t_nsamples = nsamples * nchs
    M, N, D = args.num_sources, params["Z"][0].shape[1], params["W"][0].shape[1]
    X_rob = [[] for _ in range(ncomps)]

    # horseshoe parameters (Z)
    Z = np.full([t_nsamples, N, ncomps], np.nan)
    if args.model == "sparseGFA":
        lmbZ = np.full([t_nsamples, N, ncomps], np.nan)
        tauZ = np.full([t_nsamples, ncomps], np.nan)
        if args.reghsZ:
            cZ = np.full([t_nsamples, ncomps], np.nan)

    # horseshoe parameters (W)
    W = np.full([t_nsamples, D, ncomps], np.nan)
    if "sparseGFA" in args.model:
        lmbW = np.full([t_nsamples, D, ncomps], np.nan)
        cW = np.full([t_nsamples, M, ncomps], np.nan)
    elif args.model == "GFA":
        alpha = np.full([t_nsamples, M, ncomps], np.nan)

    # Initialise parameters to select components
    storecomps = [np.arange(ncomps) for _ in range(nchs)]
    cosThr = thrs["cosineThr"]
    matchThr = thrs["matchThr"]
    nrobcomp = 0
    for c1 in range(nchs):
        max_sim = np.zeros((ncomps, nchs))
        max_simW = np.zeros((ncomps, nchs))
        max_simZ = np.zeros((ncomps, nchs))
        matchInds = np.zeros((ncomps, nchs))
        for k in storecomps[c1]:
            nonempty_chs = []
            for ne in range(len(storecomps)):
                if storecomps[ne].size > 0:
                    nonempty_chs.append(ne)

            for c2 in nonempty_chs:
                cosine = np.zeros((1, storecomps[c2].size))
                cosW = np.zeros((1, storecomps[c2].size))
                cosZ = np.zeros((1, storecomps[c2].size))
                cind = 0
                for comp in storecomps[c2]:
                    # X
                    comp1 = np.ndarray.flatten(d_comps[c2][comp])
                    comp2 = np.ndarray.flatten(d_comps[c1][k])
                    cosine[0, cind] = np.dot(comp1, comp2) / (
                        np.linalg.norm(comp1) * np.linalg.norm(comp2)
                    )
                    # W
                    compW1 = np.mean(params["W"][c2], axis=0)[:, comp]
                    compW2 = np.mean(params["W"][c1], axis=0)[:, k]
                    cosW[0, cind] = np.dot(compW1, compW2) / (
                        np.linalg.norm(compW1) * np.linalg.norm(compW2)
                    )
                    # Z
                    compZ1 = np.mean(params["Z"][c2], axis=0)[:, comp]
                    compZ2 = np.mean(params["Z"][c1], axis=0)[:, k]
                    cosZ[0, cind] = np.dot(compZ1, compZ2) / (
                        np.linalg.norm(compZ1) * np.linalg.norm(compZ2)
                    )
                    cind += 1
                # find the most similar components
                max_sim[k, c2] = cosine[0, np.argmax(cosine)]
                max_simW[k, c2] = cosW[0, np.argmax(cosine)]
                max_simZ[k, c2] = cosZ[0, np.argmax(cosine)]
                matchInds[k, c2] = storecomps[c2][np.argmax(cosine)]
                if max_sim[k, c2] < cosThr:
                    matchInds[k, c2] = -1

            if np.sum(max_sim[k, :] > cosThr) > matchThr * nchs:
                goodInds = np.where(matchInds[k, :] >= 0)
                X_rob[k] = np.zeros((N, D))
                s = 0
                for c2 in list(goodInds[0]):
                    inds = np.arange(s, s + nsamples)
                    # components in the data space
                    X_rob[k] += d_comps[c2][int(matchInds[k, c2])]
                    # parameters
                    if args.model == "sparseGFA":
                        lmbZ[inds, :, k] = params["lmbZ"][c2][
                            :, :, int(matchInds[k, c2])
                        ]
                        tauZ[inds, k] = params["tauZ"][c2][:, int(matchInds[k, c2])]
                        if args.reghsZ:
                            cZ[inds, k] = params["cZ"][c2][:, int(matchInds[k, c2])]
                    if "sparseGFA" in args.model:
                        lmbW[inds, :, k] = params["lmbW"][c2][
                            :, :, int(matchInds[k, c2])
                        ]
                        cW[inds, :, k] = params["cW"][c2][:, :, int(matchInds[k, c2])]
                    elif args.model == "GFA":
                        alpha[inds, :, k] = params["alpha"][c2][
                            :, :, int(matchInds[k, c2])
                        ]
                    # W
                    if max_simW[k, c2] > 0:
                        W[inds, :, k] = params["W"][c2][:, :, int(matchInds[k, c2])]
                    else:
                        W[inds, :, k] = -params["W"][c2][:, :, int(matchInds[k, c2])]
                    # Z
                    if max_simZ[k, c2] > 0:
                        Z[inds, :, k] = params["Z"][c2][:, :, int(matchInds[k, c2])]
                    else:
                        Z[inds, :, k] = -params["Z"][c2][:, :, int(matchInds[k, c2])]
                    # remove robust components from storecomps
                    storecomps[c2] = storecomps[c2][
                        storecomps[c2] != int(matchInds[k, c2])
                    ]
                    # update samples
                    s += nsamples
                # divided by total number of robust components
                X_rob[k] = [X_rob[k] / np.sum(matchInds[k, :] >= 0)]
                nrobcomp += 1
    success = True
    if nrobcomp > 0:
        # Remove non-robust components and discard chains
        idx_cols = ~np.isnan(np.mean(Z, axis=1)).all(axis=0)
        idx_rows = ~np.isnan(np.mean(Z, axis=1)[:, idx_cols]).any(axis=1)
        if args.model == "sparseGFA":
            lmbZ = lmbZ[idx_rows, :, :]
            lmbZ = lmbZ[:, :, idx_cols]
            tauZ = tauZ[idx_rows, :]
            tauZ = tauZ[:, idx_cols]
            if args.reghsZ:
                cZ = cZ[idx_rows, :]
                cZ = cZ[:, idx_cols]
                if cZ.size == 0:
                    logging.warning("No samples survived!")
                    success = False
        W = W[idx_rows, :, :]
        W = W[:, :, idx_cols]
        Z = Z[idx_rows, :, :]
        Z = Z[:, :, idx_cols]
        if "sparseGFA" in args.model:
            lmbW = lmbW[idx_rows, :, :]
            lmbW = lmbW[:, :, idx_cols]
            cW = cW[idx_rows, :, :]
            cW = cW[:, :, idx_cols]
        elif args.model == "GFA":
            alpha = alpha[idx_rows, :, :]
            alpha = alpha[:, :, idx_cols]
        X_rob_final = [X_rob[i] for i in range(len(X_rob)) if X_rob[i] != []]
        X_rob = X_rob_final
    else:
        logging.warning("No robust components found!")
        success = False

    # create dictionary with robust params (save the posterior mean for most parameters)
    rob_params = {"W": np.mean(W, axis=0), "Z": np.mean(Z, axis=0)}
    if "sparseGFA" in args.model:
        rob_params.update(
            {"cW_inf": np.mean(cW, axis=0), "lmbW": np.mean(lmbW, axis=0)}
        )
    elif args.model == "GFA":
        rob_params.update({"alpha_inf": np.mean(alpha, axis=0)})
    if args.model == "sparseGFA":
        if args.reghsZ:
            rob_params.update(
                {"cZ_inf": cZ, "tauZ_inf": tauZ, "lmbZ": np.mean(lmbZ, axis=0)}
            )
        else:
            rob_params.update({"tauZ_inf": tauZ, "lmbZ": np.mean(lmbZ, axis=0)})
    return rob_params, X_rob, success


def get_infparams(samples, hypers, args):
    """
    Extract inferred parameters from MCMC samples.

    Parameters
    ----------
    samples : dict
        MCMC samples
    hypers : dict
        Hyperparameters
    args : object
        Arguments containing model parameters

    Returns
    -------
    params : dict
        Inferred parameters
    d_comps : list
        Components in data space
    """
    # Get inferred parameters
    nchs, nsamples = args.num_chains, args.num_samples
    N = samples["Z"].shape[1]
    K = args.K
    # Get Z
    Z_inf = [np.zeros((nsamples, N, K)) for _ in range(nchs)]
    if args.model == "sparseGFA":
        lmbZ_inf = [np.zeros((nsamples, N, K)) for _ in range(nchs)]
        tauZ_inf = [np.zeros((nsamples, K)) for _ in range(nchs)]
        if args.reghsZ:
            cZ_inf = [np.zeros((nsamples, K)) for _ in range(nchs)]
    # Get W
    D = sum(hypers["Dm"])
    W_inf = [np.zeros((nsamples, D, K)) for _ in range(nchs)]
    if "sparseGFA" in args.model:
        lmbW_inf = [np.zeros((nsamples, D, K)) for _ in range(nchs)]
        cW_inf = [np.zeros((nsamples, K)) for _ in range(nchs)]
        tauW_inf = np.zeros((nchs * nsamples, args.num_sources))
    elif args.model == "GFA":
        alpha_inf = [np.zeros((nsamples, K)) for _ in range(nchs)]
    sigma_inf = np.zeros((nchs * nsamples, args.num_sources))
    d_comps = [[[] for _ in range(K)] for _ in range(nchs)]
    s = 0
    for c in range(nchs):
        inds = np.arange(s, s + nsamples)
        # get sigma
        sigma_inf[inds, :] = samples["sigma"][inds, 0, :]
        # get Z, tauZ, lmbZ, cZ
        for k in range(K):
            if args.model == "sparseGFA":
                tauZ_inf[c][:, k] = np.array(samples[f"tauZ"])[inds, 0, k]
                tauZ = np.reshape(tauZ_inf[c][:, k], (nsamples, 1))
                if args.reghsZ:
                    cZ_inf[c][:, k] = hypers["slab_scale"] * np.sqrt(
                        samples["cZ"][inds, 0, k]
                    )
                    cZ = np.reshape(cZ_inf[c][:, k], (nsamples, 1))
                    lmbZ_sqr = np.square(np.array(samples["lmbZ"])[inds, :, k])
                    lmbZ_inf[c][:, :, k] = np.sqrt(
                        lmbZ_sqr * cZ**2 / (cZ**2 + tauZ**2 * lmbZ_sqr)
                    )
                else:
                    lmbZ_inf[c][:, :, k] = np.array(samples["lmbZ"])[inds, :, k]
                Z_inf[c][:, :, k] = (
                    np.array(samples["Z"])[inds, :, k] * lmbZ_inf[c][:, :, k] * tauZ
                )
            else:
                Z_inf[c][:, :, k] = np.array(samples["Z"])[inds, :, k]

        # get W, tauW, lmbW, cW
        if "sparseGFA" in args.model:
            cW_inf[c] = hypers["slab_scale"] * np.sqrt(samples["cW"][inds, :, :])
            Dm = hypers["Dm"]
            d = 0
            for m in range(args.num_sources):
                lmbW_sqr = np.square(np.array(samples["lmbW"][inds, d : d + Dm[m], :]))
                tauW_inf[inds, m] = samples[f"tauW{m + 1}"][inds]
                tauW = np.reshape(tauW_inf[inds, m], (nsamples, 1, 1))
                cW = np.reshape(cW_inf[c][:, m, :], (nsamples, 1, K))
                lmbW_inf[c][:, d : d + Dm[m], :] = np.sqrt(
                    cW**2 * lmbW_sqr / (cW**2 + tauW**2 * lmbW_sqr)
                )
                W_inf[c][:, d : d + Dm[m], :] = (
                    np.array(samples["W"][inds, d : d + Dm[m], :])
                    * lmbW_inf[c][:, d : d + Dm[m], :]
                    * tauW
                )
                d += Dm[m]
        elif args.model == "GFA":
            alpha_inf[c] = samples["alpha"][inds, :, :]
            Dm = hypers["Dm"]
            d = 0
            for m in range(args.num_sources):
                alpha = np.reshape(alpha_inf[c][:, m, :], (nsamples, 1, K))
                W_inf[c][:, d : d + Dm[m], :] = np.array(
                    samples["W"][inds, d : d + Dm[m], :]
                ) * (1 / np.sqrt(alpha))
                d += Dm[m]
        # Compute components in the data space
        for k in range(K):
            z = np.reshape(np.mean(Z_inf[c][:, :, k], axis=0), (N, 1))
            w = np.reshape(np.mean(W_inf[c][:, :, k], axis=0), (D, 1))
            d_comps[c][k] = np.dot(z, w.T)
        # update samples
        s += nsamples

    params = {"W": W_inf, "Z": Z_inf, "sigma": sigma_inf}
    if "sparseGFA" in args.model:
        params.update({"lmbW": lmbW_inf, "tauW": tauW_inf, "cW": cW_inf})
    elif args.model == "GFA":
        params.update({"alpha": alpha_inf})
    if args.model == "sparseGFA":
        if args.reghsZ:
            params.update({"lmbZ": lmbZ_inf, "tauZ": tauZ_inf, "cZ": cZ_inf})
        else:
            params.update({"lmbZ": lmbZ_inf, "tauZ": tauZ_inf})

    return params, d_comps


# === PARAMETER VALIDATION FUNCTIONS ===


def validate_core_parameters(args):
    """Validate core model parameters"""
    if args.K <= 0:
        raise ValueError(
            f"Invalid number of factors K={ args.K}. Must be positive integer (e.g., K=10)."
        )

    if args.K > 100:
        logging.warning(
            f"Large number of factors K={args.K} may lead to computational issues."
        )

    if not (1 <= args.percW <= 100):
        raise ValueError(
            f"Invalid sparsity percentage percW={args.percW}. Must be between 1-100."
        )

    if args.num_samples <= 0:
        raise ValueError(
            f"Invalid num_samples={args.num_samples}. Must be positive integer."
        )

    if args.num_samples < 500:
        logging.warning(
            f"Small num_samples={ args.num_samples} may lead to poor convergence. Consider ≥1000."
        )

    if args.num_warmup <= 0:
        raise ValueError(
            f"Invalid num_warmup={args.num_warmup}. Must be positive integer."
        )

    if args.num_chains <= 0:
        raise ValueError(
            f"Invalid num_chains={args.num_chains}. Must be positive integer."
        )

    if args.num_runs <= 0:
        raise ValueError(f"Invalid num_runs={args.num_runs}. Must be positive integer.")


def validate_model_parameters(args):
    """Validate model type and dataset"""
    if args.model not in ["sparseGFA", "GFA"]:
        raise ValueError(f"Invalid model='{args.model}'. Must be 'sparseGFA' or 'GFA'.")

    if args.dataset not in ["qmap_pd", "synthetic"]:
        raise ValueError(
            f"Invalid dataset='{args.dataset}'. Must be 'qmap_pd' or 'synthetic'."
        )


def validate_cv_parameters(args, cv_available=False, neuroimaging_cv_available=False):
    """Validate cross-validation parameters"""
    # Check for conflicting parameter combinations
    if getattr(args, "cv_only", False) and not cv_available:
        raise RuntimeError(
            "CV-only mode requested (--cv_only) but cross-validation module is not available. "
            "Please install cross-validation dependencies or run without --cv_only."
        )

    # Handle neuroimaging CV fallback
    if getattr(args, "neuroimaging_cv", False) and not neuroimaging_cv_available:
        logging.warning(
            "Neuroimaging CV requested but not available. Falling back to basic CV if available."
        )
        args.neuroimaging_cv = False
        if cv_available:
            args.run_cv = True

    # Auto-enable CV for nested CV
    if (
        getattr(args, "nested_cv", False)
        and not getattr(args, "run_cv", False)
        and not getattr(args, "cv_only", False)
    ):
        logging.warning(
            "--nested_cv specified but neither --run_cv nor --cv_only set. Enabling --run_cv."
        )
        args.run_cv = True

    # Validate CV fold count
    if hasattr(args, "cv_folds") and getattr(args, "cv_folds", 5) <= 1:
        raise ValueError(f"Invalid cv_folds={args.cv_folds}. Must be > 1.")

    # Validate CV type
    if hasattr(args, "cv_type"):
        valid_cv_types = ["standard", "stratified", "grouped", "repeated"]
        if getattr(args, "cv_type", "standard") not in valid_cv_types:
            raise ValueError(f"Invalid cv_type. Must be one of: {valid_cv_types}")

    # Check CV type compatibility with data
    if getattr(args, "cv_type", None) and args.dataset == "synthetic":
        if args.cv_type in ["clinical_stratified", "site_aware"]:
            logging.warning(
                f"CV type { args.cv_type} not suitable for synthetic data. Using 'standard'."
            )
            args.cv_type = "standard"


def validate_gpu_availability(args):
    """Validate GPU availability"""
    if args.device == "gpu":
        try:
            import jax

            if not jax.devices("gpu"):
                logging.warning("GPU requested but not available. Falling back to CPU.")
                args.device = "cpu"
        except Exception as e:
            logging.warning(f"JAX GPU support not available ({e}). Using CPU.")
            args.device = "cpu"
    elif args.device not in ["cpu", "gpu"]:
        raise ValueError(f"Invalid device='{args.device}'. Must be 'cpu' or 'gpu'.")


def validate_file_paths(args):
    """Validate file paths for qMAP-PD dataset"""
    if args.dataset == "qmap_pd":
        from pathlib import Path

        data_path = Path(args.data_dir)
        if not data_path.exists():
            raise FileNotFoundError(
                f"Data directory not found: {args.data_dir}. "
                f"Please ensure qMAP-PD data is available or use --dataset synthetic."
            )

        clinical_path = data_path / args.clinical_rel
        if not clinical_path.exists():
            raise FileNotFoundError(
                f"Clinical data file not found: {clinical_path}. "
                f"Please check the --clinical_rel parameter."
            )

        volumes_path = data_path / args.volumes_rel
        if not volumes_path.exists():
            raise FileNotFoundError(
                f"Volume matrices directory not found: {volumes_path}. "
                f"Please check the --volumes_rel parameter."
            )


def log_parameter_summary(args):
    """Log important parameter combinations"""
    logging.info("=== VALIDATED PARAMETERS ===")
    logging.info(f"Model: {args.model}, Dataset: {args.dataset}")
    logging.info(
        f"Factors K={args.K}, Sparsity={args.percW}%, Samples={args.num_samples}"
    )
    logging.info(
        f"Chains={args.num_chains}, Runs={args.num_runs}, Device={args.device}"
    )

    if getattr(args, "enable_preprocessing", False):
        logging.info(
            f"Preprocessing: { args.imputation_strategy} imputation, { args.feature_selection} selection"
        )
        if getattr(args, "enable_spatial_processing", False):
            logging.info("Spatial processing: ENABLED")

    if getattr(args, "run_cv", False) or getattr(args, "cv_only", False):
        cv_folds = getattr(args, "cv_folds", 5)
        cv_type = getattr(args, "cv_type", "standard")
        logging.info(f"Cross-validation: {cv_folds} folds, {cv_type} type")
        if getattr(args, "neuroimaging_cv", False):
            neuro_cv_type = getattr(args, "neuro_cv_type", "clinical_stratified")
            logging.info(f"Neuroimaging CV: {neuro_cv_type}")
        if getattr(args, "nested_cv", False):
            logging.info("Nested CV enabled for hyperparameter optimization")

    logging.info("=" * 30)


def validate_and_setup_args(
    args,
    cv_available=False,
    neuroimaging_cv_available=False,
    factor_mapping_available=False,
    preprocessing_available=False,
):
    """
    Main validation function with enhanced checks.
    Returns validated args or raises appropriate exceptions.
    """
    try:
        validate_core_parameters(args)
        validate_model_parameters(args)
        validate_cv_parameters(args, cv_available, neuroimaging_cv_available)
        validate_gpu_availability(args)
        validate_file_paths(args)

        # Validate neuroimaging parameters
        if getattr(args, "neuroimaging_cv", False):
            valid_neuro_cv_types = ["clinical_stratified", "site_aware", "standard"]
            if (
                getattr(args, "neuro_cv_type", "clinical_stratified")
                not in valid_neuro_cv_types
            ):
                raise ValueError(
                    f"Invalid neuro_cv_type. Must be one of: {valid_neuro_cv_types}"
                )

        # Validate factor mapping
        if getattr(args, "create_factor_maps", False) and not factor_mapping_available:
            logging.warning(
                "Factor mapping requested but neuroimaging_utils module not available. "
                "Factor maps will be skipped."
            )
            args.create_factor_maps = False

        # Validate preprocessing parameters
        if getattr(args, "enable_preprocessing", False) and preprocessing_available:
            try:
                from data.preprocessing import create_preprocessor_from_args

                create_preprocessor_from_args(args, validate=True)
            except Exception as e:
                raise ValueError(f"Invalid preprocessing parameters: {e}")
        elif getattr(args, "enable_preprocessing", False):
            logging.warning("Preprocessing module not available for validation")

        log_parameter_summary(args)

        return args

    except (ValueError, RuntimeError, FileNotFoundError) as e:
        raise e

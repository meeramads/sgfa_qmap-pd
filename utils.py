import numpy as np
import logging
import contextlib
import gc
from pathlib import Path


logging.basicConfig(level=logging.INFO)

def cleanup_memory():
    """
    Clean up JAX and matplotlib memory to prevent memory leaks.
    
    This function should be called periodically during long-running analyses
    or after completing major computational tasks.
    """
    import gc
    
    # Close all matplotlib figures
    try:
        import matplotlib.pyplot as plt
        plt.close('all')
        logging.debug("Closed all matplotlib figures")
    except ImportError:
        pass
    
    # Force garbage collection
    collected = gc.collect()
    if collected > 0:
        logging.debug(f"Garbage collected {collected} objects")
    
    # Clear JAX compilation cache if available and needed
    try:
        import jax
        # Note: JAX manages its own memory fairly well, but we can clear backends if needed
        # Only do this if memory issues are detected
        # jax.clear_backends()  # Uncomment only if memory issues persist
        logging.debug("JAX memory management completed")
    except ImportError:
        pass
    except Exception as e:
        logging.debug(f"JAX cleanup warning: {e}")

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
        plt.close('all')
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
    import psutil
    import os
    
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
            logging.info(f"Completed {operation_name} - Memory: {final_memory:.1f} MB (+{memory_diff:.1f} MB)")
        else:
            logging.debug(f"Completed {operation_name} - Memory: {final_memory:.1f} MB (+{memory_diff:.1f} MB)")
        
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
        
        logging.info(f"Memory status: {used_percent:.1f}% used, {available_gb:.1f}GB/{total_gb:.1f}GB available")
        
        if available_gb < 2:
            logging.warning(
                f"Low memory warning: Only {available_gb:.1f}GB available. "
                "Consider reducing batch sizes or using fewer chains."
            )
        elif available_gb < 4:
            logging.info(f"Memory note: {available_gb:.1f}GB available. Monitor memory usage during analysis.")
        
        return available_gb
        
    except ImportError:
        logging.debug("psutil not available for memory monitoring")
        return float('inf')  # Assume sufficient memory
    except Exception as e:
        logging.debug(f"Memory check failed: {e}")
        return float('inf')

def estimate_memory_requirements(n_subjects, n_features, n_factors, n_chains, n_samples):
    """
    Estimate memory requirements for the analysis.
    
    Parameters
    ----------
    n_subjects : int
        Number of subjects
    n_features : int
        Total number of features across all views
    n_factors : int
        Number of latent factors
    n_chains : int
        Number of MCMC chains
    n_samples : int
        Number of MCMC samples per chain
    
    Returns
    -------
    estimated_gb : float
        Estimated memory requirement in GB
    """
    
    # Rough estimates based on typical usage patterns
    # These are conservative estimates
    
    # Data storage: X matrices
    data_memory = n_subjects * n_features * 8  # float64
    
    # MCMC samples: W, Z, and other parameters
    W_memory = n_features * n_factors * n_chains * n_samples * 8
    Z_memory = n_subjects * n_factors * n_chains * n_samples * 8
    other_params_memory = (n_features + n_subjects) * n_factors * n_chains * n_samples * 8 * 0.5  # other parameters
    
    # Working memory (JAX compilation, intermediate calculations)
    working_memory = (data_memory + W_memory + Z_memory) * 2  # Conservative multiplier
    
    total_bytes = data_memory + W_memory + Z_memory + other_params_memory + working_memory
    estimated_gb = total_bytes / (1024**3)
    
    logging.info(f"Estimated memory requirements:")
    logging.info(f"  Data: {data_memory/(1024**3):.2f} GB")
    logging.info(f"  MCMC samples: {(W_memory + Z_memory + other_params_memory)/(1024**3):.2f} GB")
    logging.info(f"  Working memory: {working_memory/(1024**3):.2f} GB")
    logging.info(f"  Total estimated: {estimated_gb:.2f} GB")
    
    # Provide recommendations
    if estimated_gb > 16:
        logging.warning(
            f"High memory requirement ({estimated_gb:.1f} GB). Consider:"
            "\n  - Reducing number of samples (--num-samples)"
            "\n  - Reducing number of chains (--num-chains)"
            "\n  - Using feature selection to reduce features"
            "\n  - Running on a high-memory system"
        )
    elif estimated_gb > 8:
        logging.info(
            f"Moderate memory requirement ({estimated_gb:.1f} GB). "
            "Ensure sufficient RAM is available."
        )
    
    return estimated_gb

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
        'model_params': results_dir / f"[{run_id}]Model_params.dictionary",
        'robust_params': results_dir / f"[{run_id}]Robust_params.dictionary",
        'hyperparameters': results_dir / "hyperparameters.dictionary",
        'results_txt': results_dir / "results.txt",
        'plots_dir': results_dir / f"plots_{run_id}",
        'publication_dir': results_dir / f"plots_{run_id}" / "publication",
        'preprocessing_dir': results_dir / f"plots_{run_id}" / "preprocessing",
        'factor_maps_dir': results_dir / f"plots_{run_id}" / "factor_maps"
    }
    
    return files

def safe_pickle_load(filepath, description="File"):
    """
    Safely load pickle file with error handling.
    
    Parameters
    ----------
    filepath : str or Path
        Path to pickle file
    description : str
        Description for error messages
        
    Returns
    -------
    data : object
        Loaded data or None if failed
    """
    import pickle
    
    filepath = Path(filepath)
    
    if not filepath.exists():
        logging.error(f"{description} not found: {filepath}")
        return None
    
    if filepath.stat().st_size <= 5:
        logging.error(f"{description} is empty or corrupted: {filepath}")
        return None
    
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        logging.debug(f"Successfully loaded {description}: {filepath}")
        return data
    except Exception as e:
        logging.error(f"Failed to load {description} from {filepath}: {e}")
        return None

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
        with open(filepath, 'wb') as f:
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
    clean_name = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove multiple underscores
    clean_name = re.sub(r'_+', '_', clean_name)
    
    # Remove leading/trailing underscores and dots
    clean_name = clean_name.strip('_.')
    
    # Ensure not too long (max 255 chars for most filesystems)
    if len(clean_name) > 200:
        name, ext = os.path.splitext(clean_name)
        clean_name = name[:200-len(ext)] + ext
    
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
    logging.info("Running get_robustK")
    #Initialize parameters
    ncomps = args.K
    nchs = args.num_chains
    nsamples = args.num_samples
    t_nsamples = nsamples*nchs
    M, N, D = args.num_sources, params['Z'][0].shape[1], params['W'][0].shape[1]
    X_rob = [[] for _ in range(ncomps)]
    
    #horseshoe parameters (Z)
    Z = np.full([t_nsamples, N, ncomps], np.nan)
    if args.model == 'sparseGFA':     
        lmbZ = np.full([t_nsamples, N, ncomps], np.nan)
        tauZ = np.full([t_nsamples, ncomps], np.nan)
        if args.reghsZ: 
            cZ = np.full([t_nsamples, ncomps], np.nan)          
    
    #horseshoe parameters (W)
    W = np.full([t_nsamples, D, ncomps], np.nan)
    if 'sparseGFA' in args.model:
        lmbW = np.full([t_nsamples, D, ncomps], np.nan) 
        cW = np.full([t_nsamples, M, ncomps], np.nan)
    elif args.model == 'GFA':
        alpha = np.full([t_nsamples, M, ncomps], np.nan) 
    
    #Initialise parameters to select components
    storecomps = [np.arange(ncomps) for _ in range(nchs)]
    cosThr = thrs['cosineThr']; matchThr = thrs['matchThr']
    nrobcomp = 0
    for c1 in range(nchs):
        max_sim = np.zeros((ncomps,nchs))
        max_simW = np.zeros((ncomps,nchs))
        max_simZ = np.zeros((ncomps,nchs))
        matchInds = np.zeros((ncomps,nchs))
        for k in storecomps[c1]:
            nonempty_chs = []
            for ne in range(len(storecomps)):
                if storecomps[ne].size > 0:
                    nonempty_chs.append(ne)
            
            for c2 in nonempty_chs:
                cosine = np.zeros((1,storecomps[c2].size))
                cosW = np.zeros((1,storecomps[c2].size))
                cosZ = np.zeros((1,storecomps[c2].size))
                cind = 0 
                for comp in storecomps[c2]:
                    #X
                    comp1 = np.ndarray.flatten(d_comps[c2][comp])
                    comp2 = np.ndarray.flatten(d_comps[c1][k])
                    cosine[0, cind] = np.dot(comp1, comp2)/(np.linalg.norm(comp1)*np.linalg.norm(comp2))
                    #W
                    compW1 = np.mean(params['W'][c2],axis=0)[:,comp]
                    compW2 = np.mean(params['W'][c1],axis=0)[:,k]
                    cosW[0, cind] = np.dot(compW1, compW2)/(np.linalg.norm(compW1)*np.linalg.norm(compW2))
                    #Z
                    compZ1 = np.mean(params['Z'][c2],axis=0)[:,comp]
                    compZ2 = np.mean(params['Z'][c1],axis=0)[:,k]
                    cosZ[0, cind] = np.dot(compZ1, compZ2)/(np.linalg.norm(compZ1)*np.linalg.norm(compZ2))
                    cind += 1
                #find the most similar components           
                max_sim[k,c2] = cosine[0, np.argmax(cosine)]
                max_simW[k,c2] = cosW[0, np.argmax(cosine)]
                max_simZ[k,c2] = cosZ[0, np.argmax(cosine)]
                matchInds[k,c2] = storecomps[c2][np.argmax(cosine)]
                if max_sim[k,c2] < cosThr: 
                    matchInds[k,c2] = -1
            
            if np.sum(max_sim[k,:] > cosThr) > matchThr * nchs:
                goodInds = np.where(matchInds[k,:] >= 0)
                X_rob[k] = np.zeros((N,D))
                s = 0
                for c2 in list(goodInds[0]):
                    inds = np.arange(s,s+nsamples)
                    #components in the data space
                    X_rob[k] += d_comps[c2][int(matchInds[k,c2])]
                    #parameters
                    if args.model == 'sparseGFA':
                        lmbZ[inds,:,k] = params['lmbZ'][c2][:,:,int(matchInds[k,c2])]
                        tauZ[inds,k] = params['tauZ'][c2][:,int(matchInds[k,c2])]
                        if args.reghsZ:
                            cZ[inds,k] = params['cZ'][c2][:,int(matchInds[k,c2])]                 
                    if 'sparseGFA' in args.model:
                        lmbW[inds,:,k] = params['lmbW'][c2][:,:,int(matchInds[k,c2])]
                        cW[inds,:,k] = params['cW'][c2][:,:, int(matchInds[k,c2])]
                    elif args.model == 'GFA':
                        alpha[inds,:,k] = params['alpha'][c2][:,:, int(matchInds[k,c2])]
                    #W
                    if max_simW[k,c2] > 0: 
                        W[inds,:,k] = params['W'][c2][:, :, int(matchInds[k,c2])]
                    else:
                        W[inds,:,k] = -params['W'][c2][:, :, int(matchInds[k,c2])]
                    #Z
                    if max_simZ[k,c2] > 0: 
                        Z[inds,:,k] = params['Z'][c2][:, :, int(matchInds[k,c2])]
                    else:
                        Z[inds,:,k] = -params['Z'][c2][:, :, int(matchInds[k,c2])]
                    #remove robust components from storecomps
                    storecomps[c2] = storecomps[c2][storecomps[c2] != int(matchInds[k,c2])]
                    #update samples
                    s += nsamples
                #divided by total number of robust components    
                X_rob[k] = [X_rob[k]/np.sum(matchInds[k,:]>=0)]
                nrobcomp += 1                            
    success = True
    if nrobcomp > 0:
        #Remove non-robust components and discard chains
        idx_cols = ~np.isnan(np.mean(Z,axis=1)).all(axis=0)
        idx_rows = ~np.isnan(np.mean(Z,axis=1)[:,idx_cols]).any(axis=1)
        if args.model == 'sparseGFA':
            lmbZ = lmbZ[idx_rows,:,:]; lmbZ = lmbZ[:,:,idx_cols]
            tauZ = tauZ[idx_rows,:]; tauZ = tauZ[:,idx_cols]
            if args.reghsZ:
                cZ = cZ[idx_rows,:]; cZ = cZ[:,idx_cols]
                if cZ.size == 0:
                    logging.warning('No samples survived!')
                    success = False
        W = W[idx_rows,:,:]; W = W[:,:,idx_cols]
        Z = Z[idx_rows,:,:]; Z = Z[:,:,idx_cols]
        if 'sparseGFA' in args.model:
            lmbW = lmbW[idx_rows,:,:]; lmbW = lmbW[:,:,idx_cols]
            cW = cW[idx_rows,:,:]; cW = cW[:,:,idx_cols]
        elif args.model == 'GFA':
            alpha = alpha[idx_rows,:,:]; alpha = alpha[:,:,idx_cols]
        X_rob_final = [X_rob[i] for i in range(len(X_rob)) if X_rob[i] != []]
        X_rob = X_rob_final 
    else: 
        logging.warning('No robust components found!')
        success = False            
    
    #create dictionary with robust params (save the posterior mean for most parameters)
    rob_params = {'W': np.mean(W, axis=0), 'Z': np.mean(Z, axis=0)}
    if 'sparseGFA' in args.model:
        rob_params.update({'cW_inf': np.mean(cW, axis=0), 'lmbW': np.mean(lmbW, axis=0)})
    elif args.model == 'GFA':
        rob_params.update({'alpha_inf': np.mean(alpha, axis=0)})
    if args.model == 'sparseGFA':
        if args.reghsZ:
            rob_params.update({'cZ_inf': cZ, 'tauZ_inf': tauZ, 'lmbZ': np.mean(lmbZ, axis=0)})
        else:
            rob_params.update({'tauZ_inf': tauZ, 'lmbZ': np.mean(lmbZ, axis=0)})
    return rob_params, X_rob, success

def get_infparams(samples, hypers, args):

    #Get inferred parameters
    nchs, nsamples= args.num_chains, args.num_samples
    N = samples['Z'].shape[1]
    K = args.K
    #Get Z
    Z_inf = [np.zeros((nsamples, N, K)) for _ in range(nchs)]
    if args.model == 'sparseGFA':
        lmbZ_inf = [np.zeros((nsamples, N, K)) for _ in range(nchs)]
        tauZ_inf = [np.zeros((nsamples, K)) for _ in range(nchs)]
        if args.reghsZ: 
            cZ_inf = [np.zeros((nsamples, K)) for _ in range(nchs)]
    #Get W
    D = sum(hypers['Dm'])
    W_inf = [np.zeros((nsamples, D, K)) for _ in range(nchs)]
    if 'sparseGFA' in args.model: 
        lmbW_inf = [np.zeros((nsamples, D, K)) for _ in range(nchs)]
        cW_inf = [np.zeros((nsamples, K)) for _ in range(nchs)]
        tauW_inf = np.zeros((nchs*nsamples, args.num_sources))
    elif args.model == 'GFA':
        alpha_inf = [np.zeros((nsamples, K)) for _ in range(nchs)]
    sigma_inf = np.zeros((nchs*nsamples, args.num_sources))    
    d_comps = [[[] for _ in range(K)] for _ in range(nchs)] 
    s = 0
    for c in range(nchs):   
        inds = np.arange(s,s+nsamples)
        # get sigma
        sigma_inf[inds,:] = samples['sigma'][inds,0,:]
        # get Z, tauZ, lmbZ, cZ
        for k in range(K):
            if args.model == 'sparseGFA':
                tauZ_inf[c][:,k] = np.array(samples[f'tauZ'])[inds,0,k]
                tauZ = np.reshape(tauZ_inf[c][:,k],(nsamples,1))
                if args.reghsZ:
                    cZ_inf[c][:,k] = hypers['slab_scale'] * np.sqrt(samples['cZ'][inds,0,k])
                    cZ = np.reshape(cZ_inf[c][:,k],(nsamples,1))
                    lmbZ_sqr = np.square(np.array(samples['lmbZ'])[inds,:,k])
                    lmbZ_inf[c][:,:,k] = np.sqrt(lmbZ_sqr * cZ ** 2 /(cZ ** 2 + tauZ ** 2 * lmbZ_sqr))
                else: 
                    lmbZ_inf[c][:,:,k] = np.array(samples['lmbZ'])[inds,:,k]
                Z_inf[c][:,:,k] = np.array(samples['Z'])[inds,:,k] * lmbZ_inf[c][:,:,k] * tauZ 
            else:
                Z_inf[c][:,:,k] = np.array(samples['Z'])[inds,:,k]
        
        # get W, tauW, lmbW, cW
        if 'sparseGFA' in args.model: 
            cW_inf[c] = hypers['slab_scale'] * np.sqrt(samples['cW'][inds,:,:]) 
            Dm = hypers['Dm']; d = 0
            for m in range(args.num_sources):      
                lmbW_sqr = np.square(np.array(samples['lmbW'][inds,d:d+Dm[m],:]))
                tauW_inf[inds,m] = samples[f'tauW{m+1}'][inds]
                tauW = np.reshape(tauW_inf[inds,m], (nsamples,1,1))
                cW = np.reshape(cW_inf[c][:,m,:], (nsamples,1, K))
                lmbW_inf[c][:,d:d+Dm[m],:] = np.sqrt(cW ** 2 * lmbW_sqr /
                    (cW ** 2 + tauW ** 2 * lmbW_sqr))
                W_inf[c][:,d:d+Dm[m],:] = np.array(samples['W'][inds,d:d+Dm[m],:]) * \
                    lmbW_inf[c][:,d:d+Dm[m],:] * tauW
                d += Dm[m]
        elif args.model == 'GFA':
            alpha_inf[c] = samples['alpha'][inds,:,:]
            Dm = hypers['Dm']; d = 0
            for m in range(args.num_sources): 
                alpha = np.reshape(alpha_inf[c][:,m,:], (nsamples,1, K))
                W_inf[c][:,d:d+Dm[m],:] = np.array(samples['W'][inds,d:d+Dm[m],:]) * (1/np.sqrt(alpha)) 
                d += Dm[m]
        # Compute components in the data space
        for k in range(K):
            z = np.reshape(np.mean(Z_inf[c][:,:,k],axis=0), (N, 1))
            w = np.reshape(np.mean(W_inf[c][:,:,k],axis=0), (D, 1))
            d_comps[c][k] = np.dot(z, w.T)
        # update samples
        s += nsamples

    params = {'W': W_inf, 'Z': Z_inf, 'sigma': sigma_inf}
    if 'sparseGFA' in args.model:
        params.update({'lmbW': lmbW_inf, 'tauW': tauW_inf, 'cW' : cW_inf})
    elif args.model == 'GFA':
        params.update({'alpha': alpha_inf})
    if args.model == 'sparseGFA':
        if args.reghsZ: 
            params.update({'lmbZ': lmbZ_inf, 'tauZ': tauZ_inf, 'cZ': cZ_inf})    
        else:
            params.update({'lmbZ': lmbZ_inf, 'tauZ': tauZ_inf})                        

    return params, d_comps
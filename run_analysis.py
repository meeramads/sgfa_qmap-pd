import argparse
import os
import time
import pickle
import sys
#numpy
import numpy as np
#jax
import jax
import jax.numpy as jnp
import jax.random 
#numpyro
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
#sklearn
from sklearn.preprocessing import StandardScaler
#generate/ load data
import get_data
#visualization module
import visualization
#logging
import logging
from datetime import datetime

from utils import get_infparams, get_robustK


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logging.info('Starting run_analysis.py')

def models(X, hypers, args):
    logging.debug(f"Running models with X shape: {X.shape}, hypers: {hypers}, args: {args}")

    N, M, Dm = X.shape[0], args.num_sources, hypers['Dm'] 
    D, K, percW = sum(Dm), args.K, hypers['percW']

    # Sample sigma
    sigma = numpyro.sample("sigma", dist.Gamma(hypers['a_sigma'], 
        hypers['b_sigma']),sample_shape=(1,M)) 
    
    if args.model == 'sparseGFA':
        
        # Sample Z   
        Z = numpyro.sample("Z",dist.Normal(0,1), sample_shape=(N,K))
        # Sample tau Z
        tauZ =  numpyro.sample(f'tauZ', dist.TruncatedCauchy(scale=1), sample_shape=(1,K)) 
        # Sample lambda Z
        lmbZ =  numpyro.sample("lmbZ", dist.TruncatedCauchy(scale=1), sample_shape=(N,K))
        if args.reghsZ:   
            # Sample cZ    
            cZtmp = numpyro.sample("cZ", dist.InverseGamma(0.5 * hypers['slab_df'], 
                0.5 * hypers['slab_df']), sample_shape=(1,K))
            cZ = hypers['slab_scale'] * jnp.sqrt(cZtmp)
            
            # Get regularised Z
            lmbZ_sqr = jnp.square(lmbZ)
            for k in range(K):    
                lmbZ_tilde = jnp.sqrt(lmbZ_sqr[:,k] * cZ[0,k] ** 2 / \
                    (cZ[0,k] ** 2 + tauZ[0,k] ** 2 * lmbZ_sqr[:,k]))
                Z = Z.at[:,k].set(Z[:,k] * lmbZ_tilde * tauZ[0,k])
        else:
            Z = Z * lmbZ * tauZ
    else:
        # Sample Z
        Z = numpyro.sample("Z",dist.Normal(0,1), sample_shape=(N,K))
    
    #sample W
    W = numpyro.sample("W",dist.Normal(0,1), sample_shape=(D,K))
    if 'sparseGFA' in args.model:
        # Implement regularised horseshoe prior over W
        #sample lambda W 
        lmbW =  numpyro.sample("lmbW", dist.TruncatedCauchy(scale=1), sample_shape=(D,K))
        #sample cW
        cWtmp = numpyro.sample("cW", dist.InverseGamma(0.5 * hypers['slab_df'], 
            0.5 * hypers['slab_df']), sample_shape=(M,K))
        cW = hypers['slab_scale'] * jnp.sqrt(cWtmp)
        pW = np.round((percW/100) * Dm)
        d = 0
        for m in range(M): 
            scaleW = pW[m] / ((Dm[m] - pW[m]) * jnp.sqrt(N)) 
            #sample tau W
            tauW =  numpyro.sample(f'tauW{m+1}', 
                dist.TruncatedCauchy(scale=scaleW * 1/jnp.sqrt(sigma[0,m])))     
            lmbW_sqr = jnp.reshape(jnp.square(lmbW[d:d+Dm[m],:]), (Dm[m],K))
            lmbW_tilde = jnp.sqrt(cW[m,:] ** 2 * lmbW_sqr / \
                (cW[m,:] ** 2 + tauW ** 2 * lmbW_sqr))
            W = W.at[d:d+Dm[m],:].set(W[d:d+Dm[m],:] * lmbW_tilde * tauW)
            #sample X
            numpyro.sample(f'X{m+1}', dist.Normal(jnp.dot(Z,W[d:d+Dm[m],:].T), 
                1/jnp.sqrt(sigma[0,m])), obs=X[:,d:d+Dm[m]])
            d += Dm[m]
    elif args.model == 'GFA':
        # Implement ARD prior over W
        alpha = numpyro.sample("alpha", dist.Gamma(1e-3, 1e-3), sample_shape=(M,K))
        d = 0
        for m in range(M): 
            W = W.at[d:d+Dm[m],:].set(W[d:d+Dm[m],:]* (1/jnp.sqrt(alpha[m,:])))
            #sample X
            numpyro.sample(f'X{m+1}', dist.Normal(jnp.dot(Z,W[d:d+Dm[m],:].T), 
                1/jnp.sqrt(sigma[0,m])), obs=X[:,d:d+Dm[m]])
            d += Dm[m]

def run_inference(model, args, rng_key, X, hypers):
    
    # Run inference using Hamiltonian Monte Carlo
    kernel = NUTS(model)
    mcmc = MCMC(kernel, num_warmup=args.num_warmup, num_samples=args.num_samples, 
        num_chains=args.num_chains)
    mcmc.run(rng_key, X, hypers, args, extra_fields=('potential_energy',))    
    #mcmc.print_summary() 
    return mcmc

def main(args):                           
    #Make directory to save results
    if 'synthetic' in args.dataset: 
        flag = f'K{args.K}_{args.num_chains}chs_pW{args.percW}_s{args.num_samples}_addNoise{args.noise}'
    else:
        flag = f'K{args.K}_{args.num_chains}chs_pW{args.percW}_s{args.num_samples}'
    
    if args.model == 'sparseGFA':
        if args.reghsZ:
            flag_regZ = '_reghsZ'
        else:
            flag_regZ = '_hsZ'
    else:
        flag_regZ = ''
    
    res_dir = f'../results/{args.dataset}/{args.model}_{flag}{flag_regZ}'     
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)  

    # set up parameters of the priors 
    hp_path = f'{res_dir}/hyperparameters.dictionary'
    if not os.path.exists(hp_path):   
        hypers = {'a_sigma': 1, 'b_sigma': 1,
                'nu_local': 1, 'nu_global': 1,
                'slab_scale': 2, 'slab_df': 4, 
                'percW': args.percW}
        with open(hp_path, 'wb') as parameters:
            pickle.dump(hypers, parameters)        
    else:
        with open(hp_path, 'rb') as parameters:
            hypers = pickle.load(parameters)       

    for i in range(args.num_runs):          
        logging.info(f'Initialisation: {i+1}')
        logging.info('----------------------------------')
        
        if 'synthetic' in args.dataset:  
            # Generate synthetic data 
            data_path = f'{res_dir}/[{i+1}]Data.dictionary'
            if not os.path.exists(data_path):
                data = get_data.synthetic_data(hypers, args)
                with open(data_path, 'wb') as parameters:
                    pickle.dump(data, parameters)
            else:
                with open(data_path, 'rb') as parameters:
                    data = pickle.load(parameters)
            X = data['X'] 
        elif 'genfi' in args.dataset:
            datafolder = f'../data/GENFI'
            data = get_data.genfi(datafolder)
            X = data['X']
            Y = data['Y'] 
            #standardise matrices
            X = StandardScaler().fit_transform(X)
        hypers.update({'Dm': data.get('Dm')})                                 
                                                      
        # RUN MODEL
        res_path = f'{res_dir}/[{i+1}]Model_params.dictionary'
        robparams_path = f'{res_dir}/[{i+1}]Robust_params.dictionary'
        if not os.path.exists(res_path):
            
            # Create an empty file 
            with open(res_path, 'wb') as parameters:
                pickle.dump(0, parameters)
            
            logging.info('Running Model...') 
            seed = np.random.randint(0,50)
            rng_key = jax.random.PRNGKey(seed)
            start = time.time()
            MCMCout = run_inference(models, args, rng_key, X, hypers)
            mcmc_samples = MCMCout.get_samples()
            # Compute sampling performance 
            mcmc_samples.update({'time_elapsed': (time.time() - start)/60})
            pe = MCMCout.get_extra_fields()['potential_energy']
            mcmc_samples.update({'exp_logdensity': jnp.mean(-pe)})
            # Save model
            with open(res_path, 'wb') as parameters:
                pickle.dump(mcmc_samples, parameters)
                logging.info('Inferred parameters saved.')
            
        if not os.path.exists(robparams_path) and os.stat(res_path).st_size > 5:    
            
            #Get inferred parameters within each chain
            with open(res_path, 'rb') as parameters:
                mcmc_samples = pickle.load(parameters)
            inf_params, data_comps = get_infparams(mcmc_samples, hypers, args)

            if args.num_chains > 1:
                #Find robust components
                thrs = {'cosineThr': 0.8, 'matchThr': 0.5}
                rob_params, X_rob, success = get_robustK(thrs, args, inf_params, data_comps)

                #Save robust data components
                if success:
                    rob_params.update({'sigma_inf': inf_params['sigma'], 'infX': X_rob})
                    if 'sparseGFA' in args.model:
                        rob_params.update({'tauW_inf': inf_params['tauW']}) 
                    with open(robparams_path, 'wb') as parameters:
                        pickle.dump(rob_params, parameters)
                        logging.info('Robust parameters saved')  
                else:
                    logging.warning('No robust components found') 
            else:
                W = np.mean(inf_params['W'][0],axis=0)
                Z = np.mean(inf_params['Z'][0],axis=0)
                X = [[np.dot(Z,W.T)]]
                rob_params = {'W': W, 'Z': Z, 'infX': X} 
                with open(robparams_path, 'wb') as parameters:
                    pickle.dump(rob_params, parameters)     

    # visualization
    if 'synthetic' in args.dataset:
        visualization.synthetic_data(res_dir, data, args, hypers)
    else:
        visualization.genfi(data, res_dir, args)

if __name__ == "__main__":

    # Define arguments to run analysis
    dataset = 'genfi'
    if 'genfi' in dataset:
        num_samples = 5000
        K = 20
        num_sources = 2
        num_runs = 10
    else:
        num_samples = 1500
        K = 5
        num_sources = 3
        num_runs = 5

    parser = argparse.ArgumentParser(description=" Sparse GFA with reg. horseshoe priors")
    parser.add_argument("--model", nargs="?", default='sparseGFA', type=str, 
                        help='add horseshoe prior over the latent variables')
    parser.add_argument("--num-samples", nargs="?", default=num_samples, type=int, 
                        help='number of MCMC samples')
    parser.add_argument("--num-warmup", nargs='?', default=1000, type=int, 
                        help='number of MCMC samples for warmup')
    parser.add_argument("--K", nargs='?', default=K, type=int, 
                        help='number of components')
    parser.add_argument("--num-chains", nargs='?', default=4, type=int,
                        help= 'number of MCMC chains')
    parser.add_argument("--num-sources", nargs='?', default=num_sources, type=int, 
                        help='number of data sources')
    parser.add_argument("--num-runs", nargs='?', default=num_runs, type=int, 
                        help='number of runs')
    parser.add_argument("--reghsZ", nargs='?', default=True, type=bool)
    parser.add_argument("--percW", nargs='?', default=33, type=int, 
                        help='percentage of relevant variables in each source')
    parser.add_argument("--dataset", nargs='?', default=dataset, type=str, 
                        help='choose dataset')
    parser.add_argument("--device", default='cpu', type=str, 
                        help='use "cpu" or "gpu".')
    parser.add_argument("--noise", nargs='?', default=0, type=int, 
                    help='Add noise to synthetic data (1=yes, 0=no)')
    parser.add_argument("--seed", nargs='?', default=None, type=int, 
                    help='Random seed for reproducibility (int). If not set, a random seed is used.')
    args = parser.parse_args()
    
    # Set the seed for reproducibility
    if args.seed is not None:
        seed = args.seed
    else:
        seed = np.random.randint(0, 100000)
    np.random.seed(seed)
    
    numpyro.set_platform(args.device)
    numpyro.set_host_device_count(args.num_chains)
    main(args)
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

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
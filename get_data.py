import numpy as np
import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
from scipy.stats import invgamma
import logging

logging.basicConfig(level=logging.INFO)

def synthetic_data(hypers, args):
    logging.info("Generating synthetic data")

    M = args.num_sources #number of data sources
    N = 150 #number of samples/examples 
    Dm = np.array([60, 40, 20]) #number of features/variables in each data source
    D = sum(Dm)
    K_true = 3 #number of latent components to generate the data

    # Implement regularized horseshoe prior over Z
    Z = np.reshape(np.random.normal(0, 1, N * K_true), (N,K_true))
    
    #lambda Z
    lmbZ0 = 0.001
    lmbZ = 200 * np.ones((N,K_true))
    lmbZ[50:,0] = lmbZ0
    lmbZ[0:50,1] = lmbZ0; lmbZ[100:150,1] = lmbZ0
    
    #tau Z
    tauZ = 0.01
    for k in range(K_true):
        Z[:,k] = Z[:,k] * lmbZ[:,k] * tauZ
    
    #sigmas
    sigma = np.array([3, 6, 4])
    logging.debug(f"Z shape: {Z.shape}, sigma: {sigma}")
    
    # Implement regularized horsesho prior over W
    #lambda W
    percW = 33 * np.ones((1,K_true))
    pW = np.round((percW/100) * Dm)
    lmbW = np.zeros((D,K_true)) * 0.01
    lmbW0 = 100; d = 0
    for m in range(M): 
        for k in range(K_true):
            lmbW[np.random.choice(
                    np.arange(Dm[m]), int(pW[0,m]), replace=False) + d, k] = lmbW0
        d += Dm[m]
    #tau W        
    tauW = np.zeros((1,M))
    for m in range(M):
        scaleW = pW[0,m] / ((Dm[m] - pW[0,m]) * np.sqrt(N))
        tauW[0,m] =  scaleW * 1/np.sqrt(sigma[m])   
    #c W   
    cW = np.reshape(invgamma.rvs(0.5 * hypers['slab_df'],
        scale=0.5 * hypers['slab_df'], size=M*K_true),(M,K_true))
    cW = hypers['slab_scale'] * np.sqrt(cW)    
    W = np.random.normal(0, 1, (D,K_true))
    X = np.zeros((N,D)); d = 0
    for m in range(M): 
        lmbW_sqr = np.reshape(np.square(lmbW[d:d+Dm[m],:]), (Dm[m],K_true))
        lmbW[d:d+Dm[m],:] = np.sqrt(cW[m,:] ** 2 * lmbW_sqr / 
                (cW[m,:] ** 2 + tauW[0,m] ** 2 * lmbW_sqr))
        W[d:d+Dm[m],:] = W[d:d+Dm[m],:] * lmbW[d:d+Dm[m],:] * tauW[0,m]
        
        # Generate X^(m)
        X[:,d:d+Dm[m]] = np.dot(Z,W[d:d+Dm[m],:].T) + \
            np.reshape(np.random.normal(0, 1/np.sqrt(sigma[m]), N*Dm[m]),(N,Dm[m])) 
        d += Dm[m]

    #Save parameters
    data = {'X': X, 'Z': Z, 'tauZ': tauZ, 'lmbZ': lmbZ, 'sigma': sigma,
        'W': W, 'tauW': tauW, 'lmbW': lmbW, 'cW': cW, 'K_true': K_true, 'Dm': Dm}       

    return data

def genfi(data_dir):

    arrays_path = f'{data_dir}/arrays.dictionary'
    if not os.path.exists(arrays_path):
        
        #load data
        df_data = pd.read_excel(f'{data_dir}/DATA_GENFI_PROJECT23.xlsx', engine='openpyxl')
        #Choose patients from visit=11
        df_data_red = df_data[(df_data['Visit'] == 11)]
        #remove TBK1 subtype
        df_data_red = df_data_red[(df_data_red['Genetic Group'] != 'TBK1')]
        #remove controls
        df_data_red = df_data_red[(df_data_red['Genetic status 2'] != 0)]
        #select symptomatic individuals only
        df_data_red = df_data_red[(df_data_red['Genetic status 1'] == 'A')]  
        #IDs of subjects
        ids = df_data_red['Blinded Code']

        #Brain data
        df_brain_vol = df_data_red.iloc[:, 71:106].copy() #get all GM volumes
        df_brain_vol = df_brain_vol.drop(columns=['Pons','Brain Stem','Total_Brain',
                        'Frontal lobe volume','Temporal lobe volume','Parietal lobe volume',
                        'Occipital lobe volume', 'Cingulate volume','Insula volume'])                
        df_brain_vol['Cerebellum'] = df_data_red['Total Cerebellum']
        #select rows with nans to remove
        brain_mat = df_brain_vol.to_numpy()
        keep_rows = np.arange(brain_mat.shape[0])
        ids_nans = np.any(np.isnan(brain_mat),axis=1)
        keep_rows = keep_rows[~ids_nans] 
        
        #Questionnaires
        df_clinical = df_data_red.iloc[:, 21:65]
        df_clinical = df_clinical.drop(columns=['FTLD-CDR-GLOBAL', 'BEHAV-SOB (Total Behaviour)', 
                                    'Blinded Code.2', 'Blinded Site.2', 'Visit.2'])
        #select rows with nans to remove
        num_nans = df_clinical.isnull().sum(axis=1).to_numpy()
        thr = df_clinical.shape[1]/3 # 33%
        for r in keep_rows:
            if num_nans[r] > thr:
                keep_rows = keep_rows[keep_rows != r]
        
        #confounds
        df_conf = df_data_red[['Age at visit (Years)','Gender (0=F, 1=M)','Education (Years)','TIV mm3']]
        C = df_conf.to_numpy()
        C = C[keep_rows, :]   

        #Clean dfs and save them        
        #brain clean
        df_brain_vol_clean = df_brain_vol.iloc[keep_rows,:]
        brain_mat = brain_mat[keep_rows, :]
        #compute asymmetry
        b_cols = list(df_brain_vol_clean)
        TV_right = np.zeros((keep_rows.size,1)) 
        TV_left = np.zeros((keep_rows.size,1)) 
        for i in range(len(b_cols)):
            if 'Right' in b_cols[i]:
                TV_right[:,0] += brain_mat[:,i]
            elif 'Left' in b_cols[i]:
                TV_left[:,0] += brain_mat[:,i]
        asymmetry = np.log(np.abs(TV_left-TV_right)/(TV_left+TV_right))       
        brain_mat = np.c_[brain_mat, asymmetry]
        df_brain_vol_clean['Asymmetry'] = pd.Series(asymmetry[:,0], index=df_brain_vol_clean.index)       
        df_brain_vol_clean.to_csv(f'{data_dir}/visit11_brainvols_{keep_rows.size}subjs.csv')
        
        #clinical clean
        df_clinical_clean = df_clinical.iloc[keep_rows,:]
        clinical_mat = df_clinical_clean.to_numpy()
        perc_nans = np.zeros((1,len(list(df_clinical_clean))))
        cols = list(df_clinical_clean)
        rm_cols = []; pmiss=20
        for j in range(len(cols)):
            perc_nans[0,j] = (df_clinical_clean[cols[j]].isnull().sum()/keep_rows.size) * 100
            id_na = pd.isna(df_clinical_clean[cols[j]]) * 1
            ids = np.arange(keep_rows.size)
            clinical_mat[ids[id_na==1],j] = np.nanmedian(clinical_mat[:,j])
            if perc_nans[0,j] > pmiss:
                rm_cols.append(cols[j])
        
        #remove cols with more than 10% of missing values
        clinical_mat = clinical_mat[:, perc_nans[0,:] < pmiss]
        df_clinical_clean = df_clinical_clean.drop(columns= rm_cols)
        df_clinical_clean.to_csv(f'{data_dir}/visit11_clinicaldata_{keep_rows.size}subjs.csv')

        #histograms behavioural variables
        distfolder = f'{data_dir}/vars_dists/'
        if not os.path.exists(distfolder):
            os.makedirs(distfolder) 
        for col in list(df_clinical_clean):
            plt.figure()
            plt.hist(df_clinical_clean[col], bins=30)
            plt.title(col)
            plt.savefig(f'{distfolder}{col}.png'); plt.close()
        
        #confounds clean
        df_conf_clean = df_conf.iloc[keep_rows,:]
        df_conf_clean.to_csv(f'{data_dir}/visit11_confounds_{keep_rows.size}subjs.csv')

        #save clean data
        df_data_clean = df_data_red.iloc[keep_rows,:]
        df_data_clean.to_csv(f'{data_dir}/visit11_data_{keep_rows.size}subjs.csv')
        #get var labels
        labels_file = f'{data_dir}/var_labels.csv'
        if not os.path.exists(arrays_path):
            labels = list(df_brain_vol_clean.columns) + list(df_clinical_clean.columns)
            df_vars = pd.DataFrame(columns=['labels'])
            df_vars['labels'] = labels
            df_vars.to_csv(labels_file)

        #create a dict with brain, clinical data and C arrays
        arrays = {'X1': brain_mat, 'X2': clinical_mat, 'C': C, 'nans_cols_cli': perc_nans}

        with open(arrays_path, 'wb') as parameters:
            pickle.dump(arrays, parameters)
            logging.info('Model saved')
    else:
        with open(arrays_path, 'rb') as parameters:
            arrays = pickle.load(parameters) 
    
    N = arrays.get('X1').shape[0]    
    df_data = pd.read_csv(f'{data_dir}/visit11_data_{N}subjs.csv')
    
    #Deconfound brain and clinical data separately
    X1 = arrays['X1'] - np.dot(arrays['C'], np.dot(np.linalg.pinv(arrays['C']), arrays['X1']))
    X2 = arrays['X2'] - np.dot(arrays['C'], np.dot(np.linalg.pinv(arrays['C']), arrays['X2']))
    Dm = np.array([X1.shape[1], X2.shape[1]])
    X = np.concatenate((X1, X2), axis=1) 

    #Subtypes
    ids = df_data['Blinded Code']
    Y = np.zeros((N, 1))
    labels = []
    for i in range(N):
        if 'C9ORF' in ids[i]:
            Y[i,0] = 1
            labels.append('C9orf72')
        elif 'GRN' in ids[i]:
            Y[i,0] = 2
            labels.append('GRN')
        elif 'MAPT' in ids[i]:
            Y[i,0] = 3
            labels.append('MAPT')
    
    # Store data
    data = {'X': X, 'Y': Y, 'Dm': Dm, 'labels': labels}       
    return data

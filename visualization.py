import pickle
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import gcf
import seaborn as sns
import numpy as np
#sklearn
from sklearn.preprocessing import StandardScaler
from scipy.stats import f_oneway, ttest_ind
from sklearn.metrics.pairwise import cosine_similarity

def synthetic_data(res_dir, true_params, args, hypers):
    
    ofile = open(f'{res_dir}/results.txt','w')
    
    #Find best initialisation
    exp_logs, ofile = find_bestrun(res_dir, args, ofile)
    brun = np.nanargmax(exp_logs)+1
    print('Best run: ', brun, file=ofile) 

    # plot dir
    plot_path = f'{res_dir}/plots_{brun}'
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)
        os.makedirs(f'{plot_path}/svgs')   
            
    # Plot generated training data
    pathX = f'{plot_path}/trueX'
    plot_X(true_params, args, hypers, pathX, true_data=True)   
    
    #Plot true parameters
    params_paths = {
        'W': f'{plot_path}/trueW',
        'Z': f'{plot_path}/trueZ',
        'Z_svg': f'{plot_path}/svgs/trueZ',
        'lmbW': f'{plot_path}/truelmbW',
        'lmbZ': f'{plot_path}/truelmbZ'} 
    plot_param(true_params, params_paths, args)

    rparams_path = f'{res_dir}/[{brun}]Robust_params.dictionary'
    if os.stat(rparams_path).st_size > 5:       
        
        with open(rparams_path, 'rb') as parameters:
            rob_params = pickle.load(parameters) 
                
        #Plot inferred X
        X_rob = rob_params['infX']
        pathX = f'{plot_path}/infX'
        plot_X(X_rob, args, hypers, pathX)

        # Calculate scores per true component
        Z_true = true_params['Z']
        ns = int(Z_true.shape[0]/Z_true.shape[1])
        scores_true = np.zeros((Z_true.shape[1], Z_true.shape[1]))
        scores_dist = np.zeros((Z_true.shape[1], Z_true.shape[1], ns))
        for k in range(Z_true.shape[1]):
            for s in range(Z_true.shape[1]):
                scores_true[s,k] = np.mean(np.abs(Z_true[ns*s:ns*(s+1),k]))
                scores_dist[s,k,:] = np.abs(Z_true[ns*s:ns*(s+1),k])

        #Create boxplot
        S1 = [scores_dist[0,i,:] for i in range(Z_true.shape[1])]
        S2 = [scores_dist[1,i,:] for i in range(Z_true.shape[1])]
        S3 = [scores_dist[2,i,:] for i in range(Z_true.shape[1])]
        ticks = [f'Factor {i+1}' for i in range(Z_true.shape[1])]

        plt.figure(figsize=(6, 5), dpi=300)
        dpi = plt.gcf().get_dpi()
        fontsize = 6 * (dpi / 100)
        S1_plot = plt.boxplot(S1, positions=np.array(np.arange(len(S1)))*2.0-0.6, widths=0.5)
        S2_plot = plt.boxplot(S2, positions=np.array(np.arange(len(S2)))*2.0, widths=0.5)
        S3_plot = plt.boxplot(S3, positions=np.array(np.arange(len(S3)))*2.0+0.6, widths=0.5)
        
        # setting colors for each groups
        define_box_properties(S1_plot, '#fdbb84', 'Group 1')
        define_box_properties(S2_plot, '#2b8cbe', 'Group 2')
        define_box_properties(S3_plot, '#99d8c9', 'Group 3')
        plt.xticks(np.arange(0, len(ticks) * 2, 2), ticks,fontsize=0.85*fontsize); plt.xlim(-2, len(ticks)*2)
        plt.yticks(fontsize=0.85*fontsize); plt.ylabel('Absolute latent scores', fontsize=fontsize)
        plt.legend(fontsize=0.85*fontsize)
        plt.savefig(f'{plot_path}/trueSubtype_scores_boxplot.png')
        plt.savefig(f'{plot_path}/svgs/trueSubtype_scores_boxplot.svg'); plt.close()

        # Plot percentage of subtype representation in each component (True)
        total_scores = np.sum(scores_true,axis=0)
        x = np.arange(scores_true.shape[1])
        colors = ['#fdbb84', '#2b8cbe', '#99d8c9']; height=0.3; b_diff = -height
        plt.figure(figsize=(6, 5), dpi=300)
        dpi = plt.gcf().get_dpi()
        fontsize = 6 * (dpi / 100)
        for s in range(scores_true.shape[0]):
            plt.barh(x+b_diff, width=scores_true[s,:]/total_scores, height=height, color=colors[s])
            b_diff += height
        plt.yticks(x, [f'{i+1}' for i in range(Z_true.shape[1])], fontsize=0.8*fontsize)
        plt.xlim([0,1]); plt.xticks(fontsize=0.85*fontsize)
        plt.ylabel('Factors', fontsize=fontsize); plt.xlabel('Factor contributions',fontsize=fontsize)
        plt.legend(['Group 1','Group 2','Group 3'], fontsize=0.85*fontsize); plt.tight_layout()
        plt.savefig(f'{plot_path}/trueSubtype_scores.png')
        plt.savefig(f'{plot_path}/svgs/trueSubtype_scores.svg'); plt.close()

        #Match factors
        Z_tmp = rob_params['Z']
        W_tmp = rob_params['W']
        new_comps = []
        if Z_tmp.shape[1] == Z_true.shape[1]:
            Z_inf = np.zeros((Z_true.shape[0], Z_true.shape[1]))
            W_inf = np.zeros((W_tmp.shape[0], W_tmp.shape[1]))
            sim_matrix = cosine_similarity(Z_true.T, Z_tmp.T)
            for k1 in range(Z_true.shape[1]):
                maxsim = np.argmax(np.abs(sim_matrix[k1,:]))
                new_comps.append(int(maxsim))
                if sim_matrix[k1, maxsim] > 0:
                    Z_inf[:,k1] = Z_tmp[:,maxsim]
                    W_inf[:,k1] = W_tmp[:,maxsim]
                else:       
                    Z_inf[:,k1] = -Z_tmp[:,maxsim]
                    W_inf[:,k1] = -W_tmp[:,maxsim]
            rob_params['Z'] = Z_inf
            rob_params['W'] = W_inf
        else:     
            Z_inf = rob_params['Z']

        # Calculate scores per inferred component
        scores_inf = np.zeros((Z_true.shape[1], Z_inf.shape[1]))
        scores_dist = np.zeros((Z_true.shape[1], Z_inf.shape[1], ns))
        for k in range(Z_inf.shape[1]):
            for s in range(Z_true.shape[1]):
                scores_inf[s,k] = np.mean(np.abs(Z_inf[ns*s:ns*(s+1),k]))
                scores_dist[s,k,:] = np.abs(Z_inf[ns*s:ns*(s+1),k])
        
        #Create boxplot
        S1 = [scores_dist[0,i,:] for i in range(Z_inf.shape[1])]
        S2 = [scores_dist[1,i,:] for i in range(Z_inf.shape[1])]
        S3 = [scores_dist[2,i,:] for i in range(Z_inf.shape[1])]
        ticks = [f'Factor {i+1}' for i in range(Z_inf.shape[1])]

        plt.figure(figsize=(6, 4.5), dpi=300)
        dpi = plt.gcf().get_dpi()
        fontsize = 6 * (dpi / 100)
        S1_plot = plt.boxplot(S1, positions=np.array(np.arange(len(S1)))*2.0-0.6, widths=0.5)
        S2_plot = plt.boxplot(S2, positions=np.array(np.arange(len(S2)))*2.0, widths=0.5)
        S3_plot = plt.boxplot(S3, positions=np.array(np.arange(len(S3)))*2.0+0.6, widths=0.5)
        
        # setting colors for each groups
        define_box_properties(S1_plot, '#fdbb84', 'Group 1')
        define_box_properties(S2_plot, '#2b8cbe', 'Group 2')
        define_box_properties(S3_plot, '#99d8c9', 'Group 3')
        plt.xticks(np.arange(0, len(ticks) * 2, 2), ticks,fontsize=0.85*fontsize); plt.xlim(-2, len(ticks)*2)
        plt.yticks(fontsize=0.85*fontsize); plt.ylabel('Absolute latent scores', fontsize=fontsize)
        plt.legend(fontsize=0.85*fontsize)
        plt.savefig(f'{plot_path}/infSubtype_scores_boxplot.png')
        plt.savefig(f'{plot_path}/svgs/infSubtype_scores_boxplot.svg'); plt.close()
        
        # Plot percentage of subtype representation in each component (Inferred)
        total_scores = np.sum(scores_inf,axis=0)
        x = np.arange(scores_inf.shape[1])
        colors = ['#fdbb84', '#2b8cbe', '#99d8c9']; width=0.2; b_diff = -width
        plt.figure(figsize=(7, 6), dpi=300)
        dpi = plt.gcf().get_dpi()
        fontsize = 7 * (dpi / 100)
        for s in range(scores_inf.shape[0]):
            plt.barh(x+b_diff, width=scores_inf[s,:]/total_scores, height=height, color=colors[s])
            b_diff += height
        plt.yticks(x, [f'{i+1}' for i in range(Z_inf.shape[1])], fontsize=0.8*fontsize)
        plt.xlim([0,1]); plt.xticks(fontsize=0.8*fontsize)
        plt.ylabel('Factors', fontsize=fontsize); plt.xlabel('Factor contributions',fontsize=fontsize)
        plt.legend(['Group 1','Group 2','Group 3'], fontsize=0.9*fontsize); plt.tight_layout()
        plt.savefig(f'{plot_path}/infSubtype_scores.png')
        plt.savefig(f'{plot_path}/svgs/infSubtype_scores.svg'); plt.close()

        #Dictionary with paths to plot the parameters
        inf_paths = {
            'W': f'{plot_path}/infW',
            'lmbW': f'{plot_path}/inflmbW',
            'cW': f'{plot_path}/infcW',
            'tauW': f'{plot_path}/inftauW',
            'sigma': f'{plot_path}/infsigma',
            'Z': f'{plot_path}/infZ',
            'Z_svg': f'{plot_path}/svgs/infZ',
            'tauZ': f'{plot_path}/inftauZ',
            'lmbZ': f'{plot_path}/inflmbZ',
            'cZ': f'{plot_path}/infcZ'}
        
        #Plot inferred parameters                                                   
        plot_param(rob_params, inf_paths, args, cids=None, tr_vals=true_params)              

def genfi(data, res_dir, args):
    
    ofile = open(f'{res_dir}/results.txt','w')
    
    #Find best initialisation
    exp_logs, ofile = find_bestrun(res_dir, args, ofile)
    brun = np.nanargmax(exp_logs)+1
    print('Best run: ', brun, file=ofile) 

    # plot dir
    plot_path = f'{res_dir}/plots_{brun}'
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)
        os.makedirs(f'{plot_path}/svgs')

    #Total variance explained by real data
    X = data.get('X')
    Y = data.get('Y')
    #standardise matrices
    X_stand = StandardScaler().fit_transform(X)
    Tvar = np.trace(np.dot(X_stand.T,X_stand))

    #Get model's outputs
    rparams_path = f'{res_dir}/[{brun}]Robust_params.dictionary'
    if os.stat(rparams_path).st_size > 5:       
        with open(rparams_path, 'rb') as parameters:
            rob_params = pickle.load(parameters)
        
        #calculate variance explained by each factor
        var_comps = []
        X_inf = rob_params['infX']
        for k in range(len(X_inf)):
            var_Xk = np.trace(np.dot(X_inf[k][0].T, X_inf[k][0]))/Tvar
            var_comps.append(var_Xk)

        #Scree Plot
        varexp_comps = np.array(var_comps) 
        ids_var = np.argsort(-varexp_comps)
        varexp_comps = varexp_comps[ids_var]
        x = np.arange(len(var_comps)+1)
        cum_var = [0]
        for i in range(1, varexp_comps.size+1):
            if i == 1:
                cum_var.append(varexp_comps[i-1]*100)
            else:
                cum_var.append(varexp_comps[i-1]*100 + cum_var[i-1])
        plt.figure(figsize=(5,5), dpi=300)
        plt.plot(x, cum_var, 'ko-', linewidth=2)
        plt.xlabel('Factors'); plt.ylabel('Covariance explained (%)') 
        plt.xticks(x, [f'{i}' for i in range(x.size)])   
        plt.savefig(f'{plot_path}/Scree_plot.png')
        plt.savefig(f'{plot_path}/svgs/Scree_plot.svg')
        plt.close()

        #plot weights and print total explained variance (and per component)
        df_var = pd.read_csv(f'../data/GENFI/var_labels.csv') 
        W = rob_params.get('W')[:,ids_var]
        print(f'\nTotal variance explained: {np.around(sum(var_comps) * 100, 2)}\n', file=ofile)
        cli_labels = list(df_var.iloc[28:,1])
        brain_labels = list(df_var.iloc[0:28,1])

        if 'sparseGFA' in args.model:
        
            # Create all non-imaging loadings figure
            fig, axes = plt.subplots(1, len(var_comps), figsize=(20, 10), dpi=300)
            axes = axes.flatten()
            for j in range(len(var_comps)):
                # Extract clinical weights for the current component
                w_cli = W[28:, j]
                
                # Determine colors based on signs of weights
                colours = ['#b15352' if w > 0 else '#5ba3b4' for w in w_cli]
                
                # Plot the bar plot on the current subplot
                ax = axes[j]
                ax.barh(np.arange(w_cli.size), w_cli, color=colours)
                ax.set_xlabel('Loadings')
                ax.set_title(f'Factor {j+1}')
                ax.set_xlim([-0.5, 0.5])
                ax.set_yticks(np.arange(w_cli.size))
                if j == 0:
                    ax.set_ylabel('Non-imaging variables')
                    ax.set_yticklabels(cli_labels, fontsize=10)
                else:
                    ax.set_yticklabels([])
                ax.tick_params(axis='x', labelsize=10)
                #ax.set_facecolor("white")
            plt.tight_layout()
            plt.savefig(f'{plot_path}/all_NIloadings.png')
            plt.savefig(f'{plot_path}/svgs/all_NIloadings.svg')
            plt.close()

            # Create all brain loadings figure
            fig, axes = plt.subplots(1, len(var_comps), figsize=(20, 10), dpi=300)
            axes = axes.flatten()
            for j in range(len(var_comps)):
                # Extract clinical weights for the current component
                w_brain = W[0:28,j]
                
                # Determine colors based on signs of weights
                colours = ['#b15352' if w > 0 else '#5ba3b4' for w in w_brain]
                
                # Plot the bar plot on the current subplot
                ax = axes[j]
                ax.barh(np.arange(w_brain.size), w_brain, color=colours)
                ax.set_xlabel('Loadings')
                ax.set_title(f'Factor {j+1}')
                ax.set_xlim([-0.5, 0.5])
                ax.set_yticks(np.arange(w_brain.size))
                if j == 0:
                    ax.set_ylabel('Brain variables')
                    ax.set_yticklabels(brain_labels, fontsize=10)
                else:
                    ax.set_yticklabels([])
                ax.tick_params(axis='x', labelsize=10)
                #ax.set_facecolor("white")
            plt.tight_layout()
            plt.savefig(f'{plot_path}/all_brainloadings.png')
            plt.savefig(f'{plot_path}/svgs/all_brainloadings.svg')
            plt.close()
        
        # Get top components
        top = 4
        if len(var_comps) > top:
            pass
        else:
            top = len(var_comps)
        
        for j in range(top):

            print(f'Variance explained by cmp {j+1}: {np.around(var_comps[j] * 100, 2)}', file=ofile) 
            # Plot clinical weights 
            w_cli = W[28:,j]
            w_sort = w_cli[np.argsort(w_cli)]
           
            # Determine colors based on signs of weights
            colours = ['#b15352' if w > 0 else '#5ba3b4' for w in w_sort]
            
            plt.figure(figsize=(4,7), dpi=300)
            ax = plt.axes()
            #ax.set_facecolor("white")     
            plt.barh(np.arange(w_cli.size), w_sort, color = colours)
            plt.ylabel('Non-imaging variables'); plt.xlabel('Loadings'); plt.xlim([-0.5,0.5])
            plt.yticks(np.arange(w_cli.size).tolist(), [cli_labels[np.argsort(w_cli)[i]] for i in range(len(cli_labels))], fontsize=9); plt.xticks(fontsize=9)
            plt.tight_layout()
            plt.savefig(f'{plot_path}/NI_loadings{j+1}.png') 
            plt.savefig(f'{plot_path}/svgs/NI_loadings{j+1}.svg') 
            plt.close()

            # Plot brain weights + Asymmetry
            w_brain = W[0:28,j]
            w_sort = w_brain[np.argsort(w_brain)]
        
            # Determine colors based on signs of weights
            colours = ['#b15352' if w > 0 else '#5ba3b4' for w in w_sort]

            plt.figure(figsize=(4,7),dpi=300)
            ax = plt.axes()
            #ax.set_facecolor("white")     
            plt.barh(np.arange(w_brain.size), w_sort, color = colours)
            plt.ylabel('Brain variables'); plt.xlabel('Loadings'); plt.xlim([-0.5,0.5])
            plt.yticks(np.arange(w_brain.size).tolist(), [brain_labels[np.argsort(w_brain)[i]] for i in range(len(brain_labels))], fontsize=9); plt.xticks(fontsize=9)
            plt.tight_layout()
            plt.savefig(f'{plot_path}/Brain_loadings{j+1}.png')
            plt.savefig(f'{plot_path}/svgs/Brain_loadings{j+1}.svg')
            plt.close()

        # Plot robust components
        abs_mean_scores, scores_dist = plot_components(rob_params, top, ids_var, plot_path)
        
        # Plot absolute mean score of each subtype in each factor
        total_scores = np.sum(abs_mean_scores,axis=0)
        x = np.arange(abs_mean_scores.shape[1])
        colors = ['#fdbb84', '#2b8cbe', '#99d8c9']
        width=0.3; b_diff = -width
        plt.figure(figsize=(7, 5), dpi=300)
        dpi = plt.gcf().get_dpi()
        fontsize = 5 * (dpi / 100)
        for s in range(abs_mean_scores.shape[0]):
            plt.bar(x+b_diff, abs_mean_scores[s,:]/total_scores, width=width, color=colors[s])
            b_diff += width
        plt.xticks(x, [f'{i+1}' for i in range(x.size)], fontsize=0.8*fontsize)
        plt.ylim([0, 1]); plt.yticks(fontsize=0.8*fontsize)
        plt.xlabel('Factors', fontsize=fontsize); plt.ylabel('Factor contributions', fontsize=fontsize)
        plt.legend(['C9orf72','GRN', 'MAPT'], fontsize=0.85*fontsize); plt.tight_layout()
        plt.savefig(f'{plot_path}/Subtype_scores.png')
        plt.savefig(f'{plot_path}/svgs/Subtype_scores.svg')
        plt.close()
        
        # Plot absolute mean score of each subtype in each factor (top)
        x = np.arange(top)
        height=0.2; b_diff = -height
        plt.figure(figsize=(4, 5), dpi=300)
        dpi = plt.gcf().get_dpi()
        fontsize = 5 * (dpi / 100)
        for s in range(abs_mean_scores.shape[0]):
            plt.barh(x+b_diff, width=abs_mean_scores[s,0:top]/total_scores[0:top], height=height, color=colors[s])
            b_diff += height
        plt.yticks(x, [f'{i+1}' for i in range(top)], fontsize=0.8*fontsize); plt.xticks(fontsize=0.8*fontsize)
        plt.xlabel('Factor contributions', fontsize=fontsize); plt.ylabel('Factors', fontsize=fontsize)
        plt.legend(['C9orf72','GRN', 'MAPT'], fontsize=0.85*fontsize); plt.tight_layout()
        plt.savefig(f'{plot_path}/Subtype_scores_top{top}.png')
        plt.savefig(f'{plot_path}/svgs/Subtype_scores_top{top}.svg')
        plt.close()

        # Plot boxplot
        plt.figure(figsize=(7, 5), dpi=300)
        dpi = plt.gcf().get_dpi()
        fontsize = 5 * (dpi / 100)
        C9 = [scores_dist['s1'][:,i] for i in range(top)]
        GRN = [scores_dist['s2'][:,i] for i in range(top)]
        MAPT = [scores_dist['s3'][:,i] for i in range(top)]
        ticks = [f'Factor {i+1}' for i in range(top)]
        C9_plot = plt.boxplot(C9, positions=np.array(np.arange(len(C9)))*2.0-0.6, widths=0.5)
        GRN_plot = plt.boxplot(GRN,positions=np.array(np.arange(len(GRN)))*2.0, widths=0.5)
        MAPT_plot = plt.boxplot(MAPT, positions=np.array(np.arange(len(MAPT)))*2.0+0.6, widths=0.5)    
        define_box_properties(C9_plot, '#fdbb84', 'C9orf72')
        define_box_properties(GRN_plot, '#2b8cbe', 'GRN')
        define_box_properties(MAPT_plot, '#99d8c9', 'MAPT')
        plt.xticks(np.arange(0, len(ticks) * 2, 2), ticks, fontsize=0.8*fontsize)
        plt.yticks(fontsize=0.8*fontsize)
        plt.legend(fontsize=0.8*fontsize)
        plt.xlim(-2, len(ticks)*2)
        plt.ylabel('Absolute latent scores', fontsize=fontsize)
        plt.savefig(f'{plot_path}/Subtype_scores_boxplot.png', dpi=300)
        plt.savefig(f'{plot_path}/svgs/Subtype_scores_boxplot.svg')
        plt.close()
        
        # Compute F statistic
        scores = np.abs(rob_params['Z'][:, ids_var])
        N = scores.shape[0]
        df_subjs = pd.read_csv(f'../data/GENFI/visit11_data_{N}subjs.csv')
        ids = list(df_subjs['Blinded Code'])
        ns = [sum(['C9ORF' in x for x in ids]),
                sum(['GRN' in x for x in ids]),
                sum(['MAPT' in x for x in ids])]
        g1 = scores[0:ns[0],:] # C9ORF
        g2 = scores[ns[0]:ns[0]+ns[1],:] #GRN
        g3 = scores[ns[0]+ns[1]:,:] # MAPT
        comps = [f'Factor {i+1}' for i in range(scores.shape[1])]
        
        #all
        stats_all = f_oneway(g1,g2,g3)
        df_all = pd.DataFrame(index=comps,columns= ['F score', 'p-value'])
        df_all['F score'] = stats_all[0]; df_all['p-value'] = stats_all[1]
        df_all.to_csv(f'{plot_path}/Ftest.csv')

        #Compute separate t tests
        #g1vg2
        stats_g1g2 = ttest_ind(g1,g2)
        df_g1g2 = pd.DataFrame(index=comps,columns= ['t', 'p-value'])
        df_g1g2['t'] = stats_g1g2[0]; df_g1g2['p-value'] = stats_g1g2[1]
        df_g1g2.to_csv(f'{plot_path}/Ttest_C9vGRN.csv')
        #g2vg3
        stats_g2g3 = ttest_ind(g2,g3)
        df_g2g3 = pd.DataFrame(index=comps,columns= ['t', 'p-value'])
        df_g2g3['t'] = stats_g2g3[0]; df_g2g3['p-value'] = stats_g2g3[1]
        df_g2g3.to_csv(f'{plot_path}/Ttest_MAPTvGRN.csv')
        #g1vg3
        stats_g1g3 = ttest_ind(g1,g3)
        df_g1g3 = pd.DataFrame(index=comps,columns= ['t', 'p-value'])
        df_g1g3['t'] = stats_g1g3[0]; df_g1g3['p-value'] = stats_g1g3[1]
        df_g1g3.to_csv(f'{plot_path}/Ttest_C9vMAPT.csv')

def find_bestrun(res_dir, args, ofile):
    
    # Find the best run
    exp_logs = np.nan * np.ones((1, args.num_runs))
    for r in range(args.num_runs):
        res_path = f'{res_dir}/[{r+1}]Model_params.dictionary'
        if os.stat(res_path).st_size > 5:
            with open(res_path, 'rb') as parameters:
                mcmc_samples = pickle.load(parameters)

            #get expected log joint density
            exp_logs[0,r] = mcmc_samples['exp_logdensity']
            print('Run: ', r + 1, file=ofile)
            print('MCMC elapsed time: {:.2f} h'.format(
                mcmc_samples['time_elapsed']/60), file=ofile)

            print('Expected log joint density: {:.2f}\n'.format(
                exp_logs[0,r]), file=ofile)             
        else:
            print('The model output file is empty!')
            sys.exit(1)
    return exp_logs, ofile    

def plot_components(params, top, ids_var, path):

    #Plot components for GENFI
    X = params['infX']
    N = params['Z'].shape[0]
    data_dir = '../data/GENFI'
    df_subjs = pd.read_csv(f'{data_dir}/visit11_data_{N}subjs.csv')
    ids = list(df_subjs['Blinded Code'])
    subtype_labels = df_subjs["Genetic Group"]
    df_var = pd.read_csv(f'{data_dir}/var_labels.csv')

    #subtype colors
    colors = ['#fdbb84', '#2b8cbe', '#99d8c9']
    subtype_lut = dict(zip(subtype_labels.unique(), colors))
    subtype_colors = subtype_labels.map(subtype_lut)
    subtype_colors.name = ''      
    
    #feature colors
    view_labels = df_var['view']
    view_lut = dict(zip(view_labels.unique(), ['#993404', '#fec44f']))
    view_colors = [view_labels.map(view_lut)]
    var_labels = list(df_var['new_labels'])

    Z = params['Z']
    Z = Z[:, ids_var]
    lcomps = [f'Factor {k+1}' for k in range(Z.shape[1])]
    
    # Plot clustermap all comps
    df_Z = pd.DataFrame(Z, columns = lcomps)
    cm = sns.clustermap(df_Z, 
                    vmin=np.min(Z), 
                    vmax=np.max(Z), 
                    cmap="vlag", 
                    center=0.00,
                    row_colors=subtype_colors,
                    row_cluster=False,
                    col_cluster=False,
                    xticklabels=True,
                    yticklabels=False,
                    figsize=(7.5,5)
                    )                   
    for label in subtype_labels.unique():
        cm.ax_row_dendrogram.bar(0, 0, color=subtype_lut[label], label=label, linewidth=0)   
    cm.ax_row_dendrogram.legend(loc="center", ncol=1, bbox_transform=gcf().transFigure) 
    cm.ax_row_dendrogram.legend(title='Genetic group', loc="center", ncol=1, bbox_transform=gcf().transFigure)   
    plt.savefig(f'{path}/infZ_ord.png')
    plt.savefig(f'{path}/svgs/infZ_ord.svg')
    plt.close()

    # Plot clustermap top comps
    df_Z = pd.DataFrame(Z[:,:top], columns = lcomps[:top])
    cm = sns.clustermap(df_Z, 
                    vmin=np.min(Z[:,:top]), 
                    vmax=np.max(Z[:,:top]), 
                    cmap="vlag", 
                    center=0.00,
                    row_colors=subtype_colors,
                    row_cluster=False,
                    col_cluster=False,
                    xticklabels=True,
                    yticklabels=False,
                    figsize=(6,4.5)
                    )                   
    for label in subtype_labels.unique():
        cm.ax_row_dendrogram.bar(0, 0, color=subtype_lut[label], label=label, linewidth=0)   
    cm.ax_row_dendrogram.legend(loc="center", ncol=1, bbox_transform=gcf().transFigure) 
    cm.ax_row_dendrogram.legend(title='Genetic group', loc="center", ncol=1, bbox_transform=gcf().transFigure)
    plt.savefig(f'{path}/infZ_ord_top.png')
    plt.savefig(f'{path}/svgs/infZ_ord_top.svg')
    plt.close()

    nsubt = [sum(['C9ORF' in x for x in ids]),
            sum(['GRN' in x for x in ids]),
            sum(['MAPT' in x for x in ids])]  
    subtype_scores = np.zeros((len(nsubt), len(X)))
    scores_dict = {'s1': np.zeros((nsubt[0], len(X))),
                's2': np.zeros((nsubt[1], len(X))),
                's3': np.zeros((nsubt[2], len(X)))}
    
    for k in range(len(X)):
        z_k = Z[:,k]
            
        #Subtypes
        ns = 0
        for s in range(len(nsubt)):
            z = z_k[ns:ns+nsubt[s]]
            subtype_scores[s,k] = np.mean(np.abs(z))
            scores_dict[f's{s+1}'][:,k] = np.abs(z)
            ns += nsubt[s]

    for k in range(top):
        z_k = Z[:,k]
        df_X = pd.DataFrame(X[ids_var[k]][0], columns = var_labels) 

        #Individual scores (clustermap)
        df_Z = pd.DataFrame(z_k, columns = [f'Factor {k+1}'])
        cm = sns.clustermap(df_Z, 
                        vmin=np.min(Z), 
                        vmax=np.max(Z), 
                        cmap="vlag", 
                        center=0.00,
                        row_colors=subtype_colors,
                        row_cluster=False,
                        col_cluster=False,
                        xticklabels=True,
                        yticklabels=False,
                        figsize=(2.5,5)
                        )                   
        for label in subtype_labels.unique():
            cm.ax_col_dendrogram.bar(0, 0, color=subtype_lut[label], label=label, linewidth=0)   
        cm.ax_col_dendrogram.legend(loc="center", ncol=1, bbox_transform=gcf().transFigure) 
        cm.ax_col_dendrogram.legend(title='Genetic group', loc="center", fontsize=8, ncol=1, bbox_transform=gcf().transFigure)   
        plt.savefig(f'{path}/infZ_clustermap_comp{k+1}.png', dpi=300) 
        plt.savefig(f'{path}/svgs/infZ_clustermap_comp{k+1}.svg') 
        plt.close()

        # Plot components on data space
        cm = sns.clustermap(df_X.T, 
                vmin=np.min(X[ids_var[k]][0]), 
                vmax=np.max(X[ids_var[k]][0]), 
                cmap="vlag", 
                center=0.00,
                row_colors=view_colors,
                col_colors=subtype_colors,
                row_cluster=True,
                col_cluster=True,
                xticklabels=False,
                yticklabels=True,
                figsize=(11.5,11.5)
                )                   
        for label in subtype_labels.unique():
            cm.ax_col_dendrogram.bar(0, 0, color=subtype_lut[label], label=label, linewidth=0)   
        cm.ax_col_dendrogram.legend(loc="center", ncol=1, bbox_transform=gcf().transFigure)  
        cm.ax_col_dendrogram.legend(title='Genetic Group', loc="center", ncol=1, bbox_transform=gcf().transFigure)   

        for label in view_labels.unique():
            cm.ax_row_dendrogram.bar(0, 0, color=view_lut[label], label=label, linewidth=0)   
        cm.ax_row_dendrogram.legend(title='Modality',loc="upper left", ncol=1, bbox_transform=gcf().transFigure)            
        plt.savefig(f'{path}/infX_comp{k+1}.png', dpi=300)
        plt.savefig(f'{path}/svgs/infX_comp{k+1}.svg')
        plt.close()
    
    return subtype_scores, scores_dict     

def plot_param(params, paths, args, cids=None, tr_vals=False):
    
    lcomps = list(range(1, params['W'].shape[1]+1))
    #plot W
    if 'W' in params:
        W = params['W']
        pathW = paths['W']
        sns.heatmap(W, vmin=-np.max(np.abs(W)), vmax=np.max(np.abs(W)), cmap="vlag", 
                    yticklabels=False, xticklabels=list(map(str, lcomps)))
        plt.xlabel('Factors', fontsize=11); plt.ylabel('D', fontsize=11) 
        plt.title('Loading matrices (W)', fontsize=12)                
        plt.savefig(f'{pathW}.png', dpi=200); plt.close()
    
    #plot lambda W
    if 'lmbW' in params:
        if cids is not None:
            lmbW = params['lmbW'][:,cids]
        else:
            lmbW = params['lmbW']
        pathlmbW = paths['lmbW'] 
        sns.heatmap(lmbW, vmin=-np.max(np.abs(lmbW)), vmax=np.max(np.abs(lmbW)), cmap="vlag", 
                    yticklabels=False, xticklabels=list(map(str, lcomps)))
        plt.xlabel('Factors'); plt.ylabel('D')  
        plt.savefig(f'{pathlmbW}.png'); plt.close()
    
    #plot Z
    if 'Z' in params:
        Z = params['Z']
        pathZ = paths['Z']
        pathZ_svg = paths['Z_svg']
        plt.figure(figsize=(6, 5), dpi=300)
        dpi = plt.gcf().get_dpi()
        fontsize = 6 * (dpi / 100)
        sns.heatmap(Z, vmin=-np.max(np.abs(Z)), vmax=np.max(np.abs(Z)), cmap="vlag", 
                    yticklabels=False, xticklabels=list(map(str, lcomps)))     
        plt.xlabel('Factors', fontsize=fontsize); plt.ylabel('Latent variables', fontsize=fontsize) 
        plt.xticks(fontsize=0.85*fontsize) 
        plt.savefig(f'{pathZ}.png')
        plt.savefig(f'{pathZ_svg}.svg'); plt.close()
    
    #plot lambda Z
    if 'lmbZ' in params:
        if cids is not None:
            lmbZ = params['lmbZ'][:,cids]
        else:
            lmbZ = params['lmbZ']
        pathlmbZ = paths['lmbZ'] 
        sns.heatmap(lmbZ, vmin=-np.max(np.abs(lmbZ)), vmax=np.max(np.abs(lmbZ)), cmap="vlag", 
                    yticklabels=False, xticklabels=list(map(str, lcomps)))
        plt.xlabel('Factors'); plt.ylabel('Training samples')
        plt.savefig(f'{pathlmbZ}.png'); plt.close()
    
    #plot tau W
    if 'tauW_inf' in params:
        tau = params['tauW_inf']
        pathtau = paths['tauW']
        f, axes = plt.subplots(args.num_sources, 1, figsize=(8,6))
        f.subplots_adjust(hspace=0.5, wspace=0.2)
        for m, ax in zip(range(args.num_sources), axes.flat):
            sns.histplot(tau[:,m], ax=ax, color='#2b8cbe')
            if 'synthetic' in args.dataset:
                ax.axvline(x=tr_vals['tauW'][0,m], color='red')
            ax.set_title(f'View {m+1}'); ax.set_ylabel('Number of samples')
        plt.savefig(f'{pathtau}.png'); plt.close()                        
    
    #Plot sigmas
    if 'sigma_inf' in params:
        sigma = params['sigma_inf']
        pathsig = paths['sigma']
        f, axes = plt.subplots(args.num_sources, 1, figsize=(8,6))
        f.subplots_adjust(hspace=0.5, wspace=0.2)
        for m, ax in zip(range(args.num_sources), axes.flat):
            sns.histplot(sigma[:,m], ax=ax, color='#2b8cbe')
            if 'synthetic' in args.dataset:
                ax.axvline(x=tr_vals['sigma'][m], color='red')
            ax.set_title(f'View {m+1}'); ax.set_ylabel('Number of samples')
        plt.savefig(f'{pathsig}.png'); plt.close()         

def plot_X(data, args, hypers, path, true_data=False):
    
    if true_data:
        X = np.dot(data['Z'], data['W'].T)
        K = data['Z'].shape[1]
    else:    
        X = np.zeros((data[0][0].shape[0], data[0][0].shape[1]))
        K = len(data)
    for k in range(K):
        if true_data:
            z = np.reshape(data['Z'][:,k], (data['Z'].shape[0], 1)) 
            w = np.reshape(data['W'][:,k], (data['W'].shape[0], 1))
            X_k = np.dot(z,w.T)
        else:
            X_k = data[k][0]    
            X += X_k
        fig, axes = plt.subplots(ncols=args.num_sources)
        fig.subplots_adjust(wspace=0.02)
        Dm = hypers['Dm']; d = 0
        for m in range(args.num_sources):
            if m < args.num_sources - 1:
                sns.heatmap(X_k[:,d:d+Dm[m]], 
                        vmin=np.min(X_k), 
                        vmax=np.max(X_k), 
                        cmap="vlag", 
                        ax=axes[m],
                        cbar=False,
                        xticklabels=False,
                        yticklabels=False)    
            else:
                sns.heatmap(X_k[:,d:d+Dm[m]], 
                        vmin=np.min(X_k), 
                        vmax=np.max(X_k), 
                        cmap="vlag",  
                        ax=axes[m],
                        cbar=True,
                        xticklabels=False,
                        yticklabels=False)                  
            d += Dm[m]
        plt.title(f'Factor {k+1} (Input space)')                 
        plt.savefig(f'{path}_comp{k+1}.png'); plt.close() 
    
    #Plot X 
    plt.figure()    
    sns.heatmap(X, 
                vmin=np.min(X), 
                vmax=np.max(X), 
                cmap="vlag", 
                xticklabels=False,
                yticklabels=False)
    plt.xlabel('D'); plt.ylabel('N')                    
    plt.savefig(f'{path}.png'); plt.close()

def define_box_properties(plot_name, color_code, label):
    for k, v in plot_name.items():
        plt.setp(plot_name.get(k), color=color_code)  
    # use plot function to draw a small line to name the legend.
    plt.plot([], c=color_code, label=label)
    plt.legend()
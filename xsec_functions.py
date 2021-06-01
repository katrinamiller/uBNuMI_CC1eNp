## UPDATE THE PLOTS PATH BEFORE SAVING ## 

import math
import warnings

import scipy.stats
from scipy.stats import norm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec


signal = '(nu_pdg==12 and ccnc==0 and nproton>0 and npion==0 and npi0==0 and nu_purity_from_pfp>0.5)'

plots_path = "/uboone/data/users/kmiller/searchingfornues_v33/v08_00_00_33/plots/"

##################################################################################################
def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)

    variance = np.average((values-average)**2, weights=weights)
    
    return (average, math.sqrt(variance))

##################################################################################################
def best_bin(lhs, rhs, x, df, horn, save):

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    it = 0

    for n in range(x): 
    
        reco_bin_size = rhs - lhs
        #print(reco_bin_size)

        # take events inside this reconstructed bin & plot the truth level distribution of them 
        q = signal+' and '+str(lhs)+'<shr_energy_cali<='+str(rhs)

        # best fit of raw data ( need to fix lack of weighting )
        #mu, sigma = norm.fit(df.query(q)['elec_e'])
        mu, sigma = weighted_avg_and_std(df.query(q)['elec_e'], df.query(q)['totweight_proj'])
    
        if (round(2*sigma, 2) == round(reco_bin_size, 2)) or (it==x-1): 
        
            # plot truth level distribution 
            n, b, p = plt.hist(df.query(q)['elec_e'],30, 
                histtype='step', label="# signal = "+str(round(sum(df.query(q)['totweight_proj']), 2)),
                weights=df.query(q)['totweight_proj'], color='orange', density=False)

            area = np.sum(np.diff(b)*n)
    
            # Plot the PDF.
            xmin, xmax = plt.xlim()
            x = np.linspace(xmin, xmax, 100)
            p = norm.pdf(x, mu, sigma)

            plt.plot(x, p*area, 'k', linewidth=2, color='red')

            title = "Fit results: mu = %.2f,  std = %.2f" % (mu, sigma)
            plt.title(title, fontsize=15)
    
            plt.xlabel('True Electron Energy [GeV]', fontsize=15)
            plt.legend()
            plt.tight_layout()
            
            if save: 
                plt.savefig(plots_path+horn+"/"+horn+"_res.pdf", transparent=True, bbox_inches='tight')

            plt.show()

            print('reco bin size = '+str(reco_bin_size))
            print('true 2*RMS = '+str(sigma*2))
            print('iterations = '+str(it))
        
            print('RHS value = '+str(rhs))
            print('# of signal events='+str(sum(df.query(q)['totweight_proj'])))
            #print('# of signal events='+str(sum(df.query(q)['proj_pot'])))
            
            
        
            break
        
        else: 
            rhs = rhs + 0.01 
            it += 1

##################################################################################################
# plot the detector resolution 
def true_reco_res(true_var, reco_var, df, horn, ymax, save): 
    
    if horn=='fhc': 
        pot = '2.4E22' #'2.0E20'#'9.23E20'
        
    elif horn=='rhc': 
        pot = '2.5E22' #'5.0E20' #'11.95E20'
        
    elif horn=='both': 
        pot = '4.9E22' #'7.0E20'#'2.12E21'
          
    true_values = df.query(signal)[true_var]
    reco_values = df.query(signal)[reco_var]
    
    if reco_var=="NeutrinoEnergy2": 
        reco_values = reco_values/1000
    
    res = np.array((true_values-reco_values)/true_values)
    
    mu, sigma = weighted_avg_and_std(res, df.query(signal)['totweight_intrinsic'])

    fig = plt.figure(figsize=(8, 5))
    n, b, p = plt.hist(res, histtype='step', weights=df.query(signal)['totweight_intrinsic'], color='orange', range=[-1, 1])

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('(true - reco) / true', fontsize=14)
    plt.ylabel('$\\nu$ / '+pot+' POT', fontsize=14)
    
    # Plot the PDF.
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, sigma)
    area = np.sum(np.diff(b)*n)

    plt.plot(x, p*area, 'k', linewidth=2, color='red')

    if horn=='fhc': 
        title = "FHC: mu = %.2f,  std = %.2f" % (mu, sigma)
    elif horn=='rhc': 
        title = "RHC: mu = %.2f,  std = %.2f" % (mu, sigma)
    elif horn=='both': 
        title = "FHC+RHC: mu = %.2f,  std = %.2f" % (mu, sigma)
    plt.title(title, fontsize=15)
    #plt.ylim(0, ymax)
    plt.tight_layout()
    
    if save: 
        plt.savefig(plots_path+horn+"/"+horn+"_res.pdf", transparent=True, bbox_inches='tight')
        
    plt.show()
    
    

    
    
##################################################################################################
# plot the selected truth-level distribution & efficiency as a function of true bin 
def truth_and_eff(true_var, bins, df_infv, df_sel, horn, ymax, xlbl, save):
    
    if horn=='fhc': 
        pot = '9.23E20'
        
    elif horn=='rhc': 
        pot = '11.95E20'
        
    
    gen = plt.hist(df_infv.query(signal)[true_var], bins, weights=df_infv.query(signal)['totweight_overlay'])
    sel = plt.hist(df_sel.query(signal)[true_var], bins, weights=df_sel.query(signal)['totweight_overlay'])
    plt.close()

    fig, ax1 = plt.subplots(figsize=(8, 5))
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    #plt.grid(linestyle=':')

    color = 'tab:orange'
    ax1.set_xlabel(xlbl, fontsize=15)
    ax1.set_ylabel('$\\nu$ / '+pot+' POT', color=color, fontsize=15)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0, ymax)

    ax1.hist(df_sel.query(signal)[true_var], bins, range=[0, 4], color='orange', 
         weights=df_sel.query(signal)['totweight_proj'])

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    ax2.set_ylabel('Efficiency', color='seagreen', fontsize=15)  # we already handled the x-label with ax1
    ax2.tick_params(axis='y', labelcolor='seagreen', labelsize=14) 

    eff = [i/j for i, j in zip(sel[0], gen[0])]
    eff_err = []
    for i in range(len(eff)): 
        eff_err.append(math.sqrt( (eff[i]*(1-eff[i]))/gen[0][i] ) )
    
    bc = 0.5*(sel[1][1:]+sel[1][:-1])
    x_err = []
    for i in range(len(sel[1])-1): 
        x_err.append((sel[1][i+1]-sel[1][i])/2)

    ax2.errorbar(bc, eff, xerr=x_err, yerr=eff_err, fmt='o', color='seagreen', ecolor='seagreen', markersize=3) 
    ax2.set_ylim(0, .4)
    
    if horn=='fhc': 
        plt.title('FHC', fontsize=15)
    if horn=='rhc': 
        plt.title('RHC', fontsize=15)
        
    if save: 
        plt.savefig("/uboone/data/users/kmiller/searchingfornues_v33/v08_00_00_33/plots/"+horn+"/"+horn+"_truth_eff.pdf", 
           transparent=True, bbox_inches='tight')    
        
    plt.tight_layout()
    plt.show()
    
##################################################################################################
# plot efficiency as a function of true bin 
# for both datasets, combined
def comb_truth_and_eff(true_var, bins, df_infv, df_sel, xlbl, save):
    
    pot = '2.12E21' #'7.0E20'
    
    ##################################
    # using MC statistics 
    
    fhc_gen = plt.hist(df_infv.query(signal+' and horn=="fhc"')[true_var], bins, 
                       weights=df_infv.query(signal+' and horn=="fhc"')['totweight_intrinsic'])
    fhc_sel = plt.hist(df_sel.query(signal+' and horn=="fhc"')[true_var], bins, 
                       weights=df_sel.query(signal+' and horn=="fhc"')['totweight_intrinsic'])
    plt.close()
    
    rhc_gen = plt.hist(df_infv.query(signal+' and horn=="rhc"')[true_var], bins, 
                       weights=df_infv.query(signal+' and horn=="rhc"')['totweight_intrinsic'])
    rhc_sel = plt.hist(df_sel.query(signal+' and horn=="rhc"')[true_var], bins, 
                       weights=df_sel.query(signal+' and horn=="rhc"')['totweight_intrinsic'])
    plt.close()
    
    ##################################

    fig = plt.figure(figsize=(8, 5))
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    plt.hist([df_sel.query(signal+' and horn=="fhc"')[true_var], 
                    df_sel.query(signal+' and horn=="rhc"')[true_var]], 
                   bins, stacked=True, color=['orange', 'chocolate'], label=['FHC', 'RHC'], 
                   weights=[df_sel.query(signal+' and horn=="fhc"')['totweight_proj'], 
                           df_sel.query(signal+' and horn=="rhc"')['totweight_proj']])
    
    print('total projected events = '+str(sum(df_sel.query(signal)['totweight_proj'])))

    plt.xlabel(xlbl, fontsize=15)
    plt.ylabel('$\\nu$ / '+pot+' POT', fontsize=15)
    plt.title('FHC + RHC (PROJECTED)', fontsize=15)
    plt.legend(fontsize=15)
    if save:
        plt.savefig("/uboone/data/users/kmiller/searchingfornues_v33/v08_00_00_33/plots/both/both_truth_sig.pdf", transparent=True, bbox_inches='tight')    
        
    plt.tight_layout()
    plt.show()  
    
    ################################
    # efficiency needs to take into account the weighting - don't use data POT statistics, use MC 
    
    fhc_eff = [i/j for i, j in zip(fhc_sel[0], fhc_gen[0])]
    fhc_eff_err = []
    for i in range(len(fhc_eff)): 
        fhc_eff_err.append(math.sqrt( (fhc_eff[i]*(1-fhc_eff[i]))/fhc_gen[0][i] ) )
        
    rhc_eff = [i/j for i, j in zip(rhc_sel[0], rhc_gen[0])]
    rhc_eff_err = []
    for i in range(len(rhc_eff)): 
        rhc_eff_err.append(math.sqrt( (rhc_eff[i]*(1-rhc_eff[i]))/rhc_gen[0][i] ) )
    
    # bin width (as x errors)
    bc = 0.5*(fhc_sel[1][1:]+fhc_sel[1][:-1])
    x_err = []
    for i in range(len(fhc_sel[1])-1): 
        x_err.append((fhc_sel[1][i+1]-fhc_sel[1][i])/2)
        
    ####################################
    
    fig = plt.figure(figsize=(8, 5))
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    plt.errorbar(bc, fhc_eff, xerr=x_err, yerr=fhc_eff_err, 
                 fmt='o', color='seagreen', label='FHC', ecolor='seagreen', markersize=3) 
    
    plt.errorbar(bc, rhc_eff, xerr=x_err, yerr=rhc_eff_err, 
                 fmt='o', color='lightsteelblue', label='RHC', ecolor='lightsteelblue', markersize=3) 
    
    plt.ylim(0, .4)
    plt.xlabel(xlbl, fontsize=15)
    plt.ylabel('Efficiency', fontsize=15) 
    plt.title('FHC + RHC', fontsize=15)
    plt.legend(fontsize=15)

    if save:
            plt.savefig("/uboone/data/users/kmiller/searchingfornues_v33/v08_00_00_33/plots/both/both_truth_eff.pdf", transparent=True, bbox_inches='tight')    
        
    plt.tight_layout()
    plt.show() 
    
##################################################################################################
# plot the smearing matrix 
def smear_matrix(true_var, reco_var, bins, df_sel, horn, zmax, lbl, save): 

    if horn=='fhc': 
        pot = '2.4E22'
        title = "FHC selected signal"
        
    elif horn=='rhc': 
        pot = '2.5E22'
        title = "RHC selected signal"
        
    elif horn=='both': 
        pot = '4.9E22'
        title = "FHC + RHC selected signal"
        

    x = df_sel.query(signal)[true_var]
    y = df_sel.query(signal)[reco_var]
    w = df_sel.query(signal)['totweight_intrinsic']
    
    if reco_var=='NeutrinoEnergy2': 
        y = y/1000
        

    #fig = plt.figure(figsize=(8, 5))
    fig = plt.figure(figsize=(11, 8))

    hout = plt.hist2d(x, y, bins, weights=w, cmap='OrRd', cmin=0.01)

    for i in range(len(bins)-1): # reco bins i (y axis)
        for j in range(len(bins)-1): # true bins j (x axis)
            if hout[0].T[i,j] > 0: 
                if hout[0].T[i,j]>zmax-10: 
                    col='white'
                else: 
                    col='black'
                    
                binx_centers = hout[1][j]+(hout[1][j+1]-hout[1][j])/2
                biny_centers = hout[2][i]+(hout[2][i+1]-hout[2][i])/2
                        
                plt.text(binx_centers, biny_centers, round(hout[0].T[i,j], 1), 
                    color=col, ha="center", va="center", fontsize=14)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    cbar = plt.colorbar()
    cbar.set_label('$\\nu$ / '+pot+' POT', fontsize=15)
    plt.xlabel('true '+lbl, fontsize=15)
    plt.ylabel('reco '+lbl, fontsize=15)

    plt.title(title, fontsize=15)
        
    cbar.ax.tick_params(labelsize=14)
    plt.tight_layout()
    if save: 
        plt.savefig(plots_path+horn+"/"+horn+"_smear.pdf", transparent=True, bbox_inches='tight') 

    plt.show()
    
    
    # now do the normalized version ( normalize truth columns to 1 )
    
    norm_array = hout[0].T
    
    # for each truth bin (column): 
    for col in range(len(bins)-1): 
        
        # sum all the reco bins (rows) 
        #reco_events_in_column = hout[0][0][col] + hout[0][1][col] + hout[0][2][col] + ....
        
        reco_events_in_column = [ norm_array[row][col] for row in range(len(bins)-1) ]
        tot_reco_events = np.nansum(reco_events_in_column)
        #print('total reco events in col = '+str(col)+' '+str(tot_reco_events))
        
        # replace with normalized value 
        for row in range(len(bins)-1): 
            norm_array[row][col] =  norm_array[row][col] / tot_reco_events
            
    
    
    # now plot
    #fig = plt.figure(figsize=(9, 5.5))
    fig = plt.figure(figsize=(13, 9))
    plt.pcolor(bins, bins, norm_array, cmap='OrRd', vmax=1)
    
    # Loop over data dimensions and create text annotations.
    for i in range(len(bins)-1): # reco bins (rows)
        for j in range(len(bins)-1): # truth bins (cols)
            if norm_array[i][j]>0: 
                
                if norm_array[i][j]>0.7: 
                    col = 'white'
                else: 
                    col = 'black'
                    
                binx_centers = hout[1][j]+(hout[1][j+1]-hout[1][j])/2
                biny_centers = hout[2][i]+(hout[2][i+1]-hout[2][i])/2
                
                plt.text(binx_centers, biny_centers, round(norm_array[i][j], 2), 
                     ha="center", va="center", color=col, fontsize=12)
                
    plt.title(title, fontsize=15)
    
    plt.xticks(fontsize=14)
    plt.gca().xaxis.tick_bottom()
    plt.yticks(fontsize=14)
    cbar = plt.colorbar()
    cbar.set_label(pot+' POT (column normalized)', fontsize=15)
    cbar.ax.tick_params(labelsize=14)
    plt.xlabel('true '+lbl, fontsize=15)
    plt.ylabel('reco '+lbl, fontsize=15)
    
    
    #print(norm_array)
    
    if save: 
        plt.savefig(plots_path+horn+"/"+horn+"_smear_norm.pdf", transparent=True, bbox_inches='tight') 
    
    
    plt.show()
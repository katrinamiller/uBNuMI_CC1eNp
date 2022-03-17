## UPDATE THE PLOTS PATH BEFORE SAVING ## 

import math
import warnings

import scipy.stats
from scipy.stats import norm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec

import importlib

import selection_functions 
importlib.reload(selection_functions)

import top 
from top import *


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
    
    print('update!')

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
    
    print('update!')
    
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
def truth_and_eff(true_var, bins, xlower, xupper, cut, datasets, isrun3, pot=None, ymax1=None, ymax2=None, xlbl=None, save=False):
    
    print('update!')
    
    if isrun3: 
        horn = 'rhc'
    else: 
        horn = 'fhc'
        
    infv = datasets[0]
    cosmic = datasets[2]
    
    ## ~~~~~~~~~~~~~~~~~~ COMPUTE EFFICIENCY ~~~~~~~~~~~~~~~~~~~~~~ ##
    gen = sf.generated_signal(isrun3, true_var, bins, xlower, xupper) # scaled to overlay 
    # gen = plt.hist(df_infv.query(signal)[true_var], bins, weights=df_infv.query(signal)['totweight_overlay'])
    
    infv_selected = infv.query(cut)
    sel = plt.hist(infv_selected.query(signal)[true_var], bins, weights=infv_selected.query(signal)['totweight_overlay'])
    plt.close()
    
    eff = [i/j for i, j in zip(sel[0], gen)]
    eff_err = []
    for i in range(len(eff)): 
        eff_err.append(math.sqrt( (eff[i]*(1-eff[i]))/gen[i] ) )
    
    bc = 0.5*(sel[1][1:]+sel[1][:-1])
    x_err = []
    for i in range(len(sel[1])-1): 
        x_err.append((sel[1][i+1]-sel[1][i])/2)
        
        
    
    ## ~~~~~~~~~~~~~~~~~~~~~~~~ PLOT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##

    fig, ax1 = plt.subplots(figsize=(8, 5))
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(linestyle=':')

    color = 'tab:orange'
    ax1.set_xlabel(xlbl, fontsize=15)
    
    ax1.set_ylabel('$\\nu$ / $2.0x10^{20}$ POT', color=color, fontsize=15)
    
    ax1.tick_params(axis='y', labelcolor=color)
    
    if ymax1: 
        ax1.set_ylim(0, ymax1)

    ax1.hist(infv_selected.query(signal)[true_var], bins, range=[0, 4], color='orange', 
         weights=infv_selected.query(signal)['totweight'])

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    ax2.set_ylabel('Efficiency', color='seagreen', fontsize=15)  # we already handled the x-label with ax1
    ax2.tick_params(axis='y', labelcolor='seagreen', labelsize=14) 

    ax2.errorbar(bc, eff, xerr=x_err, yerr=eff_err, fmt='o', color='seagreen', ecolor='seagreen', markersize=3) 
    
    if ymax2: 
        ax2.set_ylim(0, ymax2)
        
    if save: 
        plt.savefig("/uboone/data/users/kmiller/searchingfornues_v33/v08_00_00_33/plots/"+horn+"/"+horn+"_"+true_var+"truth_eff.pdf", 
           transparent=True, bbox_inches='tight')    
        
    plt.tight_layout()
    plt.show()
    
##################################################################################################
# plot efficiency as a function of true bin 
# for both datasets, combined
def comb_truth_and_eff(true_var, bins, df_infv, df_sel, xlbl, save):
    
    print('update!')
    
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
# scale to DATA
# UPDATED 1/24/22
def smear_matrix(true_var, reco_var, bins, isrun3, selected_signal, 
                 uv_weights=None, zmax=20, lbl=None, save=False, plot=False, eff=False, norm=None):

    if not isrun3: 
        title = "FHC Selected Signal"
        
    else: 
        title = "RHC Selected Signal"

        
    ######################################################   
    ################# SMEARING ONLY ######################
    
    fig = plt.figure(figsize=(11, 8))
    
    if uv_weights is not None: 
        hout = plt.hist2d(selected_signal[true_var], selected_signal[reco_var], bins, 
                              weights=uv_weights, cmap='OrRd', cmin=0.01)
    else: # just do the CV 
        hout = plt.hist2d(selected_signal[true_var], selected_signal[reco_var], bins, 
                              weights=selected_signal.totweight_data, cmap='OrRd', cmin=0.01)
        
    smear_array = hout[0].T #[ [0]*(len(bins)-1) for x in range(len(bins)-1) ]
    
    if norm=='column': 
    # normalize the column (truth bin)
        for col in range(len(bins)-1): 

            reco_events_in_column = [ hout[0].T[row][col] for row in range(len(bins)-1) ]
            tot_reco_events = np.nansum(reco_events_in_column)
        
            for row in range(len(bins)-1): 
                smear_array[row][col] =  hout[0].T[row][col] / tot_reco_events
                
    elif norm=='row': # normalize the row (reco bin)
        for row in range(len(bins)-1): 
            
            truth_events_in_row = [ hout[0].T[row][col] for col in range(len(bins)-1) ] 
            tot_truth_events = np.nansum(truth_events_in_row)
            
            for col in range(len(bins)-1): 
                smear_array[row][col] =  hout[0].T[row][col] / tot_truth_events

    if plot: 
        for i in range(len(bins)-1): # reco bins i (y axis)
            for j in range(len(bins)-1): # true bins j (x axis)
                if smear_array[i][j]>zmax: 
                    col='white'
                else: 
                    col='black'

                binx_centers = hout[1][j]+(hout[1][j+1]-hout[1][j])/2
                biny_centers = hout[2][i]+(hout[2][i+1]-hout[2][i])/2

                if not np.isnan(smear_array[i][j]):
                    plt.text(binx_centers, biny_centers, round(smear_array[i][j], 1), 
                            color=col, ha="center", va="center", fontsize=14)
         
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        cbar = plt.colorbar()
        #cbar.set_label('$\\nu$ / '+pot+' POT', fontsize=15)

        if lbl: 
            plt.xlabel('True '+lbl, fontsize=15)
            plt.ylabel('Reco '+lbl, fontsize=15)

        plt.title(title+' - Smearing '+str(parameters(isrun3)['beamon_pot'])+' POT', fontsize=15)

        cbar.ax.tick_params(labelsize=14)
        plt.tight_layout()
        if save: 
            plt.savefig(plots_path+horn+"/"+horn+"_smear.pdf", transparent=True, bbox_inches='tight') 

        plt.show()
    
    else: 
        plt.close()

    ##################################################################
    ##### SMEARING & EFFICIENCY ######################################
    
    smear_eff_array = [ [0]*(len(bins)-1) for x in range(len(bins)-1) ]
    gen = []
    
    if eff: 
    
        # normalize by the GENERATED events in a true bin: 
        gen = generated_signal(isrun3, true_var, bins, bins[0], bins[-1], cuts=None) # scales to standard overlay 
        pot_scale = parameters(isrun3)['beamon_pot']/parameters(isrun3)['overlay_pot']
        gen = [x*pot_scale for x in gen] 

    
        # for each truth bin (column): 
        for col in range(len(bins)-1): 
            
            for row in range(len(bins)-1): 
                smear_eff_array[row][col] =  hout[0].T[row][col] / gen[col] 
            
            
        # now plot
        if plot: 
            fig = plt.figure(figsize=(13, 9))
            plt.pcolor(bins, bins, smear_eff_array, cmap='OrRd', vmax=0.35)

            # Loop over data dimensions and create text annotations.
            for i in range(len(bins)-1): # reco bins (rows)
                for j in range(len(bins)-1): # truth bins (cols)
                    if smear_eff_array[i][j]>0: 

                        if smear_eff_array[i][j]>0.3: 
                            col = 'white'
                        else: 
                            col = 'black'

                        binx_centers = hout[1][j]+(hout[1][j+1]-hout[1][j])/2
                        biny_centers = hout[2][i]+(hout[2][i+1]-hout[2][i])/2

                        plt.text(binx_centers, biny_centers, round(smear_eff_array[i][j], 2), 
                             ha="center", va="center", color=col, fontsize=12)

            plt.title(title + ' - Smearing & Efficiency '+str(parameters(isrun3)['beamon_pot'])+' POT', fontsize=15)

            plt.xticks(fontsize=14)
            plt.gca().xaxis.tick_bottom()
            plt.yticks(fontsize=14)
            cbar = plt.colorbar()
            #cbar.set_label(pot+' POT (column normalized)', fontsize=15)
            cbar.ax.tick_params(labelsize=14)

            if lbl: 
                plt.xlabel('True '+lbl, fontsize=15)
                plt.ylabel('Reco '+lbl, fontsize=15)

            if save: 
                plt.savefig(plots_path+horn+"/"+horn+"_smear_eff.pdf", transparent=True, bbox_inches='tight') 

            plt.show()
            
        else: 
            plt.close()
        

    
    ########################################################################
    
    smear_dict = {
        "true_generated_counts" : gen, # number of events in each truth bin 
        "smear_array" : smear_array, 
        "smear_eff_array" : smear_eff_array # additional efficiency scaling , i.e. n_ij / n_j 
    }

    return smear_dict
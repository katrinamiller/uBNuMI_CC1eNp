
import math
import warnings
import importlib 

import scipy.stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec

import matplotlib.pylab as pylab
from sklearn.model_selection import train_test_split
import pickle
import xgboost as xgb
import csv
import uproot

import NuMIDetSys

importlib.reload(NuMIDetSys)
NuMIDetSysWeights = NuMIDetSys.NuMIDetSys()

import xsec_functions
importlib.reload(xsec_functions)
from xsec_functions import smear_matrix

import top 
importlib.reload(top)
from top import *

########################################################################
# vary the number of dirt interactions in the selected evt rate by 100% (assess the uncertainty in the background)
def dirt_unisim(xvar, bins, cv_total, cv_dirt, percent_variation, isrun3=None, plot=False, x_label=None, title=None):
    
    # cv total is the event rate (either before or after background subtraction) 
    
    if isrun3: 
        data_pot = parameters(isrun3)['beamon_pot'] 
    
    if x_label: 
        x = x_label
    else: 
        x = str(xvar)  

    # create & plot the variation
    uv_dirt = [count+(count*percent_variation) for count in cv_dirt]
    
    if plot: 
        
        bincenters = 0.5*(np.array(bins)[1:]+np.array(bins)[:-1])

        fig = plt.figure(figsize=(8, 5))
        
        plt.hist(bincenters, bins, histtype='step', range=[bins[0], bins[-1]], color='black', linewidth=2, 
                 weights=cv_dirt)
        plt.hist(bincenters, bins, histtype='step', range=[bins[0], bins[-1]], color='cornflowerblue', 
                 weights=uv_dirt)
        
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        plt.xlabel('Reco '+x, fontsize=15)
        if title: 
            plt.title(title, fontsize=16)

        if isrun3: 
            plt.ylabel(str(data_pot) + ' POT', fontsize=15)

        plt.show()
        
    cov_dict = calcCov(xvar, bins, cv_dirt, cv_total, [uv_dirt], 
                       plot=plot, axis_label='Reco '+x, isrun3=isrun3)
    
    #sys_err = [np.sqrt(x) for x in np.diagonal(cov_dict['cov'])]
    
    cov_dict['variations'] = uv_dirt
    
    return cov_dict

########################################################################
def pot_unisims(xvar, ncv, bins, percent_variation, isrun3, plot=False, x_label=None, title=None): 

    data_pot = str(parameters(isrun3)['beamon_pot'])
    
    if x_label: 
        x = x_label
    else: 
        x = str(xvar)
    
    # create & plot the variations
    up = [count+count*percent_variation for count in ncv]
    dn = [count-count*percent_variation for count in ncv]
        
    #if bkgd_cv_counts: # subtract off the CV background event rate
        
    #    print('Implementing background subtraction ....')

    #    up = [a-b for a,b in zip(up,bkgd_cv_counts)]
    #    dn = [a-b for a,b in zip(dn,bkgd_cv_counts)]
    #    cv = [a-b for a,b in zip(ncv,bkgd_cv_counts)]
    
    #else: 
    
    cv = ncv
     
    uni_counts = [up, dn]
    
    if plot: 
        
        bincenters = 0.5*(np.array(bins)[1:]+np.array(bins)[:-1])

        fig = plt.figure(figsize=(8, 5))
        
        for uv in uni_counts: 
            plt.hist(bincenters, bins, histtype='step', range=[bins[0], bins[-1]], color='cornflowerblue', weights=uv)

        plt.hist(bincenters, bins, histtype='step', range=[bins[0], bins[-1]], color='black', linewidth=2, weights=cv)
        
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        plt.xlabel('Reco '+x, fontsize=15)
        if title: 
            plt.title(title, fontsize=16)

        plt.ylabel(data_pot + ' POT', fontsize=15)

        plt.show()
    
    cov_dict = calcCov(xvar, bins, cv, cv, uni_counts, plot=plot, axis_label='Reco '+x, pot=data_pot, isrun3=isrun3)
    
    #sys_err = [np.sqrt(x) for x in np.diagonal(cov_dict['cov'])]
    
    cov_dict['variations'] = uni_counts
    
    return cov_dict

########################################################################
# calculate systematic error based on variation weights assigned to each bin 
# return the fractional systematic uncertainty for each bin 
# scales to DATA POT - use pot parameter to scale to something else 
def calcDetSysError(var, bin_edges, file, ISRUN3, intrinsic=False, plot=False, plot_cov=False, save=False, axis_label=None,
                   moreStats=False, xsec_units=False, signal_only=False, background_only=False): 

    data_pot = parameters(ISRUN3)['beamon_pot']
    
    # get variation counts & CV counts - from NuMIDetSys.py (normalized to DATA POT)
    uni_counts, ncv, weights = NuMIDetSysWeights.ratio_to_CV(var, bin_edges, file, ISRUN3, intrinsic=intrinsic, moreStats=moreStats, 
                                                            data_pot=data_pot, xsec_units=xsec_units)

    if plot: # these are scaled to det. sys. CV POT, unless pot parameter is specified
        NuMIDetSysWeights.plot_variations(var, bin_edges, file, ISRUN3, pot=data_pot, 
                                          intrinsic=intrinsic, axis_label=axis_label, save=save, moreStats=moreStats, xsec_units=xsec_units)
    
    # compute covariance & correlation 
    cov_dict = calcCov(var, bin_edges, ncv, uni_counts, '', unisim=True, plot=plot_cov, save=save, 
                       axis_label=axis_label, pot=str(data_pot)+' POT', isrun3=ISRUN3)
    sys_err = [np.sqrt(x) for x in np.diagonal(cov_dict['cov'])]# scaled to det. sys. CV POT - 
    
    frac = [] # list of all separate uncertainty contributions
    for k in cov_dict['individual_uncertainty']: 
        frac.append( [y/z for y,z in zip(k, ncv)] ) # turn into a percentage - FIX - SHOULD BE OVER TOTAL EVENT RATE? 
    
    d = {
        'sys_err' : sys_err, 
        'ncv' : ncv,
        'percent_error' : [a/b for a,b in zip(sys_err, ncv)], # FIX - SHOULD BE OVER TOTAL EVENT RATE? 
        'covariance' : cov_dict['cov'], 
        "fractional_covariance" : cov_dict['frac_cov'], 
        'correlation' : cov_dict['cor'], 
        'individual_uncertainty' : frac 
    }

    return d

########################################################################
# grab the event counts for systematic variations 
# for input into calcCov
def plotSysVariations(reco_var, true_var, bins, xlow, xhigh, cuts, datasets, sys_var, universes, isrun3, background_subtraction=None, plot=False, save=False, axis_label=None, ymax=None, pot=None, text=None, xtext=None, ytext=None, title=None, x_ticks=None): 

    
    ############################################################
        
    bin_widths = [ round(bins[i+1]-bins[i], 2) for i in range(len(bins)-1) ] 
    
    cv_weight = 'totweight_data'
    
    if background_subtraction: 
        print("Implementing background subtraction .... ")
        
        if sys_var=='weightsGenie':
            cv_generated_signal, nu_generated = generated_signal(isrun3, true_var, bins, bins[0], bins[-1], weight='totweight_data', genie_sys=sys_var)
            print("TOTAL CV GENERATED SIGNAL EVENTS (TRUE) = ", cv_generated_signal)
            
        elif sys_var=='weightsGenieUnisim':
            if title=='RPA': 
                cv_generated_signal, nu_generated = generated_signal(isrun3, true_var, bins, bins[0], bins[-1], weight='totweight_data', genie_sys=['knobRPAup', 'knobRPAdn'])
            else: 
                cv_generated_signal, nu_generated = generated_signal(isrun3, true_var, bins, bins[0], bins[-1], weight='totweight_data', genie_sys='knob'+title+'up')
                
    plots_path = parameters(isrun3)['plots_path']
    
    ############################################################

    if cuts == '': 
        infv = datasets['infv'].copy()
        outfv = datasets['outfv'].copy()
        
    else: 
        infv = datasets['infv'].copy().query(cuts)
        outfv = datasets['outfv'].copy().query(cuts)
    
    # total CV event rate (S+B)
    nu_selected = pd.concat([infv.copy(), outfv.copy()], ignore_index=True, sort=True) 
    
    ncv, bcv, pcv = plt.hist(nu_selected[reco_var], bins, weights=nu_selected[cv_weight])
    plt.close()

    if background_subtraction: # CV background event rate -- use when unfolding only 
        
        nu_selected_background = nu_selected.query(not_signal)
        
        ncv_bkgd = plt.hist(nu_selected_background[reco_var], bins, weights=nu_selected_background[cv_weight])[0]
        plt.close()
        
        #ncv_bkgd = [4.351890811739091, 2.3529093590530237,2.9222633345981994,4.481129901616696,7.2883077383370685]
        
        ncv = [a-b for a,b in zip(ncv, ncv_bkgd)] 
        

    ############################################################
    # histogram bin counts for all universes

    uni_counts = []
    
    if isinstance(universes, list): # in the case of unisims --> indices of the universes to vary in the weightsList 
        universes = universes
        line_width = 1
        sim = 'unisim'
        
    elif isinstance(universes, int): # in the case of multisims --> number of multisims
        universes = range(universes)
        line_width = 0.5
        sim = 'multisim'

    for u in universes: 
            
        # multiply CV weights with sys weight of universe u 
        sys_weight = list(nu_selected[sys_var].str.get(u))
        
        #########################################
        ######## NON GENIE UNCERTAINTIES ########
        
        if not "Genie" in sys_var: 

            # get the UV weights
            nu_selected['weight_sys'] = [ (x*y) for x, y in zip(sys_weight, nu_selected[cv_weight]) ]
            
            # get the binned UV event rate 
            n, b, p = plt.hist(nu_selected[reco_var], bins, weights=nu_selected['weight_sys'])
            plt.close()
            
            # background subtract (if necessary)
            if background_subtraction: # subtract off the CV background event rate
                n = [a-b for a,b in zip(n, ncv_bkgd)]

            # add to array of UV counts
            uni_counts.append(n)
            
            
        #########################################
        ###### GENIE GETS SPECIAL HANDLING ######
        
        else: 
            
            # get the UV weights
            if sys_var=='weightsGenie':  # replace the tune weight 
                nu_selected['weight_sys'] = [ x*y for x, y in zip(sys_weight, nu_selected[cv_weight]/nu_selected['weightTune']) ]  

            elif sys_var=='weightsGenieUnisim': 
                if 'scc' in title: 
                    nu_selected['weight_sys'] = [ x*y for x, y in zip(sys_weight, nu_selected[cv_weight]) ]
                else: 
                    nu_selected['weight_sys'] = [ x*y for x, y in zip(sys_weight, nu_selected[cv_weight]/nu_selected['weightTune']) ] 
                    
                    
            # get the binned UV event rate the usual way 
            n, b, p = plt.hist(nu_selected[reco_var], bins, weights=nu_selected['weight_sys'])
            plt.close()
                
            # background subtract (if necessary)
            if background_subtraction: 
                
                n = [a-b for a,b in zip(n, ncv_bkgd)]
                
                # divide out the systematic event rate change
                if sys_var=='weightsGenie': 
                    sys_weight_generated = list(nu_generated[sys_var].str.get(u)) # UV weights for generated signal 
                    nu_generated['weight_sys'] = [ x*y for x,y in zip(sys_weight_generated,nu_generated[cv_weight]/nu_generated['weightTune']) ]  
                
                elif sys_var=='weightsGenieUnisim': 
                    
                    # RPA gets special treatment as a double unisim
                    if u==1: 
                        sys_weight_generated = list(nu_generated['knob'+title+'dn'])
                    
                    else: 
                        sys_weight_generated = list(nu_generated['knob'+title+'up']) 
                        
                    sys_weight_generated = [1 if x!= x else x for x in sys_weight_generated]
                    sys_weight_generated = [1 if x==np.inf else x for x in sys_weight_generated]
                    print(sum(sys_weight_generated))

                    if 'scc' in title: 
                        nu_generated['weight_sys'] = [ x*y for x,y in zip(sys_weight_generated,nu_generated[cv_weight]) ]
                    else: 
                        nu_generated['weight_sys'] = [ x*y for x,y in zip(sys_weight_generated,nu_generated[cv_weight]/nu_generated['weightTune']) ]  
                  
                
                uv_generated_signal = plt.hist(nu_generated[true_var], bins, weights=nu_generated['weight_sys'])[0]
                plt.close()
                
                n = [ n[i]*(cv_generated_signal[i]/uv_generated_signal[i]) for i in range(len(n)) ]

            # add to array of UV counts
            uni_counts.append(n)
                
        #########################################
        #########################################

    ############################################################
    if plot:  
        
        fig = plt.figure(figsize=(8, 5))     

        # plot the systematic universes first 
        counter = 0
        
        for u in universes: 
            
            if counter==0: 
                plt.hist(0.5*(np.array(bins)[1:]+np.array(bins)[:-1]), bins,
                                 weights=uni_counts[counter], histtype='step', color='cornflowerblue', linewidth=line_width, label='UV')
            else: 
                plt.hist(0.5*(np.array(bins)[1:]+np.array(bins)[:-1]), bins,
                                 weights=uni_counts[counter], histtype='step', color='cornflowerblue', linewidth=line_width)
                    
            counter += 1
            
        #plt.hist(nu_selected[reco_var], bins, weights=nu_selected[cv_weight], label='CV', color='black', histtype='step',
        #                  linewidth=2)
            
        plt.hist(0.5*(np.array(bins)[1:]+np.array(bins)[:-1]), bins,
                 weights=ncv, label='CV', color='black', histtype='step', linewidth=2)

        if title: 
            plt.title(title, fontsize=15)
        else: 
            plt.title(sys_var, fontsize=15)

        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        
        plt.legend(fontsize=13, frameon=False)
        
        if axis_label is not None: 
            plt.xlabel(axis_label, fontsize=15)
        else: 
            plt.xlabel(reco_var, fontsize=15)
        
        if pot: 
            plt.ylabel("$\\nu$ / "+pot, fontsize=15)
        
        plt.xlim(xlow, xhigh)
        
        if ymax is not None: 
            plt.ylim(0, ymax)
            
        if x_ticks: 
            plt.xticks(x_ticks, fontsize=14)
            
        if text: 
            plt.text(xtext, ytext, text, fontsize='xx-large')
            
        if save: 
            plt.savefig(plots_path+reco_var+"_"+sys_var+".pdf", transparent=True, bbox_inches='tight') 
            print('saving to: '+plots_path)
            
        plt.show()

    
    return ncv, uni_counts # where ncv is the total neutrino CV event rate (i.e. not including EXT) OR background subtracted nu evt rate
    
########################################################################
# compute covariance & correlation matrices 
def calcCov(var, bins, ncv_nu, ncv_total, uni_counts, plot=False, save=False, axis_label=None, pot=None, isrun3=False, xticks=None, xhigh=None): 
    
    # ncv nu is the neutrino event rate -- i.e. what gets varied in the systematics 
    # ncv total is the total event rate -- (MC + EXT or estimated signal)
    # when background subtracting these two are the same  
    
    # compute the cov matrix 
    cov = [ [0]*(len(bins)-1) for x in range(len(bins)-1) ]
    frac_cov = [ [0]*(len(bins)-1) for x in range(len(bins)-1) ]
    cor = [ [0]*(len(bins)-1) for x in range(len(bins)-1) ]
    
    N = len(uni_counts)
    print('number of universes = ', N)

    #####################################################
    
    for k in range(N): 
        
        uni = uni_counts[k]

        for i in range(len(bins)-1): 

            cvi = ncv_nu[i]
            uvi = uni[i]


            for j in range(len(bins)-1): 
                
                cvj = ncv_nu[j]
                uvj = uni[j]
        
                c = ((uvi - cvi)*(uvj - cvj)) / N

                cov[i][j] += c
                
                
                if ncv_total[i]*ncv_total[j] != 0: 
                    frac_cov[i][j] += c/(ncv_total[i]*ncv_total[j])
                    #frac_cov[i][j] = c/(ncv_total[i]*ncv_total[j])
            
    #####################################################
    
    if plot: 
        fig = plt.figure(figsize=(10, 6))
        
        plt.pcolor(bins, bins, cov, cmap='OrRd', edgecolors='k')
            
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=14)
        if pot: 
            cbar.set_label(label="$\\nu^{2}$ / "+pot+"$^{2}$", fontsize=15)
        
        plt.xticks(xticks, fontsize=13)
        plt.yticks(xticks,fontsize=13)
        
        if xhigh: 
            plt.xlim(bins[0], xhigh)
            plt.ylim(bins[0], xhigh)
            
        else: 
            plt.xlim(bins[0], bins[-1])
            plt.ylim(bins[0], bins[-1])

        if axis_label is not None: 
            plt.xlabel(axis_label, fontsize=15)
            plt.ylabel(axis_label, fontsize=15)
        else: 
            plt.xlabel(var, fontsize=15)
            plt.ylabel(var, fontsize=15)

        plt.title('Covariance Matrix', fontsize=16)
        
        if save: 
            plt.savefig(save+var+"_cov.pdf", transparent=True, bbox_inches='tight') 
            print('saving to: '+save)
        plt.show()
        
        ##################################
        # fractional covariance 
        fig = plt.figure(figsize=(10, 6))
        
        plt.pcolor(bins, bins, frac_cov, cmap='OrRd', edgecolors='k')#, vmin=0, vmax=.03)
            
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=14)
        
        
        if pot: 
            cbar.set_label(label="$\\nu^{2}$ / "+pot+"$^{2}$", fontsize=15)

        if axis_label is not None: 
            plt.xlabel(axis_label, fontsize=15)
            plt.ylabel(axis_label, fontsize=15)
        else: 
            plt.xlabel(var, fontsize=15)
            plt.ylabel(var, fontsize=15)
            
        plt.xticks(xticks, fontsize=13)
        plt.yticks(xticks,fontsize=13)
        
        if xhigh: 
            plt.xlim(bins[0], xhigh)
            plt.ylim(bins[0], xhigh)
            
        else: 
            plt.xlim(bins[0], bins[-1])
            plt.ylim(bins[0], bins[-1])

        plt.title('Fractional Covariance Matrix', fontsize=16)
        
        if save: 
            plt.savefig(save+var+"_frac_cov.pdf", transparent=True, bbox_inches='tight') 
        plt.show()
        
    #####################################################    
    # compute the corr matrix 

    for i in range(len(cov)): 
        for j in range(len(cov[i])): 
            
            #print(i, j, cov[i][j], cov[i][i], cov[j][j])

            if np.sqrt(cov[i][i])*np.sqrt(cov[j][j]) != 0: 
                cor[i][j] = cov[i][j] / (np.sqrt(cov[i][i])*np.sqrt(cov[j][j]))
            
            #print(cor[i][j])
    
    #####################################################
    
    if plot: 
        fig = plt.figure(figsize=(10, 6))

        plt.pcolor(bins, bins, cor, cmap='OrRd', edgecolors='k', vmin=-1, vmax=1)
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=14)
    

        if axis_label is not None: 
            plt.xlabel(axis_label, fontsize=15)
            plt.ylabel(axis_label, fontsize=15)
        else: 
            plt.xlabel(var, fontsize=15)
            plt.ylabel(var, fontsize=15)
            
        plt.xticks(xticks, fontsize=13)
        plt.yticks(xticks,fontsize=13)
        
        if pot: 
            cbar.set_label(label="$\\nu^{2}$ / "+pot+"$^{2}$", fontsize=15)
        
        if xhigh: 
            plt.xlim(bins[0], xhigh)
            plt.ylim(bins[0], xhigh)
            
        else: 
            plt.xlim(bins[0], bins[-1])
            plt.ylim(bins[0], bins[-1])
            
        plt.title('Correlation Matrix', fontsize=16)
        if save: 
            plt.savefig(save+var+"_cor.pdf", transparent=True, bbox_inches='tight') 
        plt.show()
        
    #####################################################
    
    # sys_err = [np.sqrt(x) for x in np.diagonal(cov)]
    # percent error = [y/z for y,z in zip(sys_err, ncv)] # w.r.t. to whatever event rate is being used (total or background subtracted)
        
    dictionary = {
        'cov' : cov, 
        'frac_cov' : frac_cov, 
        'cor' : cor,
        'fractional_uncertainty' : np.sqrt(np.diag(frac_cov))
    }
           
    return dictionary

########################################################################
# plot the full covariance & correlation matrices 
def plotFullCov(frac_cov_dict, var, cv, bins, xlow, xhigh, x_ticks=None, save=False, axis_label=None, isrun3=False, pot=None): 
    
    print('Note: Input must be fractional covariance matrices !')
    
    #plots_path = parameters(isrun3)['plots_path']
    
    keys = list(frac_cov_dict.keys())
    tot_frac_cov = [ [0]*(len(bins)-1) for x in range(len(bins)-1) ]
    
    # add elements of the same index together 
    for source in keys: 
        tot_frac_cov = [ [x+y for x,y in zip(a,b)] for a,b in zip(tot_frac_cov, frac_cov_dict[source]) ]
    
    # plot 
    fig = plt.figure(figsize=(10, 6))
    
    plt.pcolor(bins, bins, tot_frac_cov, cmap='OrRd', edgecolors='k')
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=14)
    if pot: 
        cbar.set_label(label="$\\nu^{2}$ / "+pot+"$^{2}$", fontsize=15)
        
    if x_ticks: 
        plt.xticks(x_ticks, fontsize=14)
        plt.yticks(x_ticks, fontsize=14)
    else: 
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
    
    
    plt.xlim(xlow, xhigh)
    plt.ylim(xlow, xhigh)

    if axis_label is not None: 
        plt.xlabel(axis_label, fontsize=15)
        plt.ylabel(axis_label, fontsize=15)
    else: 
        plt.xlabel(var, fontsize=15)
        plt.ylabel(var, fontsize=15)

    plt.title('Fractional Covariance', fontsize=15)
    
    #if save: 
    #    plt.savefig(save+var+"_tot_frac_cov.pdf", transparent=True, bbox_inches='tight') 
    plt.show()
    
    
    # convert back to absolute covariance units 
    abs_cov = [ [0]*(len(bins)-1) for x in range(len(bins)-1) ]

    for i in range(len(bins)-1): 
        for j in range(len(bins)-1): 
            abs_cov[i][j] = tot_frac_cov[i][j] * cv[i] * cv[j]
            
    
    # plot 
    #fig = plt.figure(figsize=(13, 9))
    fig = plt.figure(figsize=(10, 6))
    
    plt.pcolor(bins, bins, abs_cov, cmap='OrRd', edgecolors='k')
    cbar = plt.colorbar()
    if pot: 
        cbar.set_label(label="$\\nu^{2}$ / "+pot+"$^{2}$", fontsize=15)
    cbar.ax.tick_params(labelsize=14)
    
    if x_ticks: 
        plt.xticks(x_ticks, fontsize=14)
        plt.yticks(x_ticks, fontsize=14)
    else: 
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
    
    
    plt.xlim(xlow, xhigh)
    plt.ylim(xlow, xhigh)

    if axis_label is not None: 
        plt.xlabel(axis_label, fontsize=15)
        plt.ylabel(axis_label, fontsize=15)
    else: 
        plt.xlabel(var, fontsize=15)
        plt.ylabel(var, fontsize=15)

    plt.title('Absolute Covariance', fontsize=15)
    
    if save: 
        plt.savefig(save+var+"_tot_abs_cov.pdf", transparent=True, bbox_inches='tight') 
    plt.show()
    
    
    return tot_frac_cov, abs_cov
########################################################################
# concat run, subrun, evt number for random seed 
# converted from Afro's ROOT functions 
def ConcatRunSubRunEvent(run, subrun, event): 
    
    # make sure the numbers are not too big 
    ModRun = run/1E6
    ModSubRun = subrun/1E6
    ModEvent = event/1E6
    
    if abs(ModRun)>1: 
        run = abs(ModRun)
        
    if abs(ModSubRun)>1: 
        subrun = abs(ModSubRun)
        
    if abs(ModEvent)>1: 
        event = abs(ModEvent)
    
    # convert integers to string 
    srun = str(run)
    ssubrun = str(subrun)
    sevent =  (str(event))
    
    # concat subrun & event. don't add run bc it makes number too long for storage
    s = ssubrun + sevent
    # print(srun + "  " + ssubrun + "  " + sevent)

    # convert back to integer 
    seed = int(s)
    # print(seed)
    
    return seed

########################################################################
# random number generator for bootstrapping method 
def PoissonRandomNumber(seed, mean=1.0, size=None): 
    
    # set seed based on run,subrun,evt 
    np.random.seed(seed)
    # print('seed='+str(seed))
    
    # generate weight using poisson distribution with mean 1 
    weight_poisson = np.random.poisson(lam=mean, size=size)
    # print(weight_poisson)
    
    return weight_poisson
    
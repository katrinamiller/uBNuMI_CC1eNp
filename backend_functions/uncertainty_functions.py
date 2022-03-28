
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
# vary the number of dirt interactions in the selected evt rate by 100% 
def dirt_unisim(xvar, bins, cv_total, cv_dirt, percent_variation, isrun3, plot=False, x_label=None, title=None, bkgd_cv_counts=None):
    
    data_pot = parameters(isrun3)['beamon_pot'] 
    
    if x_label: 
        x = x_label
    else: 
        x = str(xvar)  

    # create & plot the variation
    uv_dirt = [count+count*percent_variation for count in cv_dirt]
    uv_total = [ (a-b)+c for a,b,c in zip(cv_total, cv_dirt, uv_dirt) ] # remove CV dirt evts from CV total & replace with UV dirt evts
    
    if bkgd_cv_counts: # subtract off the CV background event rate
        
        print('Implementing background subtraction ....')

        uv_total = [a-b for a,b in zip(uv_total,bkgd_cv_counts)]
        cv_total = [a-b for a,b in zip(cv_total,bkgd_cv_counts)]
    
    if plot: 
        
        bincenters = 0.5*(np.array(bins)[1:]+np.array(bins)[:-1])

        fig = plt.figure(figsize=(8, 5))
        
        plt.hist(bincenters, bins, histtype='step', range=[bins[0], bins[-1]], color='black', linewidth=2, weights=cv_total)
        plt.hist(bincenters, bins, histtype='step', range=[bins[0], bins[-1]], color='cornflowerblue', weights=uv_total)
        
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        plt.xlabel('Reco '+x, fontsize=15)
        if title: 
            plt.title(title, fontsize=16)

        plt.ylabel("$\\nu$ / $2.0 x 10^{20}$ POT", fontsize=15)

        plt.show()
        
    cov_dict = calcCov(xvar, bins, cv_total, [uv_total], plot=plot, axis_label='Reco '+x, pot=data_pot, isrun3=isrun3)
    
    sys_err = [np.sqrt(x) for x in np.diagonal(cov_dict['cov'])]
    
    return cov_dict

########################################################################
# make flat variation for the dirt systematics
def pot_unisims(xvar, ncv, bins, percent_variation, isrun3, plot=False, x_label=None, title=None, bkgd_cv_counts=None): 

    data_pot = parameters(isrun3)['beamon_pot'] 
    
    if x_label: 
        x = x_label
    else: 
        x = str(xvar)
    
    # create & plot the variations
    up = [count+count*percent_variation for count in ncv]
    dn = [count-count*percent_variation for count in ncv]
        
    if bkgd_cv_counts: # subtract off the CV background event rate
        
        print('Implementing background subtraction ....')

        up = [a-b for a,b in zip(up,bkgd_cv_counts)]
        dn = [a-b for a,b in zip(dn,bkgd_cv_counts)]
        cv = [a-b for a,b in zip(ncv,bkgd_cv_counts)]
    
    else: 
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

        plt.ylabel("$\\nu$ / $2.0 x 10^{20}$ POT", fontsize=15)

        plt.show()
    
    cov_dict = calcCov(xvar, bins, cv, uni_counts, plot=plot, axis_label='Reco '+x, pot=data_pot, isrun3=isrun3, title='POT counting')
    
    sys_err = [np.sqrt(x) for x in np.diagonal(cov_dict['cov'])]
    
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
def plotSysVariations(true_var, reco_var, bins, xlow, xhigh, cuts, datasets, sys_var, universes, isrun3, plot=False, 
                 save=False, axis_label=None, ymax=None, pot=None, text=None, xtext=None, ytext=None, background_subtraction=False, title=None): 
    
    ############################################################
        
    bin_widths = [ round(bins[i+1]-bins[i], 2) for i in range(len(bins)-1) ] 
    
    cv_weight = 'totweight_data'
    #print("Normalized to DATA POT (using "+cv_weight+")")
    
    if background_subtraction: 
        print("Implementing background subtraction .... ")
    
    plots_path = parameters(isrun3)['plots_path']
        
    if 'Genie' in sys_var:
        print('make sure to update this calculation for background subtraction on real/fake data!')
    
    ############################################################
    
    if cuts == '': 
        infv = datasets['infv'].copy()
        outfv = datasets['outfv'].copy()
        
    else: 
        infv = datasets['infv'].copy().query(cuts)
        outfv = datasets['outfv'].copy().query(cuts)
    
    # total CV event rate (S+B)
    nu_selected = pd.concat([infv.copy(), outfv.copy()],#, cosmic.copy()], 
                            ignore_index=True, sort=True) 
    
    ncv, bcv, pcv = plt.hist(nu_selected[reco_var], bins, weights=nu_selected[cv_weight])
    plt.close()
    
    #print('total CV event rate =', ncv)

    if background_subtraction: # CV background event rate -- use when unfolding only 
        
        nu_selected_background = nu_selected.query(not_signal)
        mc_stat_err = mc_stat_error(reco_var, bins, xlow, xhigh, [nu_selected.query(signal)]) # stat err should only be on CV signal 
        
        bkgd_ncv, bkgd_bcv, bkgd_pcv = plt.hist(nu_selected_background[reco_var], bins, weights=nu_selected_background[cv_weight])
        plt.close()
        
        #print('background CV event rate =', bkgd_ncv)
        
        ncv = [a-b for a,b in zip(ncv, bkgd_ncv)]
        
        # print('background-subtracted CV event rate =', ncv)
            
    else: 
        mc_stat_err = mc_stat_error(reco_var, bins, xlow, xhigh, [nu_selected])

    mc_stat_err_scaled = [x*y for x, y in zip(ncv, mc_stat_err)]

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

        if sys_var=='weightsGenie':  # replace the tune weight 
            nu_selected['weight_sys'] = [ x*y for x, y in zip(sys_weight, nu_selected[cv_weight]/nu_selected['weightTune']) ]  
            
        elif sys_var=='weightsGenieUnisim': # except for SCC variations
            if 'scc' in title: 
                nu_selected['weight_sys'] = [ x*y for x, y in zip(sys_weight, nu_selected[cv_weight]) ]
            else: 
                nu_selected['weight_sys'] = [ x*y for x, y in zip(sys_weight, nu_selected[cv_weight]/nu_selected['weightTune']) ]  

        else: 
            nu_selected['weight_sys'] = [ x*y for x, y in zip(sys_weight, nu_selected[cv_weight]) ]

        n, b, p = plt.hist(nu_selected[reco_var], bins, weights=nu_selected['weight_sys'])
        plt.close()

        if background_subtraction: # subtract off the CV background event rate
            n = [a-b for a,b in zip(n, bkgd_ncv)]

        uni_counts.append(n)

    ############################################################
    if plot:  
        
        fig = plt.figure(figsize=(8, 5))     

        # plot the systematic universes first 
        counter = 0
        
        for u in universes: 
            
            #if len(universes)==2: # symmetric unisims
            #    if counter==0: 
            #        plt.hist(0.5*(bcv[1:]+bcv[:-1]), bins, 
            #                     weights=uni_counts[counter], histtype='step', color='forestgreen', linewidth=line_width, label='UV (+)')
            #    elif counter==1: 
            #        plt.hist(0.5*(bcv[1:]+bcv[:-1]), bins, 
            #                     weights=uni_counts[counter], histtype='step', color='firebrick', linewidth=line_width, label='UV (-)')
                
                
            #else: # single unisims & multisims
            if counter==0: 
                plt.hist(0.5*(bcv[1:]+bcv[:-1]), bins, 
                                 weights=uni_counts[counter], histtype='step', color='cornflowerblue', linewidth=line_width, label='UV')
            else: 
                plt.hist(0.5*(bcv[1:]+bcv[:-1]), bins, 
                                 weights=uni_counts[counter], histtype='step', color='cornflowerblue', linewidth=line_width)
                    
            counter += 1


        x_err = [ round(abs(bcv[x+1]-bcv[x])/2, 3) for x in range(len(bcv)-1) ]
        plt.errorbar(0.5*(bcv[1:]+bcv[:-1]), ncv, yerr=mc_stat_err_scaled, xerr=x_err, fmt='none', color='black', linewidth=2, label='CV')

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
        
        if ymax is not None: 
            plt.ylim(0, ymax)
            
        if text: 
            plt.text(xtext, ytext, text, fontsize='xx-large')
            
        if save: 
            plt.savefig(plots_path+reco_var+"_"+sys_var+".pdf", transparent=True, bbox_inches='tight') 
            print('saving to: '+plots_path)
            
        plt.show()

    
    return ncv, uni_counts
    
########################################################################
# compute covariance & correlation matrices 
def calcCov(var, bins, ncv, uni_counts, plot=False, save=False, axis_label=None, pot=None, isrun3=False, title=None): 
    
    plots_path = parameters(isrun3)['plots_path']
    
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

            cvi = ncv[i]
            uvi = uni[i]


            for j in range(len(bins)-1): 
                
                cvj = ncv[j]
                uvj = uni[j]
        
                c = ((uvi - cvi)*(uvj - cvj)) / N

                cov[i][j] += c
                
                #if c<0: 
                #    print(title, i, uvi, cvi, j, uvj, cvj, c)
                
                if ncv[i]*ncv[j] != 0: 
                    frac_cov[i][j] += c/(ncv[i]*ncv[j])
            
    #####################################################
    
    if plot: 
        fig = plt.figure(figsize=(10, 6))
        
        plt.pcolor(bins, bins, cov, cmap='OrRd', edgecolors='k')
            
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=14)
        #if pot: 
            #cbar.set_label(label="$\\nu$ / "+pot, fontsize=15)
        
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        if axis_label is not None: 
            plt.xlabel(axis_label, fontsize=15)
            plt.ylabel(axis_label, fontsize=15)
        else: 
            plt.xlabel(var, fontsize=15)
            plt.ylabel(var, fontsize=15)

        if title: 
            plt.title('Covariance ('+title+')', fontsize=15)
        
        if save: 
            plt.savefig(plots_path+var+"_cov.pdf", transparent=True, bbox_inches='tight') 
            print('saving to: '+plots_path)
        plt.show()
        
        # fractional covariance 
        fig = plt.figure(figsize=(10, 6))
        
        plt.pcolor(bins, bins, frac_cov, cmap='OrRd', edgecolors='k')
            
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=14)
        #if pot: 
            #cbar.set_label(label="$\\nu$ / "+pot, fontsize=15)
        
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        if axis_label is not None: 
            plt.xlabel(axis_label, fontsize=15)
            plt.ylabel(axis_label, fontsize=15)
        else: 
            plt.xlabel(var, fontsize=15)
            plt.ylabel(var, fontsize=15)

        if title: 
            plt.title('Fractional Covariance ('+title+')', fontsize=15)
        
        if save: 
            plt.savefig(plots_path+var+"_frac_cov.pdf", transparent=True, bbox_inches='tight') 
            print('saving to: '+plots_path)
        plt.show()
        
    #####################################################    
    # compute the corr matrix 

    for i in range(len(cov)): 
        for j in range(len(cov[i])): 

            if np.sqrt(cov[i][i])*np.sqrt(cov[j][j]) != 0: 
                cor[i][j] = cov[i][j] / np.sqrt(cov[i][i])*np.sqrt(cov[j][j])
    
    #####################################################
    
    if plot: 
        fig = plt.figure(figsize=(10, 6))

        plt.pcolor(bins, bins, cor, cmap='OrRd', edgecolors='k')#, vmin=0, vmax=1)
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=14)
        
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        if axis_label is not None: 
            plt.xlabel(axis_label, fontsize=15)
            plt.ylabel(axis_label, fontsize=15)
        else: 
            plt.xlabel(var, fontsize=15)
            plt.ylabel(var, fontsize=15)
            
        if title: 
            plt.title('Correlation ('+title+')', fontsize=15)
        if save: 
            plt.savefig(plots_path+var+"_corr.pdf", transparent=True, bbox_inches='tight') 
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
def plotFullCov(frac_cov_dict, var, cv, bins, xlow, xhigh, save=False, axis_label=None, isrun3=False, pot=None): 
    
    print('Note: Input must be fractional covariance matrices !')
    
    plots_path = parameters(isrun3)['plots_path']
    
    keys = list(frac_cov_dict.keys())
    tot_frac_cov = [ [0]*(len(bins)-1) for x in range(len(bins)-1) ]
    
    # add elements of the same index together 
    for source in keys: 
        tot_frac_cov = [ [x+y for x,y in zip(a,b)] for a,b in zip(tot_frac_cov, frac_cov_dict[source])]
    
    # plot 
    fig = plt.figure(figsize=(10, 6))
    
    plt.pcolor(bins, bins, tot_frac_cov, cmap='OrRd', edgecolors='k')
    cbar = plt.colorbar()
    #if pot: 
    #    cbar.set_label(label="$\\nu$ / "+pot, fontsize=15)
    cbar.ax.tick_params(labelsize=14)
        
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    if axis_label is not None: 
        plt.xlabel(axis_label, fontsize=15)
        plt.ylabel(axis_label, fontsize=15)
    else: 
        plt.xlabel(var, fontsize=15)
        plt.ylabel(var, fontsize=15)

    plt.title('Fractional Covariance', fontsize=15)
    
    if save: 
            plt.savefig(plots_path+var+"_tot_frac_cov.pdf", transparent=True, bbox_inches='tight') 
    plt.show()
    
    
    # convert back to absolute covariance units 
    abs_cov = [ [0]*(len(bins)-1) for x in range(len(bins)-1) ]

    for i in range(len(bins)-1): 
        for j in range(len(bins)-1): 
            abs_cov[i][j] = tot_frac_cov[i][j] * cv[i] * cv[j]
            
    
    # plot 
    fig = plt.figure(figsize=(10, 6))
    
    plt.pcolor(bins, bins, abs_cov, cmap='OrRd', edgecolors='k')
    cbar = plt.colorbar()
    #if pot: 
    #    cbar.set_label(label="$\\nu$ / "+pot, fontsize=15)
    cbar.ax.tick_params(labelsize=14)
        
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    if axis_label is not None: 
        plt.xlabel(axis_label, fontsize=15)
        plt.ylabel(axis_label, fontsize=15)
    else: 
        plt.xlabel(var, fontsize=15)
        plt.ylabel(var, fontsize=15)

    plt.title('Absolute Covariance', fontsize=15)
    
    if save: 
            plt.savefig(plots_path+var+"_tot_abs_cov.pdf", transparent=True, bbox_inches='tight') 
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
    
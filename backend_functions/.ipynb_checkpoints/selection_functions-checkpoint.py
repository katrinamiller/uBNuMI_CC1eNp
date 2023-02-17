## UPDATE THE PLOTS_PATH FOR SAVING DISTRIBUTIONS ## 

import sys

import math
import warnings
import importlib 

#import scipy.stats
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

sys.path.insert(0, 'backend_functions')

import NuMIDetSys

# importlib.reload(NuMIDetSys)
NuMIDetSysWeights = NuMIDetSys.NuMIDetSys()

import top 
importlib.reload(top)
from top import *

import uncertainty_functions
from uncertainty_functions import * 

import ROOT
from ROOT import TH1F, TH2F, TDirectory, TH1D


########################################################################
################## inputs to functions: ################################
# datasets: [infv, outfv, cosmic, ext, data]

# infv: overlay & dirt events with truth vtx in FV 
# outfv: overlay & dirt events with truth vtx in FV that are classified as neutrinos
# cosmic: overlay & dirt events with true vtx in FV that get misclassified as cosmic 
# ext: beam OFF data
# data:  beam ON data 

########################################################################

########################################################################
######################## selection functions ###########################
# construct the truth opening angle - this function needs work 
# NO LONGER IN USE 
def true_opening_angle(df): 
    
    # compute the magnitude of all the MC Particles
    df['mc_p'] = (df['mc_px']*df['mc_px']+df['mc_py']*df['mc_py']+df['mc_pz']*df['mc_pz']).pow(0.5)
    
    tksh_angle_truth = []
    for index, row in df.iterrows(): 

        proton_max_p = 0
        proton_max_p_idx = 0
    
        # for each index in the PDG list 
        for i in range(len(row['mc_pdg'])): 
        
            # check if it is a proton 
            if row['mc_pdg'][i]==2212: 
                # check if it is max
                if row['mc_p'][i]>proton_max_p: 
                    # if so, replace
                    proton_max_p = row['mc_p2'][i]
                    proton_max_p_idx = i 
                
        # now use the leading proton index to compute the opening angle 
        proton_p = [ row['mc_px'][i], row['mc_py'][i], row['mc_pz'][i] ]
    
        # and construct electron vector (should be norm'd already)
        elec_p = [ row['elec_px'], row['elec_py'], row['elec_pz'] ]
        elec_p_mag = np.sqrt((elec_p[0]*elec_p[0])+(elec_p[1]*elec_p[1])+(elec_p[2]*elec_p[2]))
    
        # opening angle 
        cos = np.dot(proton_p, elec_p) / (proton_max_p*elec_p_mag)
        if proton_max_p==0 or elec_p_mag==0: 
            tksh_angle_truth.append(np.nan)
        
        else: 
            tksh_angle_truth.append(cos)
            
    df['[true_opening_angle]'] = tksh_angle_truth 
    
    return df
########################################################################
# add angles in beam & detector coordinates
def addAngles(df): 
    
    ## rotation matrix -- convert detector to beam coordinates
    R = [
        [0.921,   4.625e-05,     -0.3895],
        [0.02271,    0.9983,     0.05383],
        [0.3888,   -0.05843,      0.9195]
    ]
    det_origin_beamcoor = [5502.0, 7259.0,  67270.0]
     
    # angles in detector coordinates
    df['thdet'] = np.arctan2(((df['true_nu_px']*df['true_nu_px'])+(df['true_nu_py']*df['true_nu_py']))**(1/2), df['true_nu_pz'])*(180/math.pi)
    df['phidet'] = np.arctan2(df['true_nu_py'], df['true_nu_px'])*(180/math.pi)
        
    # get true momentum in beam coordinates
    df['true_nu_px_beam'] = R[0][0]*df['true_nu_px'] + R[0][1]*df['true_nu_py'] + R[0][2]*df['true_nu_pz']
    df['true_nu_py_beam'] = R[1][0]*df['true_nu_px'] + R[1][1]*df['true_nu_py'] + R[1][2]*df['true_nu_pz']
    df['true_nu_pz_beam'] = R[2][0]*df['true_nu_px'] + R[2][1]*df['true_nu_py'] + R[2][2]*df['true_nu_pz']
    
    # angles in beam coordinates
    df['thbeam'] = np.arctan2(((df['true_nu_px_beam']*df['true_nu_px_beam'])+(df['true_nu_py_beam']*df['true_nu_py_beam']))**(1/2), df['true_nu_pz_beam'])*(180/math.pi)
    df['phibeam'] = np.arctan2(df['true_nu_py_beam'], df['true_nu_px_beam'])*(180/math.pi)
        
    return df
########################################################################
# use offline flux weights
def offline_flux_weights(df, ISRUN3): 
    
    if ISRUN3: 
        print("No ppfx maps for RHC!")
    
    else: 
        f = ROOT.TFile.Open("/uboone/data/users/kmiller/uBNuMI_CCNp/ppfx_maps.root", "READ")
    
    numu_map = f.Get("numu_ratio")
    numubar_map = f.Get("numubar_ratio")
    nue_map = f.Get("nue_ratio")
    nuebar_map = f.Get("nuebar_ratio")
    
    nu_flav = list(df['nu_pdg'])
    angle = list(df['thbeam'])
    true_energy = list(df['nu_e'])

    fluxweights = []

    for i in range(len(nu_flav)): 
        if nu_flav[i]==14: 
            h = numu_map
        elif nu_flav[i]==-14: 
            h = numubar_map
        elif nu_flav[i]==12: 
            h = nue_map
        elif nu_flav[i]==-12: 
            h = nuebar_map
        else: 
            print("No map to match PDG code!")
        
        fluxweights.append( h.GetBinContent(h.FindBin(true_energy[i], angle[i])) )

    df['ppfx_cv'] = fluxweights
    #mc_df[0]['weightFlux'] = fluxweights
    #mc_df[1]['weightFlux'] = [1 for i in range(len(mc_df[1]))] # for now 
    
    f.Close()
    
    return df
    
########################################################################
# error on the data/MC ratio 
def get_ratio_err(n_data, n_mc): 
    
    err = []
    for i in range(len(n_data)): 
        
        # divide the counting error by n_mc - the MC error is handled by the systematics band 
        
        if n_data[i]>=0: 
            err.append( math.sqrt(n_data[i])/n_mc[i] )
        else: 
            err.append(0)

   # err = []
   # for i in range(len(n_data)): 
   #     err.append( n_data[i]/n_mc[i] * math.sqrt( (math.sqrt(n_data[i]) / n_data[i])**2))# + (math.sqrt(n_mc[i]) / n_mc[i])**2 )) 
   # print(err)
    return err
########################################################################
# get event counts for plotting 
def event_counts(datasets, xvar, xmin, xmax, cuts, ext_norm, mc_norm, plot_data=False, bdt_scale=None):
    
    q = (xvar+">="+str(xmin)+" and "+xvar+"<="+str(xmax))
    
    if cuts: 
        q = q + " and " + cuts
        
    counts = {
        'outfv' : round(np.nansum(datasets['outfv'].query(q)[mc_norm]), 1), 
        'numu_NC_Npi0' : round(np.nansum(datasets['infv'].query(numu_NC_Npi0+" and "+ q)[mc_norm]), 1),
        'numu_CC_Npi0' : round(np.nansum(datasets['infv'].query(numu_CC_Npi0+" and "+ q)[mc_norm]), 1), 
        'numu_NC_0pi0' : round(np.nansum(datasets['infv'].query(numu_NC_0pi0+" and "+ q)[mc_norm]), 1),
        'numu_CC_0pi0' : round(np.nansum(datasets['infv'].query(numu_CC_0pi0+" and "+ q)[mc_norm]), 1), 
        'nue_NC' : round(np.nansum(datasets['infv'].query(nue_NC+" and "+ q)[mc_norm]), 1), 
        'nue_CCother' : round(np.nansum(datasets['infv'].query(nue_CCother+" and "+ q)[mc_norm]), 1), 
        'numu_Npi0' : round(np.nansum(datasets['infv'].query(numu_Npi0+" and "+ q)[mc_norm]), 1), 
        'numu_0pi0' : round(np.nansum(datasets['infv'].query(numu_0pi0+" and "+ q)[mc_norm]), 1), 
        'nue_other' : round(np.nansum(datasets['infv'].query(nue_other+" and "+ q)[mc_norm]), 1),
        'nuebar_1eNp' : round(np.nansum(datasets['infv'].query(nuebar_1eNp+" and "+ q)[mc_norm]), 1),
        'signal' : round(np.nansum(datasets['infv'].query(signal+" and "+ q)[mc_norm]), 1), 
        'ext' : round(np.nansum(datasets['ext'].query(q)[ext_norm]), 1)
    }

    if bdt_scale: 
        for category in counts.keys(): 
            counts[category] = counts[category]/bdt_scale
    
    return counts

########################################################################
# Plot MC, normalized to beam on, overlay, OR projected 
# NEED TO ADD: GENIE UNISIMS, NON-nueCC DET SYS
def plot_mc(var, nbins, xlow, xhigh, cuts, datasets, isrun3, norm='overlay', save=False, save_label=None, log=False, x_label=None, xmax=None, y_label=None, ymax=None, bdt_scale=None, text=None, xtext=None, ytext=None, osc=None, plot_bkgd=False, sys=None, x_ticks=None, bin_norm=1.0):
    
    
    # set the POT & plots_path for plotting
    plots_path = parameters(isrun3)['plots_path']

    if (cuts==""): 
        infv = datasets['infv']
        outfv = datasets['outfv']
        ext = datasets['ext']
        
    else: 
        infv = datasets['infv'].query(cuts)
        outfv = datasets['outfv'].query(cuts)
        ext = datasets['ext'].query(cuts)
    
    ## MC weights
    categories = {'ext' : ext, 
                  'outfv' : outfv, 
                  'numu_NC_Npi0' : infv.query(numu_NC_Npi0), 
                  'numu_CC_Npi0' : infv.query(numu_CC_Npi0), 
                  'numu_NC_0pi0' : infv.query(numu_NC_0pi0), 
                  "numu_CC_0pi0" : infv.query(numu_CC_0pi0), 
                  'nue_NC' : infv.query(nue_NC), 
                  'nue_CCother' : infv.query(nue_CCother), 
                  'nuebar_1eNp' : infv.query(nuebar_1eNp), 
                  'signal' : infv.query(signal),
                  }
    
    mc_norm = ''
    ext_norm = ''

    if (norm=='data'): 
        
        mc_norm = 'totweight_data'
        ext_norm = 'pot_scale'
        
    else: 
        print("update!")
        
    mc_weights = {}
    if bdt_scale: 
        print("Accounting for BDT test/train split....")
        for category in categories.keys(): 
            if category=='ext': 
                mc_weights['ext'] = [ x/(bdt_scale) for x in categories[category][ext_norm]]
            else: 
                mc_weights[category] = [ x/(bdt_scale) for x in categories[category][mc_norm]]
              
    else:
        for category in categories.keys(): 
            if category=='ext': 
                mc_weights['ext'] = categories[category][ext_norm]
            else: 
                mc_weights[category] = categories[category][mc_norm]
        
    # event counts
    counts = event_counts(datasets, var, nbins[0], nbins[-1], cuts, ext_norm, mc_norm, plot_data=False, bdt_scale=bdt_scale)
     
    # legend 
    leg = {
        'ext' : labels['ext'][0]+': '+str(counts['ext']),
        'outfv' : labels['outfv'][0]+': '+str(counts['outfv']), 
        'numu_NC_Npi0' : labels['numu_NC_Npi0'][0]+': '+str(counts['numu_NC_Npi0']), 
        'numu_CC_Npi0' : labels['numu_CC_Npi0'][0]+': '+str(counts['numu_CC_Npi0']), 
        'numu_NC_0pi0' : labels['numu_NC_0pi0'][0]+': '+str(counts['numu_NC_0pi0']), 
        'numu_CC_0pi0' : labels['numu_CC_0pi0'][0]+': '+str(counts['numu_CC_0pi0']), 
        'nue_NC' : labels['nue_NC'][0]+': '+str(counts['nue_NC']), 
        'nue_CCother' : labels['nue_CCother'][0]+': '+str(counts['nue_CCother']),
        'nuebar_1eNp' : labels['nuebar_1eNp'][0]+': '+str(counts['nuebar_1eNp']), 
        'signal' : labels['signal'][0]+': '+str(counts['signal'])
    }
        
    
    ################### oscillated event rate #########################
    
    if osc:
        
        # plot signal only 
        n_sig, b_sig, p_sig = plt.hist(infv.query(signal)[var], nbins, histtype='bar', range=[xlow, xhigh], weights=mc_weights[-2])
        plt.close()
        #print(n_sig)
        
        osc_weight = []
        
        with open(osc) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count > 0: 
                    osc_weight.append(float(row[0]))
                    #bin_centers.append(float(row[1]))
                    
                line_count += 1

        osc_counts = [ a*b for a, b in zip(n_sig,osc_weight) ]
        
    ############### Error calculation pt. 1 (pre-plotting) #######################
    
    if sys is None: 
        mc_err = mc_error(var, nbins, xlow, xhigh, [infv, outfv]) 
    
        # quick plot of ext 
        ext_counts = plt.hist(ext[var], nbins, range=[xlow, xhigh], weights=ext[ext_norm])[0]
        plt.close()
    
    ############################ PLOT ####################################### 
     
    fig = plt.figure(figsize=(8, 5))
    n, b, p = plt.hist([ext[var], outfv[var], 
                       infv.query(numu_NC_Npi0)[var],
                       infv.query(numu_CC_Npi0)[var],
                       infv.query(numu_NC_0pi0)[var],
                       infv.query(numu_CC_0pi0)[var],
                       infv.query(nue_NC)[var],
                       infv.query(nue_CCother)[var],
                       infv.query(nuebar_1eNp)[var], 
                       infv.query(signal)[var]],
            nbins, histtype='bar', range=[xlow, xhigh], stacked=True, 
            color=[labels['ext'][1], labels['outfv'][1], 
                       labels['numu_NC_Npi0'][1], 
                       labels['numu_CC_Npi0'][1], 
                       labels['numu_NC_0pi0'][1], 
                       labels['numu_CC_0pi0'][1], 
                       labels['nue_NC'][1], 
                       labels['nue_CCother'][1],
                       labels['nuebar_1eNp'][1], 
                       labels['signal'][1]], 
            label=[leg['ext'],
                   leg['outfv'], 
                   leg['numu_NC_Npi0'], 
                   leg['numu_CC_Npi0'], 
                   leg['numu_NC_0pi0'], 
                   leg['numu_CC_0pi0'], 
                   leg['nue_NC'], 
                   leg['nue_CCother'], 
                   leg['nuebar_1eNp'], 
                   leg['signal']
                  ],
            weights=[mc_weights['ext'], 
                     mc_weights['outfv'], 
                     mc_weights['numu_NC_Npi0'], 
                     mc_weights['numu_CC_Npi0'], 
                     mc_weights['numu_NC_0pi0'], 
                     mc_weights['numu_CC_0pi0'], 
                     mc_weights['nue_NC'], 
                     mc_weights['nue_CCother'], 
                     mc_weights['nuebar_1eNp'], 
                     mc_weights['signal'] 
                     ])
    
    # total selected 
    print('total selected = '+str(np.nansum(n[-1])))
    
    
    ############### Error calculation pt. 2 (post-plotting) #######################
    
    if sys is not None: 
        
        err_label = 'MC+EXT Stat.\n& Sys. Uncertainty'
        tot_percent_err = sys
        tot_err = [x*y for x,y in zip(n[-1],sys)]
        
    else: 
        ext_percent_err = np.sqrt(ext_counts)/n[-1]
        mc_percent_err = mc_err/n[-1]
    
        # add in quadrature 
        sim_percent_err = np.array([x**2+y**2 for x,y in zip(mc_percent_err, ext_percent_err)])
        sim_percent_err = np.sqrt(sim_percent_err)
    
        sim_err = [x*y for x, y in zip(n[-1], sim_percent_err)]
        
        err_label = 'MC+EXT Stat.\nUncertainty'
        
        tot_err = sim_err
        tot_percent_err =  sim_percent_err
        
    
    # uncertainty band 
    low_err = [ x-y for x,y in zip(n[-1], tot_err) ]
    low_err.insert(0, low_err[0])

    high_err = [ x+y for x,y in zip(n[-1], tot_err)]
    high_err.insert(0, high_err[0])
    
    plt.fill_between(nbins, low_err, high_err, step="pre", facecolor=(.25, .25, .25, 0), 
                     edgecolor='darkgray', 
                     hatch='.....', 
                     linewidth=0.0, zorder=2, 
                     label=err_label)
    
    #bincenters = 0.5*(b[1:]+b[:-1])
    #plt.errorbar(bincenters, n[-1], yerr=sim_err, fmt='none', color='black', linewidth=1)
    
    # simulation outline 
    tot = list([0, n[-1][0]])+list(n[-1])+[0]
    b_step = list([b[0]])+list(b)+list([b[-1]])
    plt.step(b_step, tot, color='saddlebrown', linewidth=2)
      
    ##################### Add in oscillated event rate #############################
    
    if osc:    
        # add in unoscillated background 
        osc_counts = list([0, osc_counts[0]])+osc_counts+[0]
        sig_counts = list([0, n_sig[0]])+list(n_sig)+[0]
        bkgd_counts = [y-z for y, z in zip(tot,sig_counts)]
        osc_counts = [a+b for a,b in zip(osc_counts, bkgd_counts)]
        
        plt.step(b_step, osc_counts, color='darkblue', linestyle='dashed')
    
    ############################################################################## 
   
    # plot format stuff
    plt.legend(loc='best', prop={"size":10}, ncol=3, frameon=False)
    
        
    if y_label: 
        plt.ylabel(y_label, fontsize=15)
    
    if x_label:
        plt.xlabel(x_label, fontsize=15)
    else: 
        plt.xlabel(var, fontsize=15)
    
    if x_ticks: 
        plt.xticks(x_ticks, fontsize=14)
    else: 
        plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    if log: 
        plt.yscale('log')
        
    if ymax: 
        if log: 
            plt.ylim(1, ymax)
        else: 
            plt.ylim(0, ymax)
            
    if xmax: 
        plt.xlim(xlow, xmax)
    else: 
        plt.xlim(xlow, xhigh)
            
    if text: 
        plt.text(xtext, ytext, text, fontsize='xx-large', horizontalalignment='right')
    
    if save: 
        plt.savefig(plots_path+var+"_"+save_label+".svg", transparent=True, bbox_inches='tight') 
        print('saving to: '+plots_path)
        
    plt.show()
    
    ######################### plot background only #################################
    
    
    
    mc_bkgd_err = mc_error(var, nbins, xlow, xhigh, [outfv, infv.query(not_signal)]) 

    fig = plt.figure(figsize=(8, 5))

    n2, b2, p2 = plt.hist([outfv[var], 
                           infv.query(numu_NC_Npi0)[var],
                           infv.query(numu_CC_Npi0)[var],
                           infv.query(numu_NC_0pi0)[var],
                           infv.query(numu_CC_0pi0)[var],
                           infv.query(nue_NC)[var],
                           infv.query(nue_CCother)[var],
                           infv.query(nuebar_1eNp)[var], 
                           ext[var]],
                nbins, histtype='bar', range=[xlow, xhigh], stacked=True, 
                color=[labels['outfv'][1], 
                           labels['numu_NC_Npi0'][1], 
                           labels['numu_CC_Npi0'][1], 
                           labels['numu_NC_0pi0'][1], 
                           labels['numu_CC_0pi0'][1], 
                           labels['nue_NC'][1], 
                           labels['nue_CCother'][1],
                           labels['nuebar_1eNp'][1], 
                           labels['ext'][1]], 
                label=[leg['outfv'], 
                       leg['numu_NC_Npi0'], 
                       leg['numu_CC_Npi0'], 
                       leg['numu_NC_0pi0'], 
                       leg['numu_CC_0pi0'], 
                       leg['nue_NC'], 
                       leg['nue_CCother'], 
                       leg['nuebar_1eNp'], 
                       leg['ext']],
                weights=[mc_weights['outfv'], 
                       mc_weights['numu_NC_Npi0'], 
                       mc_weights['numu_CC_Npi0'], 
                       mc_weights['numu_NC_0pi0'], 
                       mc_weights['numu_CC_0pi0'], 
                       mc_weights['nue_NC'], 
                       mc_weights['nue_CCother'], 
                       mc_weights['nuebar_1eNp'], 
                       mc_weights['ext']
                      ])
    
    if plot_bkgd: 

        plt.legend(loc='best', prop={"size":10}, ncol=3, frameon=False)

        mc_bkgd_percent_err = mc_bkgd_err/n2[-1]

        # add in quadrature 
        sim_bkgd_percent_err = np.array([x**2+y**2 for x,y in zip(mc_bkgd_percent_err, ext_percent_err)])
        sim_bkgd_percent_err = np.sqrt(sim_bkgd_percent_err)

        sim_bkgd_err = [x*y for x, y in zip(n2[-1], sim_bkgd_percent_err)]

        tot2 = list([0, n2[-1][0]])+list(n2[-1])+[0]
        b_step2 = list([b2[0]])+list(b2)+list([b2[-1]])
        plt.step(b_step2, tot2, color='black', linewidth=.7)
        
        # ERRORS
        plt.errorbar(bincenters, n2[-1], yerr=sim_bkgd_err, fmt='none', color='black', linewidth=1)
    
        # FORMATTING STUFF
        
        if pot is not None: 
            plt.ylabel("$\\nu$ / "+pot+" POT", fontsize=15)
    
        if x_label:
            plt.xlabel(x_label, fontsize=15)
        else: 
            plt.xlabel(var, fontsize=15)
    
        plt.xlim(xlow, xhigh)
    
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        
        if ymax: 
            if log: 
                plt.ylim(1, ymax)
            else: 
                plt.ylim(0, ymax)
    
        plt.title('Background Distribution', fontsize=15) 
        plt.show()
        
    else: 
        plt.close()
    
    
    ######################### Create dictionary ##################################
    # return python dictionary with bins, CV, & fractional uncertainties 
    d = { 
       "bins" : nbins, 
        "CV" : list(n[-1]), 
        "background_counts" : list(n2[-1])
    }

    return d

########################################################################
# Data/MC comparisons
def plot_data(var, nbins, xlow, xhigh, cuts, datasets, isrun3, bdt_scale=None, save=False, save_label=None, log=False, x_label=None, y_label=None, ymax=None, sys=None, text=None, xtext=None, ytext=None, ncol=None, x_ticks=None): 
    
    # set the POT & plots_path for plotting
    plots_path = parameters(isrun3)['plots_path']
    
    mc_norm = 'totweight_data'
    ext_norm = 'pot_scale'

    if (cuts==""): 
        infv = datasets['infv']
        outfv = datasets['outfv']
        ext = datasets['ext']
        data = datasets['data']
        
    else: 
        infv = datasets['infv'].query(cuts)
        outfv = datasets['outfv'].query(cuts)
        ext = datasets['ext'].query(cuts)
        data = datasets['data'].query(cuts)
    
    
    ####### get beam on histogram info #######
    n_data, b_data, p_data = plt.hist(data[var], nbins, range=[xlow, xhigh])
    integral_data = np.nansum(n_data)
    plt.close()

    if var=='tksh_angle': 
        bincenters = 0.5*(np.array(nbins)[1:]+np.array(nbins)[:-1])
    else: 
        bincenters = 0.5*(np.array(nbins[:-1]+[xhigh])[1:]+np.array(nbins[:-1]+[xhigh])[:-1])

    ####### get integral for simulated event spectrum #######
    n_sim, b_sim, p_sim = plt.hist([outfv[var], infv[var], ext[var]], 
                                  nbins, range=[xlow, xhigh], stacked=True, 
                                  weights=[#cosmic[mc_norm], 
                                           outfv[mc_norm], infv[mc_norm], ext[ext_norm]])
    integral_mc = np.nansum(n_sim[-1])
    plt.close()
    
    ####### weights for the MC plot #######
    mc_weights = []
    mc_weights_pot = [ext[ext_norm], 
                      outfv[mc_norm], 
                      infv.query(numu_NC_Npi0)[mc_norm], 
                      infv.query(numu_CC_Npi0)[mc_norm], 
                      infv.query(numu_NC_0pi0)[mc_norm], 
                      infv.query(numu_CC_0pi0)[mc_norm], 
                      infv.query(nue_NC)[mc_norm], 
                      infv.query(nue_CCother)[mc_norm], 
                      infv.query(nuebar_1eNp)[mc_norm], 
                      infv.query(signal)[mc_norm]]
    
    ####### account for POT change in the test/train splitting #######
    if bdt_scale: 
        print('Accounting for test/train split....')
        mc_weights_pot = [[x/bdt_scale for x in y] for y in mc_weights_pot]

    mc_weights = mc_weights_pot


    ######## event counts ########
    counts = event_counts(datasets, var, nbins[0], nbins[-1], cuts, ext_norm, mc_norm, plot_data=True, bdt_scale=bdt_scale)

    
    ######## legend ########
    leg = [labels['ext'][0]+': '+str(counts['ext']),
                        labels['outfv'][0]+': '+str(counts['outfv']), 
                        labels['numu_NC_Npi0'][0]+': '+str(counts['numu_NC_Npi0']), 
                        labels['numu_CC_Npi0'][0]+': '+str(counts['numu_CC_Npi0']), 
                        labels['numu_NC_0pi0'][0]+': '+str(counts['numu_NC_0pi0']), 
                        labels['numu_CC_0pi0'][0]+': '+str(counts['numu_CC_0pi0']), 
                        labels['nue_NC'][0]+': '+str(counts['nue_NC']), 
                        labels['nue_CCother'][0]+': '+str(counts['nue_CCother']), 
                        labels['nuebar_1eNp'][0]+': '+str(counts['nuebar_1eNp']),
                        labels['signal'][0]+': '+str(counts['signal'])
                        ]

    ############### error calculation pt. 1 (pre-plotting) #######################
    
    if sys is None: # then only plot the stat error 
        
        mc_err = mc_error(var, nbins, xlow, xhigh, [infv, outfv]) 
        
        # quick plot of ext 
        ext_counts = plt.hist(ext[var], nbins, range=[xlow, xhigh], weights=ext[ext_norm])[0]
        plt.close()
        

    ##############################################################################
    
    # plot 
    #fig = plt.figure(figsize=(12, 10))
    fig = plt.figure(figsize=(8, 7))

    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
    
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    
    ax1.tick_params(axis = 'both', which = 'major', labelsize = 14)
    ax2.tick_params(axis = 'both', which = 'major', labelsize = 14)
    
    ax2.yaxis.grid(linestyle="--", color='black', alpha=0.2)
    ax2.xaxis.grid(linestyle="--", color='black', alpha=0.2)
    
    if x_ticks: 
        ax1.set_xticks(x_ticks)
        ax2.set_xticks(x_ticks)
    

    n, b, p = ax1.hist([ext[var], 
                        outfv[var], 
                        infv.query(numu_NC_Npi0)[var],
                        infv.query(numu_CC_Npi0)[var],
                        infv.query(numu_NC_0pi0)[var],
                        infv.query(numu_CC_0pi0)[var],
                        infv.query(nue_NC)[var], 
                        infv.query(nue_CCother)[var], 
                        infv.query(nuebar_1eNp)[var], 
                        infv.query(signal)[var]], 
            nbins, histtype='bar', range=[xlow, xhigh], stacked=True, 
            color=[labels['ext'][1], 
                        #labels['cosmic'][1], 
                        labels['outfv'][1], 
                        labels['numu_NC_Npi0'][1], 
                        labels['numu_CC_Npi0'][1], 
                        labels['numu_NC_0pi0'][1], 
                        labels['numu_CC_0pi0'][1], 
                        labels['nue_NC'][1], 
                        labels['nue_CCother'][1], 
                        labels['nuebar_1eNp'][1], 
                        labels['signal'][1] 
                       ], 
            label=leg, 
            weights=mc_weights, zorder=1)
    
    
    
    ############################ PLOT THE BEAM-ON DATA ############################
     
    # calculate the width of each bin 
    #x_err = [ (b[i+1]-b[i])/2 for i in range(len(b)-1) ]
    
    x_err = []
    for x in range(len(bincenters)):
        if var=='tksh_angle': 
            x_err.append(round(abs((nbins)[x+1]-(nbins)[x])/2, 3))
        
        else: 
            x_err.append(round(abs((nbins[:-1]+[xhigh])[x+1]-(nbins[:-1]+[xhigh])[x])/2, 3))

    ax1.errorbar(bincenters, n_data, yerr=np.sqrt(n_data), xerr=x_err, 
             color="black", fmt='o', markersize=3, label='DATA: '+str(int(sum(n_data))), zorder=4)
    
    if y_label: 
        ax1.set_ylabel("$\\nu$ / "+y_label+ " POT", fontsize=15)
    
    ax1.set_xlim(xlow, xhigh)
    
    if ymax:
        if log: 
            ax1.set_ylim(1, ymax)
        else: 
            ax1.set_ylim(0, ymax)    
    
    ############### error calculation pt. 2 (post-plotting)#######################
    
    if sys is not None: 
        
        err_label = 'MC+EXT Stat.\n& Sys. Uncertainty'
        tot_percent_err =  sys
        tot_err = [x*y for x, y in zip(n[-1], sys)]
    
    else: 
        ext_percent_err = np.sqrt(ext_counts)/n[-1]
        mc_percent_err = mc_err/n[-1]

        # add in quadrature 
        sim_percent_err = np.array([x**2+y**2 for x,y in zip(mc_percent_err, ext_percent_err)])
        sim_percent_err = np.sqrt(sim_percent_err)

        sim_err = [x*y for x, y in zip(n[-1], sim_percent_err)]
        
        err_label = 'MC+EXT Stat.\nUncertainty'
        
        tot_err = sim_err
        tot_percent_err =  sim_percent_err

    
    low_err = [ x-y for x,y in zip(n[-1], tot_err) ]
    low_err.insert(0, low_err[0])

    high_err = [ x+y for x,y in zip(n[-1], tot_err)]
    high_err.insert(0, high_err[0])
    
    ax1.fill_between(nbins, low_err, high_err, step="pre",
                    facecolor=(.25, .25, .25, 0), 
                     edgecolor='darkgray', #(.8627, .8627, .8627, 1),  
                     hatch='.....', 
                     linewidth=0.0, zorder=2, 
                     label=err_label)
    
    # simulation outline 
    tot = list([0, n[-1][0]])+list(n[-1])+[0]
    b_step = list([b[0]])+list(b)+list([b[-1]])
    ax1.step(b_step, tot, color='saddlebrown', linewidth=2, zorder=3, alpha=0.85)
    
            
    ############################ PLOT THE RATIO ############################
            
    # ratio plot  
    ax2.errorbar(bincenters, n_data/n[-1], yerr=get_ratio_err(n_data, n[-1]), xerr=x_err, color="black", fmt='o')
    ax2.set_xlim(xlow, xhigh)
    ax2.set_ylim(0, 2)
    #ax2.yaxis.grid(linestyle="-", color='black', alpha=0.7)
    
    # horizontal line at 1 
    ax2.axhline(1.0, color='black', lw=1, linestyle='--')
    
    # MC ratio error - stat + sys 
    low_err_ratio = [ 1 - x for x in tot_percent_err ]
    low_err_ratio.insert(0, low_err_ratio[0])
    
    high_err_ratio = [ 1 + x for x in tot_percent_err ]
    high_err_ratio.insert(0, high_err_ratio[0])

    ax2.fill_between(nbins, low_err_ratio, high_err_ratio, step="pre", facecolor=(.25, .25, .25, 0), 
                     edgecolor='darkgray', #(.8627, .8627, .8627, 1.0), 
                     hatch='.....', 
                     linewidth=0.0, zorder=1)
    
    if x_label: 
        ax2.set_xlabel(x_label, fontsize=15)
    else: 
        ax2.set_xlabel(var, fontsize=15)
        
    ax2.set_ylabel("DATA / (MC+EXT)", fontsize=15)
    
    #ax2.set_yticks([0.5, 0.75, 1, 1.25, 1.5])
    #ax1.set_xticks([0, 1])
    #ax2.set_xticks([0, 1])
    
    if ncol: 
        ax1.legend(prop={"size":10}, ncol=ncol, frameon=False)
        
    else:
        ax1.legend(prop={"size":10}, ncol=3, frameon=False)
        
    if log: 
        ax1.set_yscale('log')
        
    ############################ FINAL PLOTTING DETAILS ############################
        
    ## chi2 calculation ## 
    #chi2 = 0 
    
    #for i in range(len(n[-1])): 
    #    if tot_err[i]==0 or np.isnan(((n_data[i] - n[-1][i] )**2 / tot_err[i]**2)): 
    #        continue 
    #    else: 
            
    #        chi2 = chi2 + ((n_data[i] - n[-1][i] )**2 / tot_err[i]**2)  #((i-j)*(i-j))/i stat only chi2
        #print('bin', i, 'chi2 addition', chi2 + ((n_data[i] - n[-1][i] )**2 / tot_err[i]**2))
    
    if text: 
        ax1.text(xtext, ytext, text, #text+"\n$\\chi^{2}$/n = "+str(round(chi2, 2))+"/"+str(len(b)-1), 
                 fontsize='x-large', horizontalalignment='right')
    
    if save: 
        print('saving to: ', plots_path)
        plt.savefig(plots_path+var+"_"+save_label+".svg", bbox_inches='tight')#, dpi=1000) 

    plt.show()
    
    d = {
        #'percent_errors': percent_errors, 
        #'tot_err_percent' : tot_err_percent, 
        'mc_counts' : n[-1], 
        'data_counts': n_data, 
        'data_error' : np.sqrt(n_data), 
        'data_mc_ratio' : n_data/n[-1], 
        'data_mc_ratio_err' : get_ratio_err(n_data, n[-1])
    }
    
    return d
    
########################################################################
# Return a table of the selection performance 
# normalized to data POT
def selection_performance(cuts, datasets, gen, ISRUN3):

    #################
    # cuts --> list of strings of the cuts applied
    # datasets --> list of dataframes [df_infv, df_outfv, df_cosmic, df_ext, df_data]
    # norm --> normalize to beam ON or overlay? 
    #################
    
    # no cuts on these yet, only separated into their truth categories 
    infv = datasets['infv']
    outfv = datasets['outfv']
    ext = datasets['ext']
    
    norm = 'totweight_data'
    ext_norm = 'pot_scale'
            
    df_out = pd.DataFrame(columns=['cut', '# signal after cut',  'efficiency (%)', 'rel. eff. (%)', 
                                'purity (%)', 'purity (MC only, %)'])
    
    sig_gen_norm = np.nansum(gen)
    print("total # of signal generated in FV (normalized to DATA): "+ str(sig_gen_norm))
    
    num_signal = []
    pur = []
    pur_mconly = []
    eff = []
    rel_eff = []
    cut_list = []
    
    # start with the number of signal events 
    sig_last = sig_gen_norm #round( np.nansum(infv.query(signal)[norm]), 1 )
    
    slimmed_variables = ['nslice==1', reco_in_fv_query, 'contained_fraction>0.9']
    
    q = ''
    n=0
    
    for cut in cuts: 
        
        if cut in slimmed_variables: 
            
            if q == '': 
                q = cut
                
            else: 
                q = q + ' and ' + cut

            sig_sel_norm = np.nansum(generated_signal(ISRUN3, 'nu_e', 1, 0, 20, q)[0])
            
            num_signal.append(round(sig_sel_norm, 1))
            
            eff.append(round(sig_sel_norm/sig_gen_norm * 100, 1))
            rel_eff.append(round(sig_sel_norm/sig_last * 100, 1))
            
            pur.append(np.nan)
            pur_mconly.append(np.nan)
            
            if (n==2): 
                cut_list.append("reco'd in FV")
            else: 
                cut_list.append(cut)
            
            sig_last = sig_sel_norm
            
            n = n+1
            
        else: 
        
            infv = infv.query(cut)
            outfv = outfv.query(cut)
            ext = ext.query(cut)
        
            # how many true signal gets selected?  
            sig_sel_norm = np.nansum(infv.query(signal)[norm]) 

            tot_sel_norm = np.nansum(infv[norm])+np.nansum(outfv[norm])+np.nansum(ext[ext_norm])
            tot_sel_norm_mconly = np.nansum(infv[norm])+np.nansum(outfv[norm]) # do not include EXT
        
            num_signal.append(round(sig_sel_norm, 1))
        
            eff.append(round(sig_sel_norm/sig_gen_norm * 100, 1))
            rel_eff.append(round(sig_sel_norm/sig_last * 100, 1))
        
            pur.append(round(sig_sel_norm/tot_sel_norm * 100, 1))
            pur_mconly.append(round(sig_sel_norm/tot_sel_norm_mconly * 100, 1))
        
            if (n==2): 
                cut_list.append("reco'd in FV")
            else: 
                cut_list.append(cut)
        
            sig_last = sig_sel_norm
            n = n+1
        
    df_out['cut'] = cut_list
    df_out['# signal after cut'] = num_signal
    df_out['efficiency (%)'] = eff
    df_out['rel. eff. (%)'] = rel_eff
    df_out['purity (%)'] = pur
    df_out['purity (MC only, %)'] = pur_mconly
         
    return df_out
########################################################################
# Plot the efficiency - with binomial error bars 
def plot_eff(var, nbins, xlower, xupper, cut, datasets, isrun3, save=False, x_label=None, ymax=None, text=None, xtext=None, ytext=None, x_ticks=None): 

    infv = datasets['infv']
 
    
    ############################ Generated signal ############################

    v_sig_gen = generated_signal(isrun3, var, nbins, xlower, xupper, weight='totweight_intrinsic')[0]

    print("# of generated signal in FV: "+str( np.nansum(v_sig_gen ) ) )

    
    ############################ Selected signal ############################
    
    # apply cuts
    infv_selected = infv.query(cut)
    signal_sel = infv_selected.query('is_signal==True')
    
    print("# of selected signal in FV: "+str( np.sum( signal_sel['ppfx_cv']*signal_sel['weightSplineTimesTune'] ) ) )

    v_sig_sel, b_sig_sel, p_sig_sel = plt.hist(signal_sel[var], 
                                                              nbins, 
                                                              histtype='step', range=[xlower, xupper], 
                                                              label='signal selected in FV',
                                                              weights=signal_sel['ppfx_cv']*signal_sel['weightSplineTimesTune'])
    plt.close()
    
    b_sig_sel[-1] = xupper

   ############################ Efficiency & stat error #######################
    
    #eff = [i/j for i, j in zip(v_sig_sel, v_sig_gen)]
    eff = []
    for i, j in zip(v_sig_sel, v_sig_gen): 
        e = i/j
        if np.isnan(e): 
            eff.append(0)
        else: 
            eff.append(e)
        
    eff_err = []
    for i in range(len(eff)): 
        if eff[i]==0: 
            eff_err.append(0)
        else: 
            eff_err.append(math.sqrt( (eff[i]*(1-eff[i]))/v_sig_gen[i] ))
    
    
    bincenters = 0.5*(b_sig_sel[1:]+b_sig_sel[:-1])
    binwidth = []
    
    for x in range(len(bincenters)): 
        binwidth.append(abs(b_sig_sel[x+1]-b_sig_sel[x])/2)
    
    fig = plt.figure(figsize=(8, 5))
    plt.errorbar(bincenters, eff, xerr=binwidth, yerr=eff_err, fmt='o', 
             color='seagreen', ecolor='seagreen', markersize=3) 
    
    plt.xlim(xlower, xupper)
    plt.grid(linestyle=':')
    
    if x_label: 
        plt.xlabel(x_label, fontsize=15)
    else: 
        plt.xlabel(var, fontsize=15)
    
    if ymax: 
        plt.ylim(0, ymax)
        
    if text: 
        plt.text(xtext, ytext, text, fontsize='xx-large', horizontalalignment='center')
        
    if x_ticks: 
        plt.xticks(x_ticks, fontsize=14)
        
    else: 
        plt.xticks(fontsize=14)
        
    plt.ylabel("Efficiency", fontsize=15)
    plt.tight_layout()

    plt.yticks(fontsize=14)
    
    plots_path = parameters(isrun3)['plots_path']
    
    if save: 
        plt.savefig(plots_path+"eff_"+var+".pdf", transparent=True)
        print("saving to "+plots_path)
    
    plt.show()

        
    
########################################################################
# BDT FUNCTIONS
########################################################################
# makes a copy of MC and EXT dataframes with extra columns 
# updated for modified signal
def addRelevantColumns(datasets): 
    
    mc_bdt = pd.concat([datasets['infv'], datasets['outfv']], ignore_index=True, sort=True)
    ext_bdt = datasets['ext']
    
    mc_bdt['is_mc'] = True 
    ext_bdt['is_mc'] = False
        
    mc_bdt['weight'] = mc_bdt['totweight_data']
    ext_bdt['weight'] = ext_bdt['pot_scale']
            
    df_pre = pd.concat([mc_bdt, ext_bdt], ignore_index=True, sort=True)

    
    return df_pre
########################################################################
def prep_sets(train, test, train_query, test_query, varlist):
    
    train_query = train.query(train_query)
    test_query = test.query(test_query)
    
    # Last column will be signal definition for training ('is_signal')
    X_train, y_train = train_query.iloc[:,:-1], train_query['is_signal']
    
    # Signal definition for testing will always be 'is_signal' or true signal definition
    X_test, y_test = test_query.iloc[:,:-1], test_query['is_signal']

    # Cleaning dataframe
    # Note that data for testing is also cleaned
    for column in varlist:
        X_train.loc[(X_train[column] < -1.0e37) | (X_train[column] > 1.0e37), column] = np.nan
        X_test.loc[(X_test[column] < -1.0e37) | (X_test[column] > 1.0e37), column] = np.nan
    
    # Training and Testing DMatrices are only comprised of training variable list
    dtrain = xgb.DMatrix(data=X_train[varlist], label=y_train)
    dtest = xgb.DMatrix(data=X_test[varlist], label=y_test)
    
    d = {
        'X_train': X_train, 
        'X_test': X_test, 
        'dtrain': dtrain, 
        'dtest' : dtest
    }
    
    return d
########################################################################
def bdt_raw_results(train, test, train_query, test_query, varlist, params, rounds):
    
    d = prep_sets(train, test, train_query, test_query, varlist)
    
    queried_train_df = d['X_train']
    queried_test_df = d['X_test']
    dtrain = d['dtrain']
    dtest = d['dtest']
    
    model = xgb.train(params, dtrain, rounds)
    preds = model.predict(dtest)
    
    queried_test_df['is_signal'] = dtest.get_label()
    queried_test_df['BDT_score'] = preds
    
    return queried_test_df, model
########################################################################
def main_BDT(datasets, train_query, test_query, rounds, training_parameters, isrun3, test_size=0.5):
    
    # combine MC & EXT datasets with additional columns needed for BDT analysis
    df_pre = addRelevantColumns(datasets)
    
    # compute the scale weight for model parameters 
    scale_weight = len(df_pre.query(train_query + ' and is_signal == False')) / len(df_pre.query(train_query + ' and is_signal == True'))
    print("scale pos weight (ratio of negative to positive) = "+str(scale_weight))
    
    # Split arrays or matrices into random train and test subsets
    # stratify keeps the same signal/background ratio 
    df_pre_train, df_pre_test = train_test_split(df_pre, test_size=test_size, random_state=17, stratify=df_pre['is_signal'])

    varlist = training_parameters
    
    #model params
    params = {
        'objective': 'binary:logistic',
        'booster': 'gbtree',
        'eta': 0.02,
        'tree_method': 'exact',
        'max_depth': 3,
        'subsample': 0.8,
        'colsample_bytree': 1,
        'silent': 1,
        'min_child_weight': 1,
        'seed': 2002,
        'gamma': 1,
        'max_delta_step': 0,
        'scale_pos_weight': scale_weight,
        'eval_metric': ['error', 'auc', 'aucpr']
    }
    
    # datasets get cleaned in bdt_raw_results (prep_sets)
    bdt_results_df, bdt_model  = bdt_raw_results(df_pre_train, df_pre_test, train_query, test_query, training_parameters, params, rounds)
    
    d = {
        'bdt_results_df': bdt_results_df, 
        'bdt_model': bdt_model, 
        'df_pre_train': df_pre_train, 
        'df_pre_test': df_pre_test, 
        'df_pre': df_pre
    }
    
    return d
########################################################################    
# BDT Metric evaluation
def bdt_metrics(train, test, train_query, test_query, training_parameters, isrun3, save=False, verbose=False): 
    
    scale_weight = len(train.query(train_query+' and is_signal==True')) / len(train.query(train_query+' and is_signal==False'))
    
    #model params
    params = {
        'objective': 'binary:logistic',
        'booster': 'gbtree',
        'eta': 0.02,
        'tree_method': 'exact',
        'max_depth': 3,
        'subsample': 0.8,
        'colsample_bytree': 1,
        'silent': 1,
        'min_child_weight': 1,
        'seed': 2002,
        'gamma': 1,
        'max_delta_step': 0,
        'scale_pos_weight': scale_weight,
        'eval_metric': ['error', 'auc', 'aucpr']
    }
    
    dtrain = prep_sets(train, test, train_query, test_query, training_parameters)['dtrain']
    dtest = prep_sets(train, test, train_query, test_query, training_parameters)['dtest']

    watchlist = [(dtrain, 'train'), (dtest, 'valid')]

    progress = dict()

    model = xgb.train(params, dtrain, 1000, watchlist, early_stopping_rounds=50, evals_result=progress, verbose_eval=verbose)

    # AUC
    plt.figure(figsize=(10, 5))
    
    plt.plot(progress['train']['auc'], color='orange', label='AUC (Training Sample)', markersize=3)
    plt.plot(progress['valid']['auc'], color='blue', label='AUC (Test Sample)', markersize=3)
    
    plt.grid(linestyle=":")
    plt.legend(loc='best', prop={"size":13})
    
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    
    
    if isrun3: 
        plt.title('RHC Run 3 BDT AUC', fontsize=15)
        plt.ylim(0.68, 0.79)
    else: 
        plt.title('FHC Run 1 BDT AUC', fontsize=15)
    
    plt.xlabel('Number of Boosting Rounds', fontsize=14)
    #plt.ylim(0.75, 0.8)
    
    if save: 
        plt.savefig(parameters(isrun3)['plots_path']+"BDT_AUC.pdf", transparent=True, bbox_inches='tight') 
    plt.show()
    
    
    # AUC PR
    plt.figure(figsize=(10, 5))
    
    plt.plot(progress['train']['aucpr'], color='orange', label='AUC PR (Training Sample)', markersize=3)
    plt.plot(progress['valid']['aucpr'], color='blue', label='AUC PR (Test Sample)', markersize=3)
    
    plt.grid(linestyle=":")
    plt.legend(loc='upper left', prop={"size":13})
    
    if isrun3: 
        plt.title('RHC Run 3 BDT AUCPR', fontsize=15)
    else: 
        plt.title('FHC Run 1 BDT AUCPR', fontsize=15)
        
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    
    plt.xlabel('Number of Boosting Rounds', fontsize=14)
    #plt.ylim(0.7, 0.9)
    
    if save: 
        plt.savefig(parameters(isrun3)['plots_path']+"BDT_AUCPR.pdf", transparent=True, bbox_inches='tight') 
    plt.show()
    
    
    #for metric in progress['train'].keys():
    #    plt.figure(figsize=(10, 5))
    #    plt.plot(progress['train'][metric], color='orange', label='train '+metric, markersize=3)
        #plt.plot(progress['valid'][metric], color='blue', label='test '+metric, markersize=3)  
    #    plt.grid(linestyle=":")
    #    plt.legend(loc='upper right', prop={"size":13})
    #    plt.xlabel('# of rounds')
    #    plt.show()
        
########################################################################
# BDT Purity/Efficiency 
def bdt_pe(df, xvals, gen_data, gen_intrinsic, split):
    
    ########################
    # df --> dataframe with evaluated BDT_score added 
    # xvals --> x axis array
    # full_test_df --> df used for testing (before pre/loose cuts) 
    ########################
    
    purity=[]
    purErr=[]
    eff=[]
    effErr=[]
    

    for cut_val in xvals:
    
        
        cut_val = round(cut_val, 3)
        q = BDT_LOOSE_CUTS+' and BDT_score > '+str(cut_val)
        
        # total signal selected
        tot_sel_sig = np.nansum(df.query(q+' and is_signal == True').weight)
        
        # total events selected
        tot_sel = np.nansum(df.query(q).weight)
        
        # total signal generated
        tot_sig = np.nansum(gen_data)*split # only include the amount of dataset used for TESTING
        tot_sig_intrinsic = np.nansum(gen_intrinsic)*split # for error computation, use the intrinsic event count
        
        p = tot_sel_sig / tot_sel
        purity.append(p * 100)
        purErr.append( p * np.sqrt( sum(df.query(q+' and is_signal==True').weight**2)/sum(df.query(q+' and is_signal==True').weight)**2 + sum(df.query(q).weight**2)/sum(df.query(q).weight)**2 )  *100)
        
        e = tot_sel_sig / tot_sig
        eff.append(e * 100)
        effErr.append(np.sqrt( (e * (1-e)) / tot_sig_intrinsic ) * 100)
        
        
    d = {
        'purity': purity, 
        'purErr': purErr, 
        'eff': eff, 
        'effErr': effErr
    }
    
    return d

########################################################################
########################################################################
def split_events(df):
    
    #separate by in/out FV & cosmic 
    ext_bdt = df.query('is_mc==False')
    outfv_bdt = df.query(out_fv_query+' and is_mc==True')
    #cosmic_bdt = df.query(in_fv_query+' and nu_purity_from_pfp<=0.5 and is_mc==True')
    infv_bdt = df.query(in_fv_query+' and is_mc==True')
    
    # checks 
    print('split_events check:', len(df) == len(ext_bdt)+len(outfv_bdt)+len(infv_bdt))#+len(cosmic))
    
    d = {
        'infv': infv_bdt, 
        'outfv': outfv_bdt, 
        #'cosmic': cosmic_bdt, 
        'ext': ext_bdt
    }
    
    return d
########################################################################
def bdt_svb_plot(df, is_log=False):
    
    plt.hist([df.query('is_signal == True')['BDT_score'], df.query('is_signal == False')['BDT_score']], 
             50, histtype='bar', range=[0, 1.0], stacked=True, 
             color=['orange','cornflowerblue'],
             label=['signal','background'],
             log=is_log)
    
    plt.legend(loc='upper right')
    plt.xlabel('BDT score')
    
    plt.show()
########################################################################  
def bdt_pe_plot(perf, xvals, isrun3, split, save=False):
    
    
    ########################
    # df --> dataframe with evaluated BDT_score added 
    # xvals --> x axis array
    ########################
    
    plots_path = parameters(isrun3)['plots_path']
    
    #plot pur/eff as function of bdt score
    plt.figure(figsize=(7, 5))
    
    pur, purErr, eff, effErr = perf['purity'], perf['purErr'], perf['eff'], perf['effErr']
    
    plt.errorbar(xvals, pur, yerr=purErr, marker='o', color='firebrick', label='Purity', markersize=3)
    plt.errorbar(xvals, eff, yerr=effErr, marker='o', color='seagreen', label='Efficiency', markersize=3)  

    plt.ylabel('Percentage (%)', fontsize=14)
    plt.xlabel('BDT_score > #', fontsize=14)
    plt.grid(linestyle=":")
    plt.xticks(fontsize=12)
    plt.yticks(np.arange(0,105,5), fontsize=12)
    plt.legend(loc='upper left', prop={"size":13})
    plt.ylim(0, 100)
    plt.tight_layout()
    if save: 
        plt.savefig(plots_path+"BDT_performance.pdf", transparent=True, bbox_inches='tight') 

    plt.show()
    
########################################################################    
def bdt_box_plot(results_bdt, xvals, isrun3, second_results_bdt=None, results_box=None, results_box_err=None, save=False, 
                save_label=None, title=None):
    
    ###################
    # results_bdt --> BDT performance (output of bdt_pe function)
    # results_box --> [ linear sel. purity, lineear sel. efficiency ]
    # xvals --> values on x-axis
    # second_results_bdt --> BDT performance using preselection cuts ONLY (as opposed to loose cuts)
    ###################
    
    plots_path = parameters(isrun3)['plots_path']
    
    #plot pur/eff as function of bdt score
    plt.figure(figsize=(7, 5))
    
    # Loose cut BDT results
    pur, purErr, eff, effErr = results_bdt
    
    plt.errorbar(xvals, pur, yerr=purErr, marker='o', color='maroon', label='BDT Purity', markersize=3)
    plt.errorbar(xvals, eff, yerr=effErr, marker='o', color='green', label='BDT Efficiency', markersize=3)  
    

    # Box cut results 
    if results_box: 
        plt.axhline(results_box[0], color='red', 
                    linestyle='dashed', label='Lin. Sel. Purity ('+str(round(results_box[0], 1))+'%)', linewidth=2)
        plt.axhline(results_box[1], color='limegreen', 
                    linestyle='dashed', label='Lin. Sel. Eff. ('+str(round(results_box[1], 1))+'%)', linewidth=2)
    
    if results_box_err:
        
        # purity stat error - poisson  
        plt.fill_between(xvals, results_box[0]-results_box_err[0], results_box[0]+results_box_err[0], color='red', 
                        alpha=0.15)
        
        # eff stat error - binomial 
        plt.fill_between(xvals, results_box[1]-results_box_err[1], results_box[1]+results_box_err[1], color='limegreen', 
                        alpha=0.15)
    
    plt.ylabel('Percentage (%)', fontsize=14)
    plt.xlabel('BDT_score > #', fontsize=14)
    plt.grid(linestyle=":")
    plt.xticks(fontsize=12)
    plt.xlim(0, xvals[-1])
    plt.yticks(np.arange(0,105,5), fontsize=12)
    plt.legend(prop={"size":12}, loc='upper left')
    plt.ylim(0, 100)
    if title: 
        plt.title(title, fontsize=15)
    #plt.tight_layout()
    if save: 
        plt.savefig(plots_path+"BDT_performance_"+save_label+".pdf", transparent=True, bbox_inches='tight') 

    
    plt.show()  
######################################################################## 

    

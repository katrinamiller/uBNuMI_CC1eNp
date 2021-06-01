## UPDATE THE PLOTS_PATH FOR SAVING DISTRIBUTIONS ## 

import math
import warnings

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

import NuMIDetSys
import importlib 

importlib.reload(NuMIDetSys)
NuMIDetSysWeights = NuMIDetSys.NuMIDetSys()

########################################################################
################## inputs to functions: ################################
# datasets: [infv, outfv, cosmic, ext, data]

# infv: overlay & dirt events with truth vtx in FV 
# outfv: overlay & dirt events with truth vtx in FV that are classified as neutrinos
# cosmic: overlay & dirt events with true vtx in FV that get misclassified as cosmic 
# ext: beam OFF data
# data:  beam ON data 

########################################################################
######################### plot categories ##############################

signal = 'nu_pdg==12 and ccnc==0 and nproton>0 and npion==0 and npi0==0'

numu_CC_Npi0 = '(nu_pdg==14 or nu_pdg==-14) and ccnc==0 and npi0>=1'
numu_CC_0pi0 = '(nu_pdg==14 or nu_pdg==-14) and ccnc==0 and npi0==0'
numu_NC_Npi0 = '(nu_pdg==14 or nu_pdg==-14) and ccnc==1 and npi0>=1'
numu_NC_0pi0 = '(nu_pdg==14 or nu_pdg==-14) and ccnc==1 and npi0==0'

nuebar_1eNp = '(nu_pdg==-12 and ccnc==0 and nproton>0 and npion==0 and npi0==0)'
nue_NC = '((nu_pdg==12 or nu_pdg==-12) and ccnc==1)'
nue_CCother = '( ((nu_pdg==12 and ccnc==0) and (nproton==0 or npi0>0 or npion>0)) or (nu_pdg==-12 and ccnc==0 and (nproton==0 or npion>0 or npi0>0)) )'

# less specific
nue_other = '((nu_pdg==12 or nu_pdg==-12) and ccnc==1) or (((nu_pdg==12 and ccnc==0) and (nproton==0 or npi0>0 or npion>0)) or (nu_pdg==-12 and ccnc==0))'
numu_Npi0 = '((nu_pdg==14 or nu_pdg==-14) and npi0>=1)'
numu_0pi0 = '((nu_pdg==14 or nu_pdg==-14) and npi0==0)'



########################################################################
#################### labels ############################################

labels = { 
    'signal' : ['$\\nu_e$ CC0$\pi$Np', 'orange'], 
    'numu_CC_Npi0' : ['$\\nu_\mu$ CC $\pi^{0}$', 'brown'],
    'numu_NC_Npi0' : ['$\\nu_\mu$ NC $\pi^{0}$', 'orangered'],
    'numu_NC_0pi0' : ['$\\nu_\mu$ NC', '#33FCFF'],
    'numu_CC_0pi0' : ['$\\nu_\mu$ CC', '#437ED8'],
    'nue_CCother': ['$\\nu_e$ CC other', '#05B415'], 
    'nue_NC': ['$\\nu_e$ NC', '#B8FF33'], 
    'out_fv' : ['Out FV', 'orchid'], 
    'cosmic' : ['Cosmic Cont.', 'lightpink'],
    'ext' : ['EXT', 'gainsboro'], 
    'nue_other' : ['$\\nu_e$ / $\\overline{\\nu_e}$  other', '#33db09'], 
    'numu_Npi0' : ['$\\nu_\\mu$ / $\\overline{\\nu_\\mu}$  $\pi^{0}$', '#EE1B1B'], 
    'numu_0pi0' : ['$\\nu_\\mu$ / $\\overline{\\nu_\\mu}$  other', '#437ED8'],
    'nuebar_1eNp' : ['$\\bar{\\nu}_e$ CC0$\pi$Np', 'gold'], 
    'ext' : ['EXT', 'gainsboro']
}

in_fv_query = "10<=true_nu_vtx_x<=246 and -106<=true_nu_vtx_y<=106 and 10<=true_nu_vtx_z<=1026"
out_fv_query = "((true_nu_vtx_x<10 or true_nu_vtx_x>246) or (true_nu_vtx_y<-106 or true_nu_vtx_y>106) or (true_nu_vtx_z<10 or true_nu_vtx_z>1026))"


########################################################################
######################## selection functions ###########################
# construct the truth opening angle - this function needs work 
def true_opening_angle(df): 
    
    # compute the magnitude of all the MC Particles
    df['mc_p'] = (df['mc_px']*df['mc_px']+df['mc_py']*df['mc_py']+df['mc_pz']*df['mc_pz']).pow(0.5)
    
    # find index of leading proton (highest momentum)
    
    # construct electron & proton momenta vectors 
    
    # compute opening angle 
    
    
    
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
# set the POT & plots_path for plotting
# UPDATE based on the ntuples being used 
def plot_param(ISRUN3): 
    
    if ISRUN3: # RHC
        plots_path = "/uboone/data/users/kmiller/searchingfornues_v33/v08_00_00_33/plots/rhc/"
        beamon_pot = '$5.014\\times10^{20}$'
        overlay_pot = '$1.578\\times10^{201}$'
        proj_pot = '$11.95\\times10^{20}$'
    
    else: # FHC
        plots_path = "/uboone/data/users/kmiller/searchingfornues_v33/v08_00_00_33/plots/fhc/"
        beamon_pot = '$2.0\\times10^{20}$' 
        overlay_pot = '$2.320\\times10^{21}$'
        proj_pot = '$9.23\\times10^{20}$'
     
    return beamon_pot, plots_path, overlay_pot, proj_pot
########################################################################
# add angles in beam & detector coordinates
def addAngles(df): 
    
    ## rotation matrix 
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
def offline_flux_weights(mc_df): 
    
    nu_flav = list(mc_df[0]['nu_pdg'])
    angle = list(mc_df[0]['thbeam'])
    true_energy = list(mc_df[0]['nu_e'])

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

    mc_df[0]['weightFlux'] = fluxweights
    mc_df[1]['weightFlux'] = [1 for i in range(len(mc_df[1]))] # for now 
    
    return mc_df
    
########################################################################
# pot scaling weights 
########################################################################
# error on the data/MC ratio 
def get_ratio_err(n_data, n_mc): 

    err = []
    for i in range(len(n_data)): 
        err.append(n_data[i]/n_mc[i]*math.sqrt( (math.sqrt(n_data[i]) / n_data[i])**2 + (math.sqrt(n_mc[i]) / n_mc[i])**2 )) 
    
    return err
########################################################################
# Data/MC comparisons
# NEED TO ADD: GENIE UNSIMS, NON-nueCC DET SYS
def plot_data(var, nbins, xlow, xhigh, cuts, datasets, isrun3, plt_norm='pot', bdt_scale=None, save=False, save_label=None, log=False, x_label=None, ymax=None, sys=False, text=False, xtext=None, ytext=None): 
    
    ############################
    # str var --> variable on x
    # nbins --> # of bins in histogram 
    # xlow, xhigh --> histogram x bounds
    # str cuts --> cuts query applied to data set (string)
    # datasets = list of [df_infv, df_outfv, df_cosmic, df_ext, df_data] in that order 
    # str save_label --> what label to save to pdf as? 
    # bool save --> save as pdf? 
    # bool log --> y log the plot? 
    # str plt_norm ("pot" or "area") --> POT or AREA normalize? 
    # str x_label --> label of the x-axis for the histogram 
    ############################
    
    # set the POT & plots_path for plotting
    beamon_pot = plot_param(isrun3)[0]
    plots_path = plot_param(isrun3)[1]
    proj_pot = plot_param(isrun3)[3]
    
    norm = 'totweight'
    ext_norm = 'pot_scale'
   
    
    # apply cuts 
    if (cuts==""): 
        infv = datasets[0]
        outfv = datasets[1]
        cosmic = datasets[2]
        ext = datasets[3]
        data = datasets[4]
        
    else: 
        infv = datasets[0].query(cuts) 
        outfv = datasets[1].query(cuts)
        cosmic = datasets[2].query(cuts)
        ext = datasets[3].query(cuts)
        data = datasets[4].query(cuts)
    
    # get beam on histogram info 
    v_data, b_data, p_data = plt.hist(data[var], nbins, range=[xlow, xhigh])
    integral_data = sum(v_data)
    data_bins = 0.5*(b_data[1:]+b_data[:-1])
    plt.close()

    # get integral for simulated event spectrum 
    v_sim, b_sim, p_sim = plt.hist([cosmic[var], outfv[var], infv[var], ext[var]], 
                                  nbins, range=[xlow, xhigh], stacked=True, 
                                  weights=[cosmic[norm], outfv[norm], infv[norm], ext[ext_norm]])
    integral_mc = sum(v_sim[-1])
    plt.close()
    
    # weights to be used 
    mc_weights = []
    mc_weights_pot = [ext[ext_norm], 
                      cosmic[norm], 
                      outfv[norm], 
                      infv.query(numu_NC_Npi0)[norm], 
                      infv.query(numu_CC_Npi0)[norm], 
                      infv.query(numu_NC_0pi0)[norm], 
                      infv.query(numu_CC_0pi0)[norm], 
                      infv.query(nue_NC)[norm], 
                      infv.query(nue_CCother)[norm], 
                      #infv.query(numu_Npi0)[norm], infv.query(numu_0pi0)[norm], infv.query(nue_other)[norm],
                      infv.query(nuebar_1eNp)[norm], 
                      infv.query(signal)[norm]]
    
    if bdt_scale is not None:
        mc_weights_pot = [[x/bdt_scale for x in y] for y in mc_weights_pot]

    if (plt_norm=='pot'):
        mc_weights = mc_weights_pot

    elif (plt_norm=='area'):         
        area_scale = integral_data/integral_mc  
        
        for l in mc_weights_pot: 
            mc_weights.append([ k*area_scale for k in l ])

    # event counts
    counts = [round(sum(ext[ext_norm]), 1), 
             round(sum(cosmic[norm]), 1), 
             round(sum(outfv[norm]), 1), 
             round(sum(infv.query(numu_NC_Npi0)[norm]), 1), 
             round(sum(infv.query(numu_CC_Npi0)[norm]), 1), 
             round(sum(infv.query(numu_NC_0pi0)[norm]), 1), 
             round(sum(infv.query(numu_CC_0pi0)[norm]), 1), 
             round(sum(infv.query(nue_NC)[norm]), 1), 
             round(sum(infv.query(nue_CCother)[norm]), 1), 
             round(sum(infv.query(numu_Npi0)[norm]), 1), 
             round(sum(infv.query(numu_0pi0)[norm]), 1), 
             round(sum(infv.query(nue_other)[norm]), 1), 
             round(sum(infv.query(nuebar_1eNp)[norm]), 1), 
             round(sum(infv.query(signal)[norm]), 1)]
    
    if bdt_scale is not None: 
        counts = [ x/bdt_scale for x in counts]

    
    # legend 
    leg = [labels['ext'][0]+': '+str(counts[0]),
                        labels['cosmic'][0]+': '+str(counts[1]), 
                        labels['out_fv'][0]+': '+str(counts[2]), 
                        labels['numu_NC_Npi0'][0]+': '+str(counts[3]), 
                        labels['numu_CC_Npi0'][0]+': '+str(counts[4]), 
                        labels['numu_NC_0pi0'][0]+': '+str(counts[5]), 
                        labels['numu_CC_0pi0'][0]+': '+str(counts[6]), 
                        labels['nue_NC'][0]+': '+str(counts[7]), 
                        labels['nue_CCother'][0]+': '+str(counts[8]), 
                        #labels['numu_Npi0'][0]+': '+str(counts[9]), 
                        #labels['numu_0pi0'][0]+': '+str(counts[10]), 
                        #labels['nue_other'][0]+':  '+str(counts[11]), 
                        labels['nuebar_1eNp'][0]+': '+str(counts[12]),
                        labels['signal'][0]+': '+str(counts[13])
                        ]

    #######################################
    # calclulate the errors before plotting 
    
    stat_err = percent_stat_error(var, nbins, xlow, xhigh, [infv, outfv, cosmic, ext])
    if sys: 
        ppfx_err = calcSysError(var, nbins, xlow, xhigh, cuts, datasets, 'weightsPPFX', 600, plot=False)
        beamline_err = calcSysError(var, nbins, xlow, xhigh, cuts, datasets, 'weightsNuMIGeo', 20, plot=False)
        genie_err = calcSysError(var, nbins, xlow, xhigh, cuts, datasets, 'weightsGenie', 600, plot=False)
        reint_err = calcSysError(var, nbins, xlow, xhigh, cuts, datasets, 'weightsReint', 1000, plot=False)
        #det_err_bkgd = calcDetSysError(var, nbins, intrinsic=False)
        det_err_nueCC = calcDetSysError(var, nbins, intrinsic=True)
        
    #######################################
    
    # Finally, plot for real 
    fig = plt.figure(figsize=(8, 7))

    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
    
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    
    ax1.tick_params(axis = 'both', which = 'major', labelsize = 14)
    ax2.tick_params(axis = 'both', which = 'major', labelsize = 14)
    

    v, b, p = ax1.hist([ext[var], 
                        cosmic[var], 
                        outfv[var], 
                        infv.query(numu_NC_Npi0)[var],
                        infv.query(numu_CC_Npi0)[var],
                        infv.query(numu_NC_0pi0)[var],
                        infv.query(numu_CC_0pi0)[var],
                        infv.query(nue_NC)[var], 
                        infv.query(nue_CCother)[var], 
                        #infv.query(numu_Npi0)[var], 
                        #infv.query(numu_0pi0)[var], 
                        #infv.query(nue_other)[var],
                        infv.query(nuebar_1eNp)[var], 
                        infv.query(signal)[var]], 
            nbins, histtype='bar', range=[xlow, xhigh], stacked=True, 
            color=[labels['ext'][1], 
                        labels['cosmic'][1], 
                        labels['out_fv'][1], 
                        labels['numu_NC_Npi0'][1], 
                        labels['numu_CC_Npi0'][1], 
                        labels['numu_NC_0pi0'][1], 
                        labels['numu_CC_0pi0'][1], 
                        labels['nue_NC'][1], 
                        labels['nue_CCother'][1], 
                        #labels['numu_Npi0'][1], 
                        #labels['numu_0pi0'][1], 
                        #labels['nue_other'][1],
                        labels['nuebar_1eNp'][1], 
                        labels['signal'][1] 
                       ], 
            label=leg, 
            weights=mc_weights, zorder=1)
    
    # scale percent stat error to the (weighted?) # of events in hist bins
    stat_err_scaled = [x*y for x, y in zip(v[-1], stat_err)]
    tot_err = []
    tot_err_percent = []
    
    percent_errors = [stat_err]
    
    if sys:   
        # scale to the (weighted?) number of events 
        ppfx_err_scaled = [x*y for x, y in zip(v[-1], ppfx_err[1])]
        beamline_err_scaled = [x*y for x, y in zip(v[-1], beamline_err[1])]
        genie_err_scaled = [x*y for x, y in zip(v[-1], genie_err[1])]
        reint_err_scaled = [x*y for x, y in zip(v[-1], reint_err[1])]
        #det_err_bkgd_scaled = [x*y for x, y in zip(v[-1], det_err_bkgd[1])]
        det_err_nueCC_scaled = [x*y for x, y in zip(v[-1], det_err_nueCC[1])]
        
        percent_errors += [ppfx_err[1], beamline_err[1], genie_err[1], reint_err[1], #det_err_bkgd[1], 
                           det_err_nueCC[1]]
        
        # add in quadrature the stat error & the sys error 
        tot_err = [ np.sqrt(u**2 + v**2 + w**2 + x**2 + y**2 + z**2) for u,v,w,x,y,z in zip(stat_err_scaled, ppfx_err_scaled, beamline_err_scaled, genie_err_scaled, reint_err_scaled, det_err_nueCC_scaled) ]
        
        #tot_err = [ np.sqrt(t**2 + u**2 + v**2 + w**2 + x**2 + y**2 + z**2) for t,u,v,w,x,y,z in zip(stat_err_scaled, ppfx_err_scaled, beamline_err_scaled, genie_err_scaled, reint_err_scaled, det_err_bkgd_scaled, det_err_nueCC_scaled) ]

        tot_err_percent = [ x/y for x,y in zip(tot_err, v[-1]) ]
            
    else: 
        tot_err = stat_err_scaled
        tot_err_percent = stat_err
    
    #bincenters = 0.5*(b[1:]+b[:-1])
    #ax1.errorbar(bincenters, v[-1], yerr=tot_err, fmt='none', color='black', linewidth=1, zorder=3)
    
    low_err = [ x-y for x,y in zip(v[-1], tot_err) ]
    low_err.insert(0, low_err[0])

    high_err = [ x+y for x,y in zip(v[-1], tot_err)]
    high_err.insert(0, high_err[0])
    
    ax1.fill_between(nbins, low_err, high_err, step="pre", facecolor="peru", edgecolor="peru", 
                     linewidth=0.0, zorder=2, alpha=0.5)
    
    ############################ DATA ############################
    # calculate the width of each bin 
    x_err = [ (b[i+1]-b[i])/2 for i in range(len(b)-1) ]

    ax1.errorbar(data_bins, v_data, yerr=np.sqrt(v_data), xerr=x_err, 
             color="black", fmt='o', markersize=3, label='DATA: '+str(len(data)), zorder=3)
    ax1.set_ylabel("$\\nu$ / "+beamon_pot+ " POT", fontsize=15)
    ax1.set_xlim(xlow, xhigh)
    
    if ymax:
        if log: 
            ax1.set_ylim(1, ymax)
        else: 
            ax1.set_ylim(0, ymax)
            
    #ax1.set_xticks(fontsize=13)
    #ax1.set_xticks([0.1, 0.4, 0.7, 1.0, 1.3, 1.6])

    # ratio plot  
    ax2.errorbar(data_bins, v_data/v[-1], yerr=get_ratio_err(v_data, v[-1]), xerr=x_err, color="black", fmt='o')
    ax2.set_xlim(xlow, xhigh)
    ax2.set_ylim(0.35, 1.65)
    ax2.yaxis.grid(linestyle=':')
    
    # horizontal line at 1 
    ax2.axhline(1.0, color='black', lw=1, linestyle='--')
    
    # MC ratio error - stat + sys 
    low_err_ratio = [ 1 - x for x in tot_err_percent ]
    low_err_ratio.insert(0, low_err_ratio[0])
    
    high_err_ratio = [ 1 + x for x in tot_err_percent ]
    high_err_ratio.insert(0, high_err_ratio[0])

    ax2.fill_between(nbins, low_err_ratio, high_err_ratio, step="pre", facecolor="peru", edgecolor="peru", 
                     linewidth=0.0, zorder=2, alpha=0.5)

    
    if x_label: 
        ax2.set_xlabel(x_label, fontsize=15)
    else: 
        ax2.set_xlabel(var, fontsize=15)
        
    ax2.set_ylabel("DATA / (MC+EXT)", fontsize=15)
    #ax2.set_yticks([0.5, 0.75, 1, 1.25, 1.5])
    #ax2.set_xticks([0.1, 0.4, 0.7, 1.0, 1.3, 1.6])
    
    ax1.legend(prop={"size":10}, ncol=3, frameon=False)
    if log: 
        ax1.set_yscale('log')
        
    if text: 
        ax1.text(xtext, ytext, text, fontsize='xx-large')
    
    if save: 
        if (plt_norm=='area'): 
            plt.savefig(plots_path+var+"_"+save_label+"_area.pdf", transparent=True, bbox_inches='tight') 
        else: 
            plt.savefig(plots_path+var+"_"+save_label+"_pot.pdf", transparent=True, bbox_inches='tight') 

    plt.show()
    
    # returns 2D list of percent errors for each source of systematic, 1D list of total percent errors
    return percent_errors, tot_err_percent
        
    
    
########################################################################
# Return a table of the selection performance 
def sel_perf(cuts, datasets, norm):
    
    
    #################
    # cuts --> list of strings of the cuts applied
    # datasets --> list of dataframes [df_infv, df_outfv, df_cosmic, df_ext, df_data]
    # norm --> normalize to beam ON or overlay? 
    #################
    
    # no cuts on these yet, only separated into their truth categories 
    infv = datasets[0]
    outfv = datasets[1]
    cosmic = datasets[2]
    ext = datasets[3]
    
    ext_norm = ''
    if (norm=='totweight'): 
        ext_norm = 'pot_scale'
    elif (norm=='totweight_overlay'): 
        ext_norm = 'pot_scale_overlay'
            
    df_out = pd.DataFrame(columns=['cut', '# signal after cut',  'efficiency (%)', 'rel. eff. (%)', 
                                'purity (%)', 'purity (MC only, %)'])

    
    # how many true signal generated in FV? (including cosmic contaminated)
    sig_gen_norm = round( sum(infv.query(signal)[norm])+sum(cosmic.query(signal+' and '+in_fv_query)[norm]), 1) 
    print("total # of signal generated in FV : "+ str(sig_gen_norm))
    
    num_signal = []
    pur = []
    pur_mconly = []
    eff = []
    rel_eff = []
    cut_list = []
    
    # start with the number of signal events that are not cosmic contaminated
    sig_last = round( sum(infv.query(signal)[norm]), 1)
    n=0
    
    for cut in cuts: 
        
        infv = infv.query(cut)
        outfv = outfv.query(cut)
        cosmic = cosmic.query(cut)
        ext = ext.query(cut)
        
        # how many true signal gets selected?  
        sig_sel_norm = sum(infv.query(signal)[norm]) 
        
        # how many total selected? 
        # normalize to overlay pot
        tot_sel_norm = sum(infv[norm])+sum(outfv[norm])+sum(cosmic[norm])+sum(ext[ext_norm]) 
        tot_sel_norm_mconly = sum(infv[norm])+sum(outfv[norm])+sum(cosmic[norm]) # do not include EXT
        
        num_signal.append(round(sig_sel_norm, 1))
        
        eff.append(round(sig_sel_norm/sig_gen_norm * 100, 1))
        rel_eff.append(round(sig_sel_norm/sig_last * 100, 1))
        
        pur.append(round(sig_sel_norm/tot_sel_norm * 100, 1))
        pur_mconly.append(round(sig_sel_norm/tot_sel_norm_mconly * 100, 1))
        
        if (n==1): 
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
# FIX error calculation - should use the nue intrinsic event count 
def plot_eff(var, nbins, xlower, xupper, cut, datasets, isrun3, norm='totweight_overlay', save=False, x_label=None, ymax=None): 
         
    # no cuts on these yet, only separated into their truth categories 
    infv = datasets[0]
    outfv = datasets[1]
    cosmic = datasets[2]
    ext = datasets[3]
 
    ext_norm = ''
    
    if (norm=='totweight'): 
        ext_norm = 'pot_scale'
        print("Normalized to data POT")
    elif (norm=='totweight_overlay'): 
        ext_norm = 'pot_scale_overlay'
        print("Normalized to overlay POT")
      
    # true signal generated in FV  # normalized
    signal_gen =  pd.concat([infv.query(signal), cosmic.query(signal+' and '+in_fv_query)], ignore_index=True)
    print("# of generated signal in FV == "+ str(sum(signal_gen[norm])))
    v_sig_gen, b_sig_gen, p_sig_gen = plt.hist(signal_gen[var], nbins, histtype='step', range=[xlower, xupper], 
                                   label='signal generated in FV', weights=signal_gen[norm])
    plt.close()
    
    # apply cuts
    infv_cut = infv.query(cut)
    
    # true signal selected in FV, not cosmic contaminated # normalized
    signal_sel = infv_cut.query(signal)
    print("# of selected signal in FV: "+str( sum(signal_sel[norm] ) ) )

    v_sig_sel, b_sig_sel, p_sig_sel = plt.hist(signal_sel[var], 
                                                              nbins, 
                                                              histtype='step', range=[xlower, xupper], 
                                                              label='signal selected in FV',
                                                              weights=signal_sel[norm])
    plt.close()
    
    ## EFFICIENCY ## 
    eff = [i/j for i, j in zip(v_sig_sel, v_sig_gen)]
    eff_err = []
    for i in range(len(eff)): 
        eff_err.append(math.sqrt( (eff[i]*(1-eff[i]))/v_sig_gen[i] ) )

    bincenters = 0.5*(b_sig_gen[1:]+b_sig_gen[:-1])
 
    binwidth = [abs(b_sig_gen[1]-b_sig_gen[0])/2 for x in range(len(bincenters))]
    print(abs(b_sig_gen[1]-b_sig_gen[0])/2)
    
    fig = plt.figure(figsize=(8, 5))
    plt.errorbar(bincenters, eff, xerr=binwidth, yerr=eff_err, fmt='o', 
             color='seagreen', ecolor='seagreen', markersize=3) 
    
    plt.xlim(xlower, xupper)
    plt.title('')
    plt.grid(linestyle=':')
    
    if x_label: 
        plt.xlabel(x_label, fontsize=15)
    else: 
        plt.xlabel(var, fontsize=15)
    
    if ymax: 
        plt.ylim(0, ymax)
        
    plt.ylabel("Efficiency", fontsize=15)
    plt.tight_layout()

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    plots_path = plot_param(isrun3)[1]
    if save: 
        plt.savefig(plots_path+"eff_"+var+".pdf", transparent=True)
    plt.show()
    #return eff, eff_err, bincenters
########################################################################
# non-POT scaled histogram for stat error counting 
# cuts need to already be applied! 
def percent_stat_error(var, nbins, xlow, xhigh, datasets): 
    
    # combine the dataset - cuts already applied
    selected = pd.concat(datasets, ignore_index=True)

    # plot the histogram 
    n, b, p = plt.hist(selected[var], nbins, histtype='bar', range=[xlow, xhigh], stacked=True)
    plt.close()
    
    # total, non-POT scaled counts ( no GENIE & PPFX weights ) in each bin 
    tot = n

    # stat errors of these 
    tot_err = np.sqrt(tot)

    # finally, compute the percent stat error
    tot_err_per = [y/z for y, z in zip(tot_err, tot)]
    
    # print 
    #print("non scaled # of events")
    #print(n[-1])
    #print('errors')
    #print(tot_err)
    #print('percentage')
    #print(tot_err_per)
    
    return tot_err_per

########################################################################
# Plot MC, normalized to beam on, overlay, OR projected 
# NEED TO ADD: GENIE UNSIMS, NON-nueCC DET SYS
def plot_mc(var, nbins, xlow, xhigh, cuts, datasets, isrun3, plt_norm='overlay', pot=None, save=False, save_label=None, log=False, x_label=None, ymax=None, sys=False, bdt_scale=None, text=None, xtext=None, ytext=None, osc=None, createDict=False):
    
    ############################
    # str var --> variable on x
    # nbins --> # of bins in histogram 
    # xlow, xhigh --> histogram x bounds
    # str cuts --> cuts query applied to data set (string)
    # datasets = list of [df_infv, df_outfv, df_cosmic, df_ext, df_data] in that order 
    # str save_label --> what label to save to pdf as? 
    # bool save --> save as pdf? 
    # bool log --> y log the plot? 
    # str plt_norm --> what POT do we want to scale to: data (totweight), overlay, or proj?
    # str x_label --> label of the x-axis for the histogram 
    ############################
    
    
    # set the POT & plots_path for plotting
    plots_path = plot_param(isrun3)[1]
    
    #if (plt_norm == 'data'): 
    #    pot = plot_param(isrun3)[0]
    #elif (plt_norm=='overlay'): 
    #    pot = plot_param(isrun3)[2]
    #elif (plt_norm =='proj'):
    #    pot = plot_param(isrun3)[3]

    if (cuts==""): 
        infv = datasets[0]
        outfv = datasets[1]
        cosmic = datasets[2]
        ext = datasets[3]
        
    else: 
        infv = datasets[0].query(cuts) 
        outfv = datasets[1].query(cuts)
        cosmic = datasets[2].query(cuts)
        ext = datasets[3].query(cuts) 
    
    ## MC weights
    categories = [cosmic, outfv, 
                        infv.query(numu_NC_Npi0), 
                        infv.query(numu_CC_Npi0), 
                        infv.query(numu_NC_0pi0), 
                        infv.query(numu_CC_0pi0), 
                        infv.query(nue_NC), 
                        infv.query(nue_CCother), 
                        #infv.query(numu_Npi0), 
                        #infv.query(numu_0pi0), 
                        #infv.query(nue_other),
                        infv.query(nuebar_1eNp), 
                        infv.query(signal),
                        ext]
    
    ## Legend counters
    norm = ''
    ext_norm = ''
    
    mc_weights = []
    if (plt_norm=='data'): 
        if bdt_scale is not None: 
            mc_weights = [[ x/bdt_scale for x in d['totweight'] ] for d in categories[:-1]]
            mc_weights.append( [x/bdt_scale for x in categories[-1]['pot_scale']] )
        #else: 
        mc_weights = [d['totweight'] for d in categories[:-1]]
        mc_weights.append( categories[-1]['pot_scale'] )
        
        norm = 'totweight'
        ext_norm = 'pot_scale'
    
    elif (plt_norm=='overlay'):
        if bdt_scale is not None: 
            mc_weights = [[ x/bdt_scale for x in d['totweight_overlay'] ] for d in categories[:-1]]
            mc_weights.append( [x/bdt_scale for x in categories[-1]['pot_scale_overlay']] )
            
        #else: 
        mc_weights = [d['totweight_overlay'] for d in categories[:-1]]
        mc_weights.append( categories[-1]['pot_scale_overlay'] )
        
        norm = 'totweight_overlay'
        ext_norm = 'pot_scale_overlay'
    
    elif (plt_norm=='proj'):
        if bdt_scale is not None: 
            mc_weights = [[ x/bdt_scale for x in d['totweight_proj'] ] for d in categories[:-1]]
            mc_weights.append( [x/bdt_scale for x in categories[-1]['pot_scale_proj']] )
        
        #else:  
        mc_weights = [d['totweight_proj'] for d in categories[:-1]]
        mc_weights.append( categories[-1]['pot_scale_proj'] )
        
        norm = 'totweight_proj'
        ext_norm = 'pot_scale_proj'
        
    
    # event counts
    counts = [round(sum(cosmic[norm]), 1), 
             round(sum(outfv[norm]), 1), 
             round(sum(infv.query(numu_NC_Npi0)[norm]), 1), 
             round(sum(infv.query(numu_CC_Npi0)[norm]), 1), 
             round(sum(infv.query(numu_NC_0pi0)[norm]), 1), 
             round(sum(infv.query(numu_CC_0pi0)[norm]), 1), 
             round(sum(infv.query(nue_NC)[norm]), 1), 
             round(sum(infv.query(nue_CCother)[norm]), 1), 
             round(sum(infv.query(numu_Npi0)[norm]), 1), 
             round(sum(infv.query(numu_0pi0)[norm]), 1), 
             round(sum(infv.query(nue_other)[norm]), 1), 
             round(sum(infv.query(nuebar_1eNp)[norm]), 1), 
             round(sum(infv.query(signal)[norm]), 1), 
             round(sum(ext[ext_norm]), 1)]
    
    if bdt_scale is not None: 
        counts = [ x/bdt_scale for x in counts]

    
    # legend 
    leg = [labels['cosmic'][0]+': '+str(counts[0]), 
                        labels['out_fv'][0]+': '+str(counts[1]), 
                        labels['numu_NC_Npi0'][0]+': '+str(counts[2]), 
                        labels['numu_CC_Npi0'][0]+': '+str(counts[3]), 
                        labels['numu_NC_0pi0'][0]+': '+str(counts[4]), 
                        labels['numu_CC_0pi0'][0]+': '+str(counts[5]), 
                        labels['nue_NC'][0]+': '+str(counts[6]), 
                        labels['nue_CCother'][0]+': '+str(counts[7]),
                        #labels['numu_Npi0'][0]+': '+str(counts[8]), 
                        #labels['numu_0pi0'][0]+': '+str(counts[9]), 
                        #labels['nue_other'][0]+': '+str(counts[10]), 
                        labels['nuebar_1eNp'][0]+': '+str(counts[11]), 
                        labels['signal'][0]+': '+str(counts[12]), 
                        labels['ext'][0]+': '+str(counts[13])]
    
    #######################################
    # calclulate the errors before plotting 
    
    stat_err = percent_stat_error(var, nbins, xlow, xhigh, [infv, outfv, cosmic, ext])
    
    if sys: 
        ppfx_err = calcSysError(var, nbins, xlow, xhigh, cuts, datasets, 'weightsPPFX', 600, plot=False)
        beamline_err = calcSysError(var, nbins, xlow, xhigh, cuts, datasets, 'weightsNuMIGeo', 20, plot=False)
        genie_err = calcSysError(var, nbins, xlow, xhigh, cuts, datasets, 'weightsGenie', 600, plot=False)
        reint_err = calcSysError(var, nbins, xlow, xhigh, cuts, datasets, 'weightsReint', 1000, plot=False)
        #det_err_bkgd = calcDetSysError(var, nbins, intrinsic=False)
        det_err_nueCC = calcDetSysError(var, nbins, intrinsic=True)
        #print(ppfx_err)
        
    #######################################
    
    
    ################################################################## 
    # oscillated event rate
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
        
    ##################################################################  
    
    
    # now to actually plot 
    fig = plt.figure(figsize=(8, 5))
    n, b, p = plt.hist([cosmic[var], 
                       outfv[var], 
                       infv.query(numu_NC_Npi0)[var],
                       infv.query(numu_CC_Npi0)[var],
                       infv.query(numu_NC_0pi0)[var],
                       infv.query(numu_CC_0pi0)[var],
                       infv.query(nue_NC)[var],
                       infv.query(nue_CCother)[var],
                       #infv.query(numu_Npi0)[var], 
                       #infv.query(numu_0pi0)[var], 
                       #infv.query(nue_other)[var],
                       infv.query(nuebar_1eNp)[var], 
                       infv.query(signal)[var],
                       ext[var]],
            nbins, histtype='bar', range=[xlow, xhigh], stacked=True, 
            color=[labels['cosmic'][1], 
                       labels['out_fv'][1], 
                       labels['numu_NC_Npi0'][1], 
                       labels['numu_CC_Npi0'][1], 
                       labels['numu_NC_0pi0'][1], 
                       labels['numu_CC_0pi0'][1], 
                       labels['nue_NC'][1], 
                       labels['nue_CCother'][1],
                       #labels['numu_Npi0'][1], 
                       #labels['numu_0pi0'][1], 
                       #labels['nue_other'][1], 
                       labels['nuebar_1eNp'][1], 
                       labels['signal'][1], 
                       labels['ext'][1]], 
            label=leg,
            weights=mc_weights)
    
    # total selected 
    print('total selected = '+str(sum(n[-1])))
    
    # scale percent stat error to the # of events in hist bins
    stat_err_scaled = [x*y for x, y in zip(n[-1], stat_err)]

    if sys:   
        # scale to the (weighted?) number of events 
        ppfx_err_scaled = [x*y for x, y in zip(n[-1], ppfx_err[1])]
        beamline_err_scaled = [x*y for x, y in zip(n[-1], beamline_err[1])]
        genie_err_scaled = [x*y for x, y in zip(n[-1], genie_err[1])]
        reint_err_scaled = [x*y for x, y in zip(n[-1], reint_err[1])]
        #det_err_bkgd_scaled = [x*y for x, y in zip(n[-1], det_err_bkgd[1])]
        det_err_nueCC_scaled = [x*y for x, y in zip(n[-1], det_err_nueCC[1])]
        
        # add in quadrature the stat error & the sys error 
        tot_err = [ np.sqrt(u**2 + v**2 + w**2 + x**2 + y**2 + z**2) for u,v,w,x,y,z in zip(stat_err_scaled, ppfx_err_scaled, beamline_err_scaled, genie_err_scaled, reint_err_scaled, det_err_nueCC_scaled) ]
        
        #tot_err = [ np.sqrt(t**2 + u**2 + v**2 + w**2 + x**2 + y**2 + z**2) for t, u,v,w,x,y,z in zip(stat_err_scaled, ppfx_err_scaled, beamline_err_scaled, genie_err_scaled, reint_err_scaled, det_err_bkgd_scaled, det_err_nueCC_scaled) ]
        
    else: 
        tot_err = stat_err_scaled
    
    bincenters = 0.5*(b[1:]+b[:-1])
    plt.errorbar(bincenters, n[-1], yerr=tot_err, fmt='none', color='black', linewidth=1)
    
    # error outline 
    tot = list([0, n[-1][0]])+list(n[-1])+[0]
    b_step = list([b[0]])+list(b)+list([b[-1]])
    plt.step(b_step, tot, color='black', linewidth=.7)
      
    ##################################################################  
    if osc:    
        # add in unoscillated background 
        osc_counts = list([0, osc_counts[0]])+osc_counts+[0]
        sig_counts = list([0, n_sig[0]])+list(n_sig)+[0]
        bkgd_counts = [y-z for y, z in zip(tot,sig_counts)]
        osc_counts = [a+b for a,b in zip(osc_counts, bkgd_counts)]
        
        plt.step(b_step, osc_counts, color='darkblue', linestyle='dashed')
    ##################################################################  
        
    # plot format stuff
    plt.legend(loc='best', prop={"size":10}, ncol=3, frameon=False)
        
    if pot is not None: 
        plt.ylabel("$\\nu$ / "+pot+" POT", fontsize=15)
    
    if x_label:
        plt.xlabel(x_label, fontsize=15)
    else: 
        plt.xlabel(var, fontsize=15)
    
    plt.xlim(xlow, xhigh)
    
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    if log: 
        plt.yscale('log')
        
    if ymax: 
        if log: 
            plt.ylim(1, ymax)
        else: 
            plt.ylim(0, ymax)
            
    if text: 
        plt.text(xtext, ytext, text, fontsize='xx-large', horizontalalignment='right')
    
    if save: 
        plt.savefig(plots_path+var+"_"+save_label+".pdf", transparent=True, bbox_inches='tight') 
        print('saving to: '+plots_path)
        
    plt.show()
    ###########################################
    # return python dictionary with bins, CV, & fractional uncertainties 
    d = { 
       "bins" : nbins, 
        "CV" : list(n[-1]), 
        "stat" : stat_err
    }
    
    if sys: 
        d["ppfx"] = ppfx_err[1], 
        d["beamline"] = beamline_err[1],
        d["genie"] = genie_err[1], 
        d["reint"] = reint_err[1],
        d["det_intrinsic"] = det_err_nueCC[1], 
        d["tot"] = [ x/y for x,y in zip(tot_err, n[-1]) ]  
    
    else: 
        d["tot"] = [ x/y for x,y in zip(tot_err, n[-1]) ]  
    ###########################################

    # returns 2D list of percent errors for each source of systematic, 1D list of total percent errors
    return d

########################################################################
# Print out counts for each type of neutrino background inside the FV
def check_counts(in_fv, norm, cuts): 
    
    #################
    # in_fv --> in FV dataframe 
    # norm --> totweight, totweight_overlay, or totweight_proj
    # cuts --> cuts query for the dataframe 
    #################

    infv = in_fv.query(cuts)
    
    print('numu_Npi0 = '+str(round(sum(infv.query(numu_Npi0)[norm]), 1)))
    print('numu_0pi0 = '+str(round(sum(infv.query(numu_0pi0)[norm]), 1)))
    print('nue_other = '+str(round(sum(infv.query(nue_other)[norm]), 1)))
    print('  ')
    print('numu_NC_Npi0 = '+str(round(sum(infv.query(numu_NC_Npi0)[norm]), 1)))
    print('numu_CC_Npi0 = '+str(round(sum(infv.query(numu_CC_Npi0)[norm]), 1)))
    print('numu_NC_0pi0 = '+str(round(sum(infv.query(numu_NC_0pi0)[norm]), 1)))
    print('numu_CC_0pi0 = '+str(round(sum(infv.query(numu_CC_0pi0)[norm]), 1)))
    print('nue_CCother = '+str(round(sum(infv.query(nue_CCother)[norm]), 1)))
    print('nue_NC = '+str(round(sum(infv.query(nue_NC)[norm]), 1)))
    print('  ')
    print('signal = '+str(round(sum(infv.query(signal)[norm]), 1)))
    print('nuebar 1eNp = '+str(round(sum(infv.query(nuebar_1eNp)[norm]), 1)))
    print('  ')
    print('total nue/nuebar = '+str(round(sum(infv.query('nu_pdg==12 or nu_pdg==-12')[norm]), 1)))
    print('total numu/numubar = '+str(round(sum(infv.query('nu_pdg==14 or nu_pdg==-14')[norm]), 1)))
    print('  ')
    print('total  = '+str(round(sum(infv[norm]), 1)))
    
########################################################################
# BDT FUNCTIONS
########################################################################
# makes a copy of MC and EXT dataframes with extra columns 
def addRelevantColumns(datasets, useData=False, ISNUEBAR=False, ):
    
    #datasets = [infv, outfv, cosmic, ext, data]
    
    if len(datasets)>1: 
        mc_bdt = pd.concat(datasets[:3], ignore_index=True)
        ext_bdt = datasets[3]
        data = datasets[4]
    
        mc_bdt['is_mc'] = 1   
        ext_bdt['is_mc'] = 0
    
        if useData:
            mc_bdt['weight'] = mc_bdt['totweight_data']
            ext_bdt['weight'] = ext_bdt['pot_scale_data']
        else:
            mc_bdt['weight'] = mc_bdt['totweight_overlay']
            ext_bdt['weight'] = ext_bdt['pot_scale_overlay']
            
        df_pre = pd.concat([mc_bdt, ext_bdt], ignore_index=True)
        
        nan_var = ['is_mc', 'weight', 'is_cont_signal', 'is_signal']
        for var in nan_var: 
            data[var] = np.nan
    
        
    else: 
        df_pre = datasets[0]

    #cosmic cont. in FV signal definition for convenience in efficiency calculations later
    df_pre['is_cont_signal'] = np.where(((df_pre.nu_pdg == 12) & (df_pre.ccnc == 0) & (df_pre.nproton > 0) & (df_pre.npion == 0) & (df_pre.npi0 == 0)
                                   & (df_pre.nu_purity_from_pfp <= 0.5)
                                   & (10 <= df_pre.true_nu_vtx_x) & (df_pre.true_nu_vtx_x <= 246)
                                    & (-106 <= df_pre.true_nu_vtx_y) & (df_pre.true_nu_vtx_y <= 106)
                                   & (10 <= df_pre.true_nu_vtx_z) & (df_pre.true_nu_vtx_z <= 1026)), 1, 0)

    #true signal definition
    df_pre['is_signal'] = np.where(((df_pre.nu_pdg == 12) & (df_pre.ccnc == 0) & (df_pre.nproton > 0) & (df_pre.npion == 0) & (df_pre.npi0 == 0)
                                   & (df_pre.nu_purity_from_pfp > 0.5)
                                   & (10 <= df_pre.true_nu_vtx_x) & (df_pre.true_nu_vtx_x <= 246)
                                    & (-106 <= df_pre.true_nu_vtx_y) & (df_pre.true_nu_vtx_y <= 106)
                                   & (10 <= df_pre.true_nu_vtx_z) & (df_pre.true_nu_vtx_z <= 1026)), 1, 0)

    if ISNUEBAR:
        #bdt signal definition (doesn't distinguish between nue and nuebar)
        df_pre['is_nuebar_signal'] = np.where((((df_pre.nu_pdg == 12) | (df_pre.nu_pdg == -12)) & (df_pre.ccnc == 0) & (df_pre.nproton > 0) & (df_pre.npion == 0) & (df_pre.npi0 == 0)
                                       & (df_pre.nu_purity_from_pfp > 0.5)
                                       & (10 <= df_pre.true_nu_vtx_x) & (df_pre.true_nu_vtx_x <= 246)
                                        & (-106 <= df_pre.true_nu_vtx_y) & (df_pre.true_nu_vtx_y <= 106)
                                       & (10 <= df_pre.true_nu_vtx_z) & (df_pre.true_nu_vtx_z <= 1026)), 1, 0)

    
    return df_pre
########################################################################
def prep_sets(train, test, query, varlist, ISNUEBAR=False):
    train_query = train.query(query)
    test_query = test.query(query)
    
    # Last column will be signal definition for training, 'is_nuebar_signal' if ISNUEBAR is true, 'is_signal' otherwise
    X_train, y_train = train_query.iloc[:,:-1], train_query.iloc[:,-1]
    
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
    
    return X_train, X_test, dtrain, dtest
########################################################################
def bdt_raw_results(train, test, query, varlist, params, rounds, ISNUEBAR=False):
    queried_train_df, queried_test_df, dtrain, dtest = prep_sets(train, test, query, varlist, ISNUEBAR)
    
    model = xgb.train(params, dtrain, rounds)
    preds = model.predict(dtest)
    
    queried_test_df['is_signal'] = dtest.get_label()
    queried_test_df['BDT_score'] = preds
    
    return queried_test_df, model
########################################################################
# mc = dirt & overlay (with SW trigger applied)
# returns: df with BDT_score, BDT model, training df (before cuts), testing df (before cuts), full combined MC+EXT df (before cuts)
def main_BDT(datasets, query, rounds, test_size=0.5, ISDATA=False, ISNUEBAR=False):
    
    # combine MC & EXT datasets with additional columns needed for BDT analysis
    df_pre = addRelevantColumns(datasets, ISDATA, ISNUEBAR)
    
    # compute the scale weight for model parameters 
    scale_weight = len(df_pre.query(query + ' and is_signal == 0')) / len(df_pre.query(query + ' and is_signal == 1'))
    print("scale weight = "+str(scale_weight))
    
    # Split arrays or matrices into random train and test subsets
    df_pre_train, df_pre_test = train_test_split(df_pre, test_size=test_size, random_state=17, stratify=df_pre['is_signal'])

    varlist = [
    "shr_score", "shrmoliereavg", "trkpid",
    "n_showers_contained", "shr_tkfit_dedx_Y", "tksh_distance",
    "tksh_angle", "subcluster", "trkshrhitdist2"]
    
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
        'scale_pos_weight': 0.55,#scale_weight,
        'eval_metric': ['error', 'auc', 'aucpr']
    }
    
    bdt_results_df, bdt_model  = bdt_raw_results(df_pre_train, df_pre_test, query, varlist, params, rounds, ISNUEBAR=False)
    
    return bdt_results_df, bdt_model, df_pre_train, df_pre_test, df_pre
########################################################################    
# BDT Metric evaluation
def bdt_metrics(train, test, query, ISNUEBAR=False, verbose=False): 
    varlist = [
    "shr_score", "shrmoliereavg", "trkpid",
    "n_showers_contained", "shr_tkfit_dedx_Y", "tksh_distance",
    "tksh_angle", "subcluster", "trkshrhitdist2"]
    
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
        'scale_pos_weight': 0.55,
        'eval_metric': ['error', 'auc', 'aucpr']
    }
    
    dtrain = prep_sets(train, test, query, varlist, ISNUEBAR)[2]
    dtest = prep_sets(train, test, query, varlist, ISNUEBAR)[3]

    watchlist = [(dtrain, 'train'), (dtest, 'valid')]

    progress = dict()

    model = xgb.train(params, dtrain, 1000, watchlist, early_stopping_rounds=50, evals_result=progress, verbose_eval=verbose)

    for metric in progress['train'].keys():
        plt.figure(figsize=(10, 5))
        plt.plot(progress['train'][metric], color='orange', label='train '+metric, markersize=3)
        plt.plot(progress['valid'][metric], color='blue', label='test '+metric, markersize=3)  
        plt.grid(linestyle=":")
        plt.legend(loc='upper right', prop={"size":13})
        plt.xlabel('# of rounds')
        plt.show()
########################################################################
# BDT Purity/Efficiency 
def bdt_pe(df, xvals, full_test_df):#, ISDATA=False):
    
    ########################
    # df --> dataframe with evaluated BDT_score added 
    # xvals --> x axis array
    # full_test_df --> df used for testing (before pre/loose cuts) 
    ########################
    
    purity=[]
    purErr=[]
    eff=[]
    effErr=[]
    
    #if ISDATA:
    #    mc_weight = 'totweight_data'
    #else:
        #mc_weight = 'totweight_overlay'
    
    for cut_val in xvals:
        tot_sel_sig = sum(df.query('is_signal == 1 and BDT_score > '+str(cut_val))['totweight_overlay'])
        tot_sel = sum(df.query('BDT_score > '+str(cut_val))['weight'])
        tot_sig = sum(full_test_df.query('is_signal==1 or is_cont_signal==1')['totweight_overlay'])
        
        p = tot_sel_sig / tot_sel
        purity.append(p * 100)
        purErr.append(math.sqrt(tot_sel_sig) / tot_sel * 100)
        
        e = tot_sel_sig / tot_sig
        eff.append(e * 100)
        effErr.append(math.sqrt((e * (1-e)) / tot_sig) * 100)
    
    return purity, purErr, eff, effErr
########################################################################
# BDT PLOTTING FUNCTIONS
########################################################################
def split_events(df):
    #separate by in/out FV & cosmic 
    infv_bdt = df.query(in_fv_query+' and nu_purity_from_pfp>0.5 and is_mc==1')
    cosmic_bdt = df.query(in_fv_query+' and nu_purity_from_pfp<=0.5 and is_mc==1')
    outfv_bdt = df.query(out_fv_query+' and is_mc==1')
    ext_bdt = df.query('is_mc==0')

    return [infv_bdt, outfv_bdt, cosmic_bdt, ext_bdt]
########################################################################
def bdt_svb_plot(df, is_log=False):
    plt.hist([df.query('is_signal == 1')['BDT_score'], df.query('is_signal == 0')['BDT_score']], 50, histtype='bar', range=[0, 1.0], stacked=True, 
                color=['red','blue'],
                label=['signal','background'],
                log=is_log)
    plt.legend(loc='center right', bbox_to_anchor=(1, 0.5), prop={"size":13})
    plt.xlabel('BDT_score')
    
    plt.show()
########################################################################     
def bdt_pe_plot(df, xvals, full_test_df, isrun3, ISDATA=False, save=False):
    
    ########################
    # df --> dataframe with evaluated BDT_score added 
    # xvals --> x axis array
    # full_test_df --> df used for testing (before pre/loose cuts) 
    ########################
    
    plots_path = plot_param(isrun3)[1]
    
    #plot pur/eff as function of bdt score
    plt.figure(figsize=(7, 5))
    pur, purErr, eff, effErr = bdt_pe(df, xvals, full_test_df)
    plt.errorbar(xvals, pur, yerr=purErr, marker='o', color='blue', label='purity', markersize=3)
    plt.errorbar(xvals, eff, yerr=effErr, marker='o', color='red', label='efficiency', markersize=3)  

    plt.ylabel('Percentage (%)', fontsize=14)
    plt.xlabel('BDT_score > #', fontsize=14)
    plt.grid(linestyle=":")
    plt.xticks(fontsize=12)
    plt.yticks(np.arange(0,105,5), fontsize=12)
    plt.legend(loc='upper left', prop={"size":13})
    plt.tight_layout()
    if save: 
        plt.savefig(plots_path+"BDT_performance.pdf", transparent=True, bbox_inches='tight') 

    plt.show()
########################################################################    
def bdt_box_plot(results_bdt, results_box, xvals, isrun3, second_results_bdt=None, results_box_err=None, save=False):
    
    ###################
    # results_bdt --> BDT performance (output of bdt_pe function)
    # results_box --> [ linear sel. purity, lineear sel. efficiency ]
    # xvals --> values on x-axis
    # second_results_bdt --> BDT performance using preselection cuts ONLY (as opposed to loose cuts)
    ###################
    
    plots_path = plot_param(isrun3)[1]
    
    #plot pur/eff as function of bdt score
    plt.figure(figsize=(7, 5))
    
    # Loose cut BDT results
    pur, purErr, eff, effErr = results_bdt
    plt.errorbar(xvals, pur, yerr=purErr, marker='o', color='maroon', label='Loose BDT Purity', markersize=3)
    plt.errorbar(xvals, eff, yerr=effErr, marker='o', color='green', label='Loose BDT Eff.', markersize=3)  
    
    # Preselection BDT results
    if second_results_bdt is not None:
        pur2, purErr2, eff2, effErr2 = second_results_bdt
        plt.errorbar(xvals, pur2, yerr=purErr2, marker='o', color='salmon', label='Pre BDT Purity', markersize=3)
        plt.errorbar(xvals, eff2, yerr=effErr2, marker='o', color='mediumseagreen', label='Pre BDT Eff.', markersize=3)
        
    # Box cut results 
    plt.axhline(results_box[0], color='red', linestyle='dashed', label='Lin. Sel. Purity ('+str(round(results_box[0], 2))+'%)', linewidth=2)
    plt.axhline(results_box[1], color='limegreen', linestyle='dashed', label='Lin. Sel. Eff. ('+str(round(results_box[1], 2))+'%)', linewidth=2)
    
    if results_box_err is not None:
        
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
    plt.legend(prop={"size":12})
    #plt.tight_layout()
    if save: 
        plt.savefig(plots_path+"BDT_performance_linear.pdf", transparent=True, bbox_inches='tight') 

    
    plt.show()   
########################################################################
# caluclate systematic error (normalized to data POT) based on variation weights assigned to each bin 
# return the fractional systematic uncertainty for each bin 
def calcDetSysError(var, bins, intrinsic=False, plot=False, plot_cov=False, save=False, axis_label=None, pot=None): 
    
    # get variation counts & CV counts - from NuMIDetSys.py 
    uni_counts, ncv, weights = NuMIDetSysWeights.ratio_to_CV(var, bins, intrinsic=intrinsic)
    
    if plot: 
        NuMIDetSysWeights.plot_variations(var, bins, intrinsic=intrinsic, axis_label=axis_label, save=save, pot=pot)
    
    # compute covariance & correlation 
    cov, cor, d = calcCov(var, bins, ncv, uni_counts, unisim=True, plot=plot_cov, save=save, axis_label=axis_label, pot=pot)
    
    # return a list of values of the fractional systematic uncertainty
    sys_err = [np.sqrt(x) for x in np.diagonal(cov)]
    percent_sys_error = [y/z for y,z in zip(sys_err, ncv)]
    
    frac = []
    for k in d: 
        frac.append( [y/z for y,z in zip(k, ncv)] ) # turn into a fractional error 

    return sys_err, percent_sys_error, cov, cor, frac
    
########################################################################
# calculate systematic error (normalized to data POT) based on variation weights assigned to each event
# return the fractional systematic uncertainty for each bin
# does not include EXT events - neutrino only 
def calcSysError(var, bins, xlow, xhigh, cuts, datasets, sys_var, universes, isrun3=False, plot=False, plot_cov=False, save=False, axis_label=None, ymax=None, pot=None, text=None, xtext=None, ytext=None): 
    
    # datasets 
    # infv - overlay & dirt events with truth vtx in FV 
    # outfv - overlay & dirt events with truth vtx in FV that are classified as neutrinos
    # cosmic - overlay & dirt events with true vtx in FV that get misclassified as cosmic 
    # ext - beam off data
    # data - won't use here
    
    weight = 'totweight' # scale to data POT
    #ext_weight = 'pot_scale' # scale to data POT
    
    plots_path = plot_param(isrun3)[1]
    
    # apply cuts 
    if cuts == '': 
        infv = datasets[0].copy()
        outfv = datasets[1].copy()
        cosmic = datasets[2].copy()
        #ext = datasets[3].copy()
        
    else: 
        infv = datasets[0].copy().query(cuts)
        outfv = datasets[1].copy().query(cuts)
        cosmic = datasets[2].copy().query(cuts)
        #ext = datasets[3].copy().query(cuts)
    
    
    # stat errors - for plotting later 
    per = percent_stat_error(var, bins, xlow, xhigh, [infv, outfv, cosmic])#, ext])
    
    nu_selected = pd.concat([infv, outfv, cosmic], ignore_index=True)

    # histogram bin counts for all universes
    uni_counts = []

    fig = plt.figure(figsize=(8, 5))
    for u in range(universes): 
    
        # CV weight only 
        nu_selected['weight'] = nu_selected[weight]
        #ext['weight'] = ext[ext_weight]
        
        # multiply in with sys weight 
        sys_weight = list(nu_selected[sys_var].str.get(u))

        # for GENIE systematics, replace the tune weight
        if sys_var == 'weightsGenie':  
            tot_weight = [ x*y for x, y in zip(sys_weight, nu_selected['weight']/nu_selected['weightTune']) ]
            
        else: 
            tot_weight = [ x*y for x, y in zip(sys_weight, nu_selected['weight']) ]
        
        for i in range(len(sys_weight)): 
            if np.isnan(sys_weight[i])==True: 
                print('NaN for index '+str(i))
        
        nu_selected['weight_sys'] = tot_weight
        #ext['weight_sys'] = ext['weight']
        
        full_selected = nu_selected #pd.concat([nu_selected], ext], ignore_index=True)

        # plot variation
        n, b, p = plt.hist(full_selected[var], bins, histtype='step', weights=full_selected['weight_sys'], 
                            linewidth=0.5, color='cornflowerblue')
        uni_counts.append(n)

            
    ncv, bcv, pcv = plt.hist(full_selected[var], bins, histtype='step', 
                             weights=full_selected['weight'], linewidth=2, color='black')
    

    stat_err_scaled = [x*y for x, y in zip(ncv, per)]
    plt.errorbar(0.5*(bcv[1:]+bcv[:-1]), ncv, yerr=stat_err_scaled, fmt='none', color='black', linewidth=2)
    plt.title(sys_var, fontsize=15)
    
    if plot: 
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        
        if axis_label is not None: 
            plt.xlabel(axis_label, fontsize=15)
        else: 
            plt.xlabel(var, fontsize=15)
        if pot: 
            plt.ylabel("$\\nu$ / "+pot, fontsize=15)
        if ymax is not None: 
            plt.ylim(0, ymax)
            
        if text: 
            plt.text(xtext, ytext, text, fontsize='xx-large')
            
        if save: 
            plt.savefig(plots_path+var+"_"+sys_var+".pdf", transparent=True, bbox_inches='tight') 
            print('saving to: '+plots_path)
            
        plt.show()
    
    if not plot: 
        plt.close()
        
    # compute covariance & correlation 
    if sys_var == "weightsNuMIGeo": 
        unisim = True
    else: 
        unisim = False
        
    cov, cor, d = calcCov(var, bins, ncv, uni_counts, unisim=unisim, plot=plot_cov, save=save, axis_label=axis_label, pot=pot)
    
    # return a list of values of the fractional systematic uncertainty 
    sys_err = [np.sqrt(x) for x in np.diagonal(cov)]
    percent_sys_error = [y/z for y,z in zip(sys_err, ncv)]
    
    frac = []
    for k in d: 
        frac.append( [y/z for y,z in zip(k, ncv)] ) # turn into a fractional error 
               
    return sys_err, percent_sys_error, cov, cor, frac
    
########################################################################
# compute covariance & correlation matrices 
# returns cov & cor - UPDATE: need to correct correlation calculation
def calcCov(var, bins, ncv, uni_counts, unisim=False, plot=False, save=False, axis_label=None, pot=None, isrun3=False): 
    
    plots_path = plot_param(isrun3)[1]
    
    # compute the cov matrix 
    cov = [ [0]*(len(bins)-1) for x in range(len(bins)-1) ]

    # unisims: N=1, multisims N = number of universes
    if unisim: 
        N = 1
    else: 
        N = len(uni_counts)
    
    d = [] # 2D list of uncertainty of each bin for each variation - for unisim variations ONLY
    for k in range(len(uni_counts)): 
        uni = uni_counts[k]
        d_k = [] # uncertainty of each bin for variation k 

        for i in range(len(bins)-1): 
            cvi = ncv[i]
            uvi = uni[i]

            for j in range(len(bins)-1): 
                cvj = ncv[j]
                uvj = uni[j]
        
                c = ((uvi - cvi)*(uvj - cvj)) / N
                cov[i][j] += c
                
                if unisim and i==j: 
                    d_k.append(np.sqrt(c)) # for diagonals 
        
        if unisim: 
            d.append(d_k)
                    
    if plot: 
        fig = plt.figure(figsize=(10, 6))
        
        plt.pcolor(bins, bins, cov, cmap='OrRd', edgecolors='k')
            
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=14)
        if pot: 
            cbar.set_label(label="$\\nu$ / "+pot, fontsize=15)
        
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        if axis_label is not None: 
            plt.xlabel(axis_label, fontsize=15)
            plt.ylabel(axis_label, fontsize=15)
        else: 
            plt.xlabel(var, fontsize=15)
            plt.ylabel(var, fontsize=15)

        plt.title('Covariance', fontsize=15)
        
        if save: 
            plt.savefig(plots_path+var+"_cov.pdf", transparent=True, bbox_inches='tight') 
            print('saving to: '+plots_path)
        plt.show()
            
    # compute the corr matrix 
    cor = [ [0]*(len(bins)-1) for x in range(len(bins)-1) ]

    for i in range(len(cov)): 
        cii = cov[i][i]
    
        for j in range(len(cov[i])): 
            cjj = cov[j][j]
            n = np.sqrt(cii*cjj)
            cor[i][j] = cov[i][j] / n 
            #print(cor[i][j])
    
    if plot: 
        fig = plt.figure(figsize=(10, 6))

        plt.pcolor(bins, bins, cor, cmap='OrRd', edgecolors='k')#, vmin=0.8, vmax=1)
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

        plt.title('Correlation', fontsize=15)
        if save: 
            plt.savefig(plots_path+var+"_corr.pdf", transparent=True, bbox_inches='tight') 
        plt.show()
            
    return cov, cor, d

########################################################################
# plot the full covariance & correlation matrices 
# UPDATE with other sources of systematic error (only includes flux right now)
def plotFullCov(var, bins, xlow, xhigh, cuts, datasets, save=False, axis_label=None, isrun3=False, pot=None): 
    
    plots_path = plot_param(isrun3)[1]
    
    ppfx_sys = calcSysError(var, bins, xlow, xhigh, cuts, datasets, 'weightsPPFX', 600, 
                            isrun3=isrun3, save=save, axis_label=axis_label)
    
    geom_sys = calcSysError(var, bins, xlow, xhigh, cuts, datasets, 'weightsNuMIGeo', 20, 
                            isrun3=isrun3, save=save, axis_label=axis_label)
    
    
    # add the covariance matrices & plot 
    ppfx_cov = ppfx_sys[2]
    geom_cov = geom_sys[2]

    tot_cov = [ [ x+y for x,y in zip(a, b) ] for a, b in zip(ppfx_cov, geom_cov) ]
    
    fig = plt.figure(figsize=(10, 6))
        
    plt.pcolor(bins, bins, tot_cov, cmap='OrRd', edgecolors='k')
    cbar = plt.colorbar()
    if pot: 
        cbar.set_label(label="$\\nu$ / "+pot, fontsize=15)
    cbar.ax.tick_params(labelsize=14)
        
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    if axis_label is not None: 
        plt.xlabel(axis_label, fontsize=15)
        plt.ylabel(axis_label, fontsize=15)
    else: 
        plt.xlabel(var, fontsize=15)
        plt.ylabel(var, fontsize=15)

    plt.title('Covariance', fontsize=15)
    if save: 
            plt.savefig(plots_path+var+"_tot_cov.pdf", transparent=True, bbox_inches='tight') 
    plt.show()

    
    # compute the correlation matrix & plot
    cor = [ [0]*(len(bins)-1) for x in range(len(bins)-1) ]

    for i in range(len(tot_cov)): 
        cii = tot_cov[i][i]
    
        for j in range(len(tot_cov[i])): 
            cjj = tot_cov[j][j]
            n = np.sqrt(cii*cjj)
            cor[i][j] = tot_cov[i][j] / n 
    

    fig = plt.figure(figsize=(10, 6))

    plt.pcolor(bins, bins, cor, cmap='OrRd', edgecolors='k')#, vmin=0.5, vmax=1)
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

    plt.title('Correlation', fontsize=15)
    if save: 
        plt.savefig(plots_path+var+"_tot_corr.pdf", transparent=True, bbox_inches='tight') 
    plt.show()
    
    
    
    
    
########################################################################
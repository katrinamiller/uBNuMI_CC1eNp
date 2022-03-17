# important parameters & some top level functions for the xsec analysis 

import math
import warnings
import numpy as np
import pandas as pd
import uproot

import matplotlib.pyplot as plt


# FULL SIGNAL DEFINITION 
#### passes software trigger 
#### 'nu_pdg==12 and ccnc==0 
#### 1 proton > 40 MeV 
#### no pions above 40 MeV 
#### within a FV defined by: 10<=true_nu_vtx_x<=246 and -106<=true_nu_vtx_y<=106 and 10<=true_nu_vtx_z<=1026'


# applied tunes 
dirt_tune = 0.35
ext_tune = 1



######################### analysis parameters ##############################
# set the POT & plots_path for plotting
# UPDATE based on the ntuples being used 
def parameters(ISRUN3): 
    
    dirt_tune = 0.35
    
    rho_argon = 1.3836 # g/cm^3
    fv = 236*212*1016
    n_a = 6.022E23
    n_nucleons = 40
    m_mol = 39.95 #g/mol
    
    n_target = (rho_argon * fv * n_a * n_nucleons) / m_mol
    
    bdt_training_parameters = [
        "shr_score", "shrmoliereavg", "trkpid",
        "shr_tkfit_dedx_Y", "tksh_distance", 
        "subcluster", "trkshrhitdist2"]
    
   # FHC
    if not ISRUN3:  
        plots_path = "/uboone/data/users/kmiller/uBNuMI_CCNp/plots/fhc/"
        cv_ntuple_path = "/uboone/data/users/kmiller/uBNuMI_CCNp/ntuples/run1/cv_slimmed/nuepresel/"
        
        ext_tune = .98
        
        overlay_pot = 2.33652E21
        intrinsic_pot = 2.37838E22
        
        dirt_pot = 1.67392E21 # david's file
        beamon_pot = 2.0E20 # v5
    
        # proj_pot = 4.125E20 # FHC Runs 1-5: 9.23E20, FHC Runs 1-3: 4.125E20 

        beamon_ntrig =  5268051.0 # v5 (EA9CNT_wcut)
        beamoff_ntrig = 9199232.74  # v5 (EXT_NUMIwin_FEMBeamTriggerAlgo)
        
        NUE = 'neutrinoselection_filt_run1_overlay_intrinsic_v7' 
        
        integrated_flux_per_pot = 1.18069E-11 # needs to be scaled to appropriate POT
        
        bdt_model = 'BDT_models/bdt_FHC_FEB2022.model' # removes n_showers_contained from training parameters
        bdt_score_cut = 0.5
    
    # RHC 
    else: 
        #plots_path = "/uboone/data/users/kmiller/searchingfornues_v33/v08_00_00_33/plots/rhc/"
        #nue_path = "/uboone/data/users/kmiller/ntuples/run3b/"
        
        plots_path = "/uboone/data/users/kmiller/uBNuMI_CCNp/plots/rhc/"
        cv_ntuple_path = "/uboone/data/users/kmiller/uBNuMI_CCNp/ntuples/run3b/cv_slimmed/nuepresel/"
        
        ext_tune = .94
        
        overlay_pot = 1.98937E21
        intrinsic_pot = 2.5345E22 # v7
        
        dirt_pot = 1.03226E21
        beamon_pot = 5.0E20
    
        # proj_pot = 8.624E20 # RHC Runs 1-5: 11.95E20, RHC Runs 1-3: 8.624E20 

        beamon_ntrig = 10363728.0 #v5
        beamoff_ntrig = 32878305.25 # v5
        
        NUE = 'neutrinoselection_filt_run3b_overlay_intrinsic_v7' 
        
        integrated_flux_per_pot = 1
        
        bdt_model = 'BDT_models/bdt_RHC_oct2021.model'
        bdt_score_cut = 0.575
        
    # create a dictionary 
    d = { 
        "plots_path" : plots_path, 
        "cv_ntuple_path" : cv_ntuple_path, 
        "dirt_tune" : dirt_tune, 
        "ext_tune" : ext_tune, 
        "overlay_pot" : overlay_pot, 
        "intrinsic_pot" : intrinsic_pot, 
        "dirt_pot" : dirt_pot, 
        "beamon_pot" : beamon_pot, 
        "beamon_ntrig" : beamon_ntrig, 
        "beamoff_ntrig" : beamoff_ntrig,
        "NUE" : NUE, 
        "integrated_flux_per_pot" : integrated_flux_per_pot, 
        "n_target" : n_target,
        "bdt_model" : bdt_model, 
        "bdt_training_parameters" : bdt_training_parameters, 
        "bdt_score_cut" : bdt_score_cut
    }
    
    return d


######################### plot categories ##############################
# for events that are not cosmic contaminated

in_fv_query = "10<=true_nu_vtx_x<=246 and -106<=true_nu_vtx_y<=106 and 10<=true_nu_vtx_z<=1026"
out_fv_query = "((true_nu_vtx_x<10 or true_nu_vtx_x>246) or (true_nu_vtx_y<-106 or true_nu_vtx_y>106) or (true_nu_vtx_z<10 or true_nu_vtx_z>1026))"

reco_in_fv_query = "10<=reco_nu_vtx_sce_x<=246 and -106<=reco_nu_vtx_sce_y<=106 and 10<=reco_nu_vtx_sce_z<=1026"

numu_CC_Npi0 = '(nu_purity_from_pfp>0.5) and ((nu_pdg==14 or nu_pdg==-14) and ccnc==0 and npi0>=1)'
numu_CC_0pi0 = '(nu_purity_from_pfp>0.5) and ((nu_pdg==14 or nu_pdg==-14) and ccnc==0 and npi0==0)'

numu_NC_Npi0 = '(nu_purity_from_pfp>0.5) and ((nu_pdg==14 or nu_pdg==-14) and ccnc==1 and npi0>=1)'
numu_NC_0pi0 = '(nu_purity_from_pfp>0.5) and ((nu_pdg==14 or nu_pdg==-14) and ccnc==1 and npi0==0)'

nuebar_1eNp = '(nu_purity_from_pfp>0.5) and ((nu_pdg==-12 and ccnc==0 and nproton>0 and npion==0 and npi0==0))'
nue_NC = '(nu_purity_from_pfp>0.5) and ((nu_pdg==12 or nu_pdg==-12) and ccnc==1)'

nue_CCother = '(nu_purity_from_pfp>0.5) and (((nu_pdg==12 and ccnc==0) and (nproton==0 or npi0>0 or npion>0)) or (nu_pdg==-12 and ccnc==0 and (nproton==0 or npion>0 or npi0>0)))'

# less specific categories 
nue_other = '(nu_purity_from_pfp>0.5) and (((nu_pdg==12 or nu_pdg==-12) and ccnc==1) or (((nu_pdg==12 and ccnc==0) and (nproton==0 or npi0>0 or npion>0)) or (nu_pdg==-12 and ccnc==0)))'
numu_Npi0 = '(nu_purity_from_pfp>0.5) and ( (nu_pdg==14 or nu_pdg==-14) and npi0>=1)'
numu_0pi0 = '(nu_purity_from_pfp>0.5) and ( (nu_pdg==14 or nu_pdg==-14) and npi0==0)'

# signal vs. not signal 
signal = in_fv_query+' and  (nu_pdg==12 and ccnc==0 and nproton>0 and npion==0 and npi0==0)'
not_signal = out_fv_query+' or (nu_pdg!=12) or (nu_pdg==12 and ccnc==1) or (nu_pdg==12 and ccnc==0 and (nproton==0 or npi0>0 or npion>0))'

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
    'outfv' : ['Out FV', 'orchid'], 
    #'cosmic' : ['Cosmic Cont.', 'lightpink'],
    'ext' : ['EXT', 'gainsboro'], 
    'nue_other' : ['$\\nu_e$ / $\\overline{\\nu_e}$  other', '#33db09'], 
    'numu_Npi0' : ['$\\nu_\\mu$ / $\\overline{\\nu_\\mu}$  $\pi^{0}$', '#EE1B1B'], 
    'numu_0pi0' : ['$\\nu_\\mu$ / $\\overline{\\nu_\\mu}$  other', '#437ED8'],
    'nuebar_1eNp' : ['$\\bar{\\nu}_e$ CC0$\pi$Np', 'gold']
}
########################################################################
# MC Stat Error Counting (unweighted, unscaled histogram)
def mc_stat_error(var, nbins, xlow, xhigh, datasets): 
    
    #### combine the datasets - cuts should already be applied ####
    selected = pd.concat(datasets, ignore_index=True, sort=True)

    ##### MC statistical uncertainty ####
    n, b, p = plt.hist(selected[var], nbins, histtype='bar', range=[xlow, xhigh], stacked=True)     # plot the histogram 
    plt.close()
    
    mc_total_error = np.sqrt(n)

    #### compute the percent stat error ####
    mc_percent_error = [y/z for y, z in zip(mc_total_error, n)]
    
    return mc_percent_error
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
# detector variation - POT values
detvar_run1_fhc = {
    "LYAttenuation": 7.51336E20,
    "LYRayleigh": 7.60573E20, 
    "LYDown": 7.43109E20, 
    "SCE": 7.39875E20, 
    "Recomb2": 7.59105E20, 
    "WireModX": 7.64918E20, 
    "WireModYZ": 7.532E20, 
    "WireModThetaXZ": 7.64282E20,
    "WireModThetaYZ_withSigmaSplines": 7.64543E20, 
    "WireModThetaYZ_withoutSigmaSplines": 7.5783E20, 
    "CV": 7.59732E20
}

intrinsic_detvar_run1_fhc = {
    "LYAttenuation_intrinsic": 2.3837E22, 
    "LYRayleigh_intrinsic": 2.38081E22, 
    "LYDown_intrinsic": 2.24505E22, 
    "SCE_intrinsic": 2.39023E22, 
    "Recomb2_intrinsic": 2.38193E22, 
    "WireModX_intrinsic": 2.38318E22, 
    "WireModYZ_intrinsic": 2.38416E22,
    "WireModThetaXZ_intrinsic": 2.31518E22, 
    "WireModThetaYZ_withSigmaSplines_intrinsic": 2.31421E22, 
    "WireModThetaYZ_withoutSigmaSplines_intrinsic": 2.31755E22, 
    "CV_intrinsic": 2.37261E22   
}
########################################################################
# scales to standard overlay 
def generated_signal(ISRUN3, var, bins, xlow, xhigh, cuts=None, weights=None):
    
    print('generated_signal scales to standard overlay!')
        
    fold = "nuselection"
    tree = "NeutrinoSelectionFilter"
    
    variables = ["swtrig_pre", "nu_pdg", "ccnc", "nproton", "npion", "npi0", 
                "true_nu_vtx_x", "true_nu_vtx_y", "true_nu_vtx_z", "ppfx_cv", "weightSplineTimesTune", 
                "nu_purity_from_pfp", "nslice", "reco_nu_vtx_sce_x", 
                 "reco_nu_vtx_sce_y","reco_nu_vtx_sce_z", "contained_fraction", 
                "shr_energy_tot_cali", "opening_angle"]
    
    if var not in variables: 
        variables.append(var)
        
    overlay_pot = parameters(ISRUN3)['overlay_pot']
    intrinsic_pot = parameters(ISRUN3)['intrinsic_pot']
        
    
    f = uproot.open(parameters(ISRUN3)['cv_ntuple_path']+parameters(ISRUN3)['NUE']+".root")[fold][tree]
    df = f.pandas.df(variables, flatten=False)
    
    df.loc[ df['weightSplineTimesTune'] <= 0, 'weightSplineTimesTune' ] = 1.
    df.loc[ df['weightSplineTimesTune'] == np.inf, 'weightSplineTimesTune' ] = 1.
    df.loc[ df['weightSplineTimesTune'] > 100, 'weightSplineTimesTune' ] = 1.
    df.loc[ np.isnan(df['weightSplineTimesTune']) == True, 'weightSplineTimesTune' ] = 1.
    
    df['is_signal'] = np.where((df.swtrig_pre == 1) 
                             & (df.nu_pdg==12) & (df.ccnc==0) & (df.nproton>0) & (df.npion==0) & (df.npi0==0)
                             & (10 <= df.true_nu_vtx_x) & (df.true_nu_vtx_x <= 246)
                             & (-106 <= df.true_nu_vtx_y) & (df.true_nu_vtx_y <= 106)
                             & (10 <= df.true_nu_vtx_z) & (df.true_nu_vtx_z <= 1026), True, False)
    
    df_signal = df.query('is_signal==True').copy()
    df_signal['pot_scale'] = overlay_pot/intrinsic_pot

    df_signal['totweight_overlay'] = df_signal['ppfx_cv']*df_signal['weightSplineTimesTune']*df_signal['pot_scale']
    
    if cuts: 
        df_signal = df_signal.query(cuts)
     
    # make a histogram of the generated SIGNAL ONLY events 
    n, b, p = plt.hist(df_signal[var], bins, histtype='bar', range=[xlow, xhigh], weights=df_signal['totweight_overlay'])
    plt.close()
    
    return n.tolist()
    
########################################################################
# parameters for the xsec variables 
def xsec_variables(xvar, ISRUN3): 
    
    if not ISRUN3: 
        data_pot = "$2.0\\times10^{20}$ POT"
    else: 
        print('No parameters for RHC! ') 

    if xvar == 'tksh_angle': 
        bins = [-1, -0.6, -0.2, 0.2, 0.6, 1]
        fine_bins = [-1, -0.9, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 0.9, 1]
        true_var = 'opening_angle'
        x_label = "cos $\\theta_{ep}$"

        xlow = -1
        xhigh = 1
        
    elif xvar=='shr_energy_cali': 
        bins = [0.09, 0.4, 0.65, 1, 3]
        fine_bins = [0.09, .2, .3, .4, .5, .65, .75, .85, 1.0, 1.5, 2, 2.5, 3]
        true_var = "elec_e"
        x_label = "Electron Energy [GeV]" 
        xlow = 0.09
        xhigh = 3
        
    elif xvar=='NeutrinoEnergy2_GeV': 
        bins = [0.19, .4, .65, .85, 1.15, 1.5, 4]
        fine_bins = [.19, .5, .75, 1, 1.5, 2, 2.5, 3, 3.5, 4]
        true_var = 'true_e_visible'
        x_label = "Total Visible Energy [GeV]" 
        xlow = 0
        xhigh = 4
        
    elif xvar=='nproton': 
        bins = [1, 2, 3, 7]
        fine_bins = [1, 2, 3, 4, 5, 6, 7]
        true_var = "nproton"
        x_label = "Proton Multiplicity"
        xlow = 1
        xhigh = 7
    
    else: 
        print('No parameters for this variable! ')

    d = {
        'bins': bins, 
        'fine_bins': fine_bins, 
        'true_var': true_var, 
        'x_label': x_label, 
        'beamon_pot': data_pot, 
        'xlow': xlow,
        'xhigh': xhigh
    }
    
    return d


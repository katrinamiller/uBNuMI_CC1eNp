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

reco_in_fv_query = "10<=reco_nu_vtx_sce_x<=246 and -106<=reco_nu_vtx_sce_y<=106 and 10<=reco_nu_vtx_sce_z<=1026"

training_parameters = [
        "shr_score", "shrmoliereavg", "trkpid",
        "shr_tkfit_dedx_Y", "tksh_distance", 
        "subcluster", "trkshrhitdist2"]
    
selection_variables = ['nslice', "reco_nu_vtx_sce_x", "reco_nu_vtx_sce_y", "reco_nu_vtx_sce_z", 
                      "contained_fraction",  'n_tracks_contained', 
                       'trk_energy', 'shr_score', 'shrmoliereavg', 'trkpid', 
                      'n_showers_contained', 'shr_tkfit_dedx_Y', 'tksh_distance', 
                       'tksh_angle', 'trkshrhitdist2', 'subcluster']

# quality cuts
BDT_PRE_QUERY = 'swtrig_pre==1 and nslice==1'
BDT_PRE_QUERY += ' and ' + reco_in_fv_query
BDT_PRE_QUERY +=' and contained_fraction>0.9'

# signal definition - shower constraints
BDT_PRE_QUERY += ' and n_showers_contained==1'

# signal definition - track constraints
BDT_PRE_QUERY += ' and n_tracks_contained>0'
BDT_PRE_QUERY += ' and trk_energy>0.04' 
    
BDT_LOOSE_CUTS = BDT_PRE_QUERY

# loose shower constraints
BDT_LOOSE_CUTS +=' and shr_score<0.3'
BDT_LOOSE_CUTS += ' and shrmoliereavg<15'
BDT_LOOSE_CUTS += ' and shr_tkfit_dedx_Y<7'

# loose track constraints
BDT_LOOSE_CUTS += ' and trkpid<0.35'
BDT_LOOSE_CUTS += ' and tksh_distance<12'



######################### analysis parameters ##############################
# set the POT & plots_path for plotting
# UPDATE based on the ntuples being used 
def parameters(ISRUN3): 
    
    rho_argon = 1.3836 # g/cm^3
    fv = 236*212*1016
    n_a = 6.022E23
    n_nucleons = 40
    m_mol = 39.95 #g/mol
    
    n_target = (rho_argon * fv * n_a * n_nucleons) / m_mol
    
    ext_tune = 0.98 # validated

    
   # FHC
    if not ISRUN3: 
        
        plots_path = "/uboone/data/users/kmiller/uBNuMI_CCNp/plots/fhc/"
        cv_ntuple_path = "/uboone/data/users/kmiller/uBNuMI_CCNp/ntuples/run1/cv_slimmed/qualcuts/" 
        full_ntuple_path = "/uboone/data/users/kmiller/uBNuMI_CCNp/ntuples/run1/cv/"
        
        dirt_tune = 0.65 # validated
        
        beamon_pot = 2.0E20 # v5

        NUE = 'neutrinoselection_filt_run1_overlay_intrinsic_v7' 
        
        integrated_flux_per_pot = 1.1864531e-11 # 1.18069E-11 # [ nu / cm^2 / POT]  , includes 60 MeV neutrino energy threshold
        
        bdt_model = 'BDT_models/bdt_FHC_may2022_subset.model' 
        bdt_score_cut = 0.55
        
        detsys = 0.122
    
    # RHC 
    else: 
        #plots_path = "/uboone/data/users/kmiller/searchingfornues_v33/v08_00_00_33/plots/rhc/"
        #nue_path = "/uboone/data/users/kmiller/ntuples/run3b/"
        
        plots_path = "/uboone/data/users/kmiller/uBNuMI_CCNp/plots/rhc/"
        cv_ntuple_path = "/uboone/data/users/kmiller/uBNuMI_CCNp/ntuples/run3b/cv_slimmed/qualcuts/"
        full_ntuple_path = "/uboone/data/users/kmiller/uBNuMI_CCNp/ntuples/run3b/cv/"
        
        dirt_tune = 0.45 # validated
        
        beamon_pot = 5.014E20
        
        NUE = 'neutrinoselection_filt_run3b_overlay_intrinsic_v7' 
        
        integrated_flux_per_pot =  8.6283762e-12 #3.2774914e-12 # [ nu / cm^2 / POT]  , includes 60 MeV neutrino energy threshold
        
        bdt_model = 'BDT_models/bdt_RHC_may2022_subset.model'
        bdt_score_cut = 0.575 
        
        detsys = 0.129 #0.133
        
    # create a dictionary 
    d = { 
        "plots_path" : plots_path, 
        "cv_ntuple_path" : cv_ntuple_path, 
        "full_ntuple_path" : full_ntuple_path, 
        "dirt_tune" : dirt_tune, 
        "ext_tune" : ext_tune, 
        "beamon_pot" : beamon_pot, 
        "NUE" : NUE, 
        "integrated_flux_per_pot" : integrated_flux_per_pot, 
        "n_target" : n_target,
        "bdt_model" : bdt_model, 
        "bdt_score_cut" : bdt_score_cut, 
        "detsys_flat" : detsys
    }
    
    return d


######################### plot categories ##############################
# everything must pass software trigger ! 

in_fv_query = "10<=true_nu_vtx_x<=246 and -106<=true_nu_vtx_y<=106 and 10<=true_nu_vtx_z<=1026"
out_fv_query = "((true_nu_vtx_x<10 or true_nu_vtx_x>246) or (true_nu_vtx_y<-106 or true_nu_vtx_y>106) or (true_nu_vtx_z<10 or true_nu_vtx_z>1026))"

numu_CC_Npi0 = 'swtrig_pre==1 and ((nu_pdg==14 or nu_pdg==-14) and ccnc==0 and npi0>=1)'
numu_CC_0pi0 = 'swtrig_pre==1 and ((nu_pdg==14 or nu_pdg==-14) and ccnc==0 and npi0==0)'

numu_NC_Npi0 = 'swtrig_pre==1 and ((nu_pdg==14 or nu_pdg==-14) and ccnc==1 and npi0>=1)'
numu_NC_0pi0 = 'swtrig_pre==1 and ((nu_pdg==14 or nu_pdg==-14) and ccnc==1 and npi0==0)'

nuebar_1eNp = 'swtrig_pre==1 and ((nu_pdg==-12 and ccnc==0 and nproton>0 and npion==0 and npi0==0))'
nue_NC = 'swtrig_pre==1 and ((nu_pdg==12 or nu_pdg==-12) and ccnc==1)'

nue_CCother = 'swtrig_pre==1 and (((nu_pdg==12 and ccnc==0) and (nproton==0 or npi0>0 or npion>0)) or (nu_pdg==-12 and ccnc==0 and (nproton==0 or npion>0 or npi0>0)))'

# less specific categories 
nue_other = 'swtrig_pre==1 and (((nu_pdg==12 or nu_pdg==-12) and ccnc==1) or (( (nu_pdg==12 or nu_pdg==-12) and ccnc==0) and (nproton==0 or npi0>0 or npion>0)))'
numu_Npi0 = 'swtrig_pre==1 and ( (nu_pdg==14 or nu_pdg==-14) and npi0>=1)'
numu_0pi0 = 'swtrig_pre==1 and ( (nu_pdg==14 or nu_pdg==-14) and npi0==0)'

# signal vs. not signal 
signal = in_fv_query+' and  swtrig_pre==1 and (nu_pdg==12 and ccnc==0 and nproton>0 and npion==0 and npi0==0)'
not_signal = "(swtrig_pre==0) or (swtrig_pre==1 and (" + out_fv_query+' or (nu_pdg!=12) or (nu_pdg==12 and ccnc==1) or (nu_pdg==12 and ccnc==0 and (nproton==0 or npi0>0 or npion>0))))'

# for replacing nue CC 
in_AV_query = "-1.55<=true_nu_vtx_x<=254.8 and -116.5<=true_nu_vtx_y<=116.5 and 0<=true_nu_vtx_z<=1036.8"
nueCC_query = 'abs(nu_pdg)==12 and ccnc==0 and '+in_AV_query

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
    'ext' : ['EXT', 'lightpink'],
    'nue_other' : ['$\\nu_e$ / $\\overline{\\nu}_e$  other', '#33db09'], 
    'numu_Npi0' : ['$\\nu_\\mu$ / $\\overline{\\nu}_\\mu$  $\pi^{0}$', '#EE1B1B'], 
    'numu_0pi0' : ['$\\nu_\\mu$ / $\\overline{\\nu}_\\mu$  other', '#437ED8'],
    'nuebar_1eNp' : ['$\\bar{\\nu}_e$ CC0$\pi$Np', 'gold']
}

########################################################################
# get rid of 30 MeV threshold on visible energy 
def vis_e_fix(df): 
    
    elec_e = np.array(df.elec_e)
    elec_ke = [0 for i in range(len(df))]
    
    # if electron energy is filled
    for i in range(len(elec_e)): 
        if elec_e[i] > 0: 
            elec_ke[i] = elec_e - 0.000511
            
    df['elec_ke'] = elec_ke
    
    print('added electron kinetic energy')
    
    E_vis = np.array(df.true_e_visible)
    E_vis_new = [0 for i in range(len(E_vis))]

    for i in range(len(E_vis)): 
    
        # for electrons above the 30 MeV threshold - do nothing 
        if elec_ke[i] > 0.03: 
            E_vis_new[i] = E_vis[i]

        # for electrons below the 30 MeV threshold - add to the visible energy 
        elif 0<elec_ke[i]<=0.03: 
            E_vis_new[i] = E_vis[i] + elec_ke[i] 
            
    df['true_e_visible2'] = E_vis_new
    
    print('added new visible energy')
    
    return df

########################################################################
# function to properly scale the RHC Run 3 (before & after software trigger change)
def pot_scale(df, df_type, ISRUN3, tune=True): 
    
    if tune: 
        print('Adding pot_scale column using dirt & EXT tune....')

        dirt_tune = parameters(ISRUN3)['dirt_tune']
        ext_tune = parameters(ISRUN3)['ext_tune']
    
    else: 
        print('Adding pot_scale column without dirt & EXT tune....')

        dirt_tune = 1
        ext_tune = 1     
    
    if ISRUN3: 

        df_before = df.query('run<16880').copy()
        df_after = df.query('run>=16880').copy()

        if df_type == 'overlay': 
            df_before['pot_scale'] = (4.108e+20/1.53689e+21)
            df_after['pot_scale'] = (9.055e+19/4.52483e+20)

        elif df_type == 'intrinsic': 
            df_before['pot_scale'] = (4.108e+20/1.40784e+22)
            df_after['pot_scale'] = (9.055e+19/1.12667e+22)

        elif df_type == 'dirt': 
            df_before['pot_scale'] = (4.108e+20/6.01415e+20)*dirt_tune
            df_after['pot_scale'] = (9.055e+19/4.30847e+20)*dirt_tune

        elif df_type == "ext": 
            df_before['pot_scale'] = (8526417.0/18605756.575)*ext_tune
            df_after['pot_scale'] = (1846526.0/14299750.15)*ext_tune

        else: 
            print(" No scaling for this df type! ")
        
        df_new = pd.concat([df_before, df_after], ignore_index=True, sort=True) 
    
    
    else: 
        
        overlay_pot =  2.33652E21  
        dirt_pot = 1.67392E21 # david's file
        beamon_pot = 2.0E20 #v5

        beamon_ntrig =  5268051.0 # v5 (EA9CNT_wcut)
        beamoff_ntrig = 9199232.74  # v5 (EXT_NUMIwin_FEMBeamTriggerAlgo)

        nue_intrinsic_pot = 2.37838E22
        
        df_new = df.copy()
        
        if df_type == 'overlay': 
            df_new['pot_scale'] = beamon_pot/overlay_pot

        elif df_type == 'intrinsic': 
            df_new['pot_scale'] = beamon_pot/nue_intrinsic_pot

        elif df_type == 'dirt': 
            df_new['pot_scale'] = (beamon_pot/dirt_pot)*dirt_tune

        elif df_type == "ext": 
            df_new['pot_scale'] = (beamon_ntrig/beamoff_ntrig)*ext_tune
        

    return df_new

########################################################################
# MC Stat Error Counting -- sum of the weights 
def mc_error(var, bins, xlow, xhigh, datasets): 
    
    # combine the datasets -- cuts should have already been applied 
    selected = pd.concat(datasets, ignore_index=True, sort=True)
    
    mc_stat = []
    
    for i in range(len(bins)-1):

        if i==len(bins)-2: # if the last bin, 
            bin_query = var+' >= '+str(bins[i])+' and '+var+' <= '+str(bins[i+1])
        
        else: 
            bin_query = var+' >= '+str(bins[i])+' and '+var+' < '+str(bins[i+1])

        mc_stat.append( np.sqrt(sum(selected.query(bin_query).totweight_data ** 2)) )
    
    return mc_stat
########################################################################
# Print out counts for each type of neutrino background inside the FV
def check_counts(in_fv, norm, cuts): 
    
    # (outdated)
    
    #################
    # in_fv --> in FV dataframe 
    # norm --> totweight_data
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
    "LYRayleigh": 7.59732E20, #7.60573E20, 
    "LYDown": 7.43109E20, 
    "SCE": 7.39875E20, 
    "Recomb2": 7.59105E20, 
    "WireModX": 7.64918E20, 
    "WireModYZ": 7.532E20, 
    "WireModThetaXZ": 7.64282E20,
    "WireModThetaYZ_withSigmaSplines": 7.64543E20, 
    "CV": 7.59732E20
}

intrinsic_detvar_run1_fhc = {
    "LYRayleigh_intrinsic": 2.67655E22, #2.38081E22, 
    "LYDown_intrinsic": 2.24505E22, 
    "SCE_intrinsic": 2.60685E22, #2.39023E22, 
    "Recomb2_intrinsic":  2.60657E22, #2.38193E22, 
    "WireModX_intrinsic": 2.66184E22, #2.38318E22, 
    "WireModYZ_intrinsic":  2.62256E22, #2.38416E22,
    "WireModThetaXZ_intrinsic": 2.65175E22, #2.31518E22, 
    "WireModThetaYZ_withSigmaSplines_intrinsic": 2.62256E22, #2.31421E22, 
    "CV_intrinsic": 2.68294E22 #2.37261E22   
}

detvar_run3_rhc = {
    "LYAttenuation": 3.31177E20,
    "LYRayleigh":  3.15492E20, # 2.81E20, 
    "LYDown": 3.2338E20, #2.81E20, 
    "SCE": 3.33283E20, 
    "Recomb2": 3.29539E20, 
    "WireModX": 3.24286E20, 
    "WireModYZ": 3.36399E20, 
    "WireModThetaXZ": 3.20027E20,
    "WireModThetaYZ_withSigmaSplines": 3.35762E20, 
    "CV": 2.87219E20 #2.72E20
    
}

intrinsic_detvar_run3_rhc = {
    "LYAttenuation_intrinsic": 2.5392E22,
    "LYRayleigh_intrinsic": 2.53581E22, 
    "LYDown_intrinsic": 2.53082E22, 
    "SCE_intrinsic": 2.54153E22,  
    "Recomb2_intrinsic": 2.54549E22,  
    "WireModX_intrinsic": 2.50092E22, 
    "WireModYZ_intrinsic": 2.54089E22, 
    "WireModThetaXZ_intrinsic": 2.44365E22, 
    "WireModThetaYZ_withSigmaSplines_intrinsic":2.5992E22, 
    "CV_intrinsic": 2.5392E22
    
}
########################################################################
# corrected visible energy variable - account for electrons below 30 MeV 
def visible_energy_nothres(df): 
    
    df['elec_ke'] = df.elec_e - 0.000511
    elec_ke = list(df['elec_ke'])

    E_vis = np.array(df.true_e_visible)
    E_vis_new = [0 for i in range(len(E_vis))]

    for i in range(len(E_vis)): 
    
        # for electrons above the 30 MeV threshold - do nothing 
        if elec_ke[i] > 0.03 or elec_ke[i] < 0: 
            E_vis_new[i] = E_vis[i]

        # for electrons below the 30 MeV threshold - add to the total visible energy 
        elif 0<elec_ke[i]<=0.03: 
            E_vis_new[i] = E_vis[i] + elec_ke[i] 

    df['true_e_visible2'] = E_vis_new
########################################################################
# scales to standard overlay 
def generated_signal(ISRUN3, var, bins, xlow, xhigh, cuts=None, weight='totweight_data', genie_sys=None):
    
    # print('WARNING: generated_signal now scales to beam on POT unless otherwise specified! --> make sure to update functions using this!')
        
    fold = "nuselection"
    tree = "NeutrinoSelectionFilter"
    
    variables = ["swtrig_pre", 'run', "nu_pdg", "ccnc", "nproton", "npion", "npi0", 
                "true_nu_vtx_x", "true_nu_vtx_y", "true_nu_vtx_z", "ppfx_cv", "weightSplineTimesTune", "weightTune",
                "nslice", 
                 "elec_e", "shr_energy_cali", 
                 "NeutrinoEnergy2", "true_e_visible", 
                 "opening_angle", "tksh_angle"] 
    
    
    if var not in variables: 
        if var is not "true_e_visible2": 
            if var is not "NeutrinoEnergy2_GeV": 
                variables.append(var)
    
    if genie_sys: 
        if isinstance(genie_sys, list): 
            variables = variables + genie_sys
            
        else: 
            variables.append(genie_sys)
        
    f = uproot.open(parameters(ISRUN3)['full_ntuple_path']+parameters(ISRUN3)['NUE']+".root")[fold][tree]
    df = f.pandas.df(variables, flatten=False)
    
    df.loc[ df['weightSplineTimesTune'] <= 0, 'weightSplineTimesTune' ] = 1.
    df.loc[ df['weightSplineTimesTune'] == np.inf, 'weightSplineTimesTune' ] = 1.
    df.loc[ df['weightSplineTimesTune'] > 100, 'weightSplineTimesTune' ] = 1.
    df.loc[ np.isnan(df['weightSplineTimesTune']) == True, 'weightSplineTimesTune' ] = 1.
    
    df.loc[ df['weightTune'] <= 0, 'weightTune' ] = 1.
    df.loc[ df['weightTune'] == np.inf, 'weightTune' ] = 1.
    df.loc[ df['weightTune'] > 100, 'weightTune' ] = 1.
    df.loc[ np.isnan(df['weightTune']) == True, 'weightTune' ] = 1.
    
    df['is_signal'] = np.where((df.swtrig_pre == 1)
                             & (df.nu_pdg==12) & (df.ccnc==0) & (df.nproton>0) & (df.npion==0) & (df.npi0==0)
                             & (10 <= df.true_nu_vtx_x) & (df.true_nu_vtx_x <= 246)
                             & (-106 <= df.true_nu_vtx_y) & (df.true_nu_vtx_y <= 106)
                             & (10 <= df.true_nu_vtx_z) & (df.true_nu_vtx_z <= 1026), 
                               True, False)
    
    df['NeutrinoEnergy2_GeV'] = df['NeutrinoEnergy2']/1000
    visible_energy_nothres(df)
    
    
    df_signal = df.query('is_signal==True').copy()
    df_signal = pot_scale(df_signal, 'intrinsic', ISRUN3)
    
    #print('Tune weight is off!')
    #df_signal['weightSplineTimesTune'] = [1 for x in range(len(df_signal))]
    #df_signal['weightTune'] = [1 for x in range(len(df_signal))]

    df_signal['totweight_data'] = df_signal['ppfx_cv']*df_signal['pot_scale']*df_signal['weightSplineTimesTune']
    df_signal['totweight_intrinsic'] = df_signal['ppfx_cv']*df_signal['weightSplineTimesTune']
    
    
    if genie_sys=='weightsGenie': 
        df_signal[genie_sys] = df_signal[genie_sys]/1000
        
        for ievt in range(df_signal.shape[0]):
            # check for NaNs separately        
            if np.isnan(df_signal['weightsGenie'].iloc[ievt]).any() == True: 
                df_signal['weightsGenie'].iloc[ievt][ np.isnan(df_signal['weightsGenie'].iloc[ievt]) ] = 1.

            reweightCondition = ((df_signal['weightsGenie'].iloc[ievt] > 60) | (df_signal['weightsGenie'].iloc[ievt] < 0)  | 
                                 (df_signal['weightsGenie'].iloc[ievt] == np.inf) | (df_signal['weightsGenie'].iloc[ievt] == np.nan))
            df_signal['weightsGenie'].iloc[ievt][ reweightCondition ] = 1.

            # if no variations exist for the event
            if not list(df_signal['weightsGenie'].iloc[ievt]): 
                df_signal['weightsGenie'].iloc[ievt] = [1.0 for k in range(600)]
                


    if cuts: 
        df_signal = df_signal.query(cuts)
        
    # make a histogram of the generated SIGNAL ONLY events (CV)
    n, b, p = plt.hist(df_signal[var], bins, histtype='bar', range=[xlow, xhigh], weights=df_signal[weight])
    plt.close()
    
    if genie_sys: 
        if isinstance(genie_sys, list): 
           df_weights = df_signal[genie_sys+['weightTune', 'totweight_data', var]].copy() 
            
        else: 
            df_weights = df_signal[[genie_sys, 'weightTune', 'totweight_data', var]].copy()
        
    else: 
        df_weights = None
       
    generated_sumw2 = []
    if weight=='totweight_data': 
        
        for i in range(len(bins)-1): 
            
            if i==len(bins)-2: 
                bin_query = var+'>='+str(bins[i])+' and '+var+'<='+str(bins[i+1])
            else: 
                bin_query = var+'>='+str(bins[i])+' and '+var+'<'+str(bins[i+1])

            generated_sumw2.append( sum(df_signal.query(bin_query).totweight_data ** 2) )


    
    return n.tolist(), df_weights, generated_sumw2
    
########################################################################
# parameters for the xsec variables 
# Outdated
def xsec_variables(xvar, ISRUN3): 
    
    print("Need to update before using these! ")
    
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


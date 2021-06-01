'''''
Detector Systematics

'''''

import sys
import selection_functions as sf

import importlib

import uproot
import matplotlib.pylab as pylab
import numpy as np
import math
from sklearn.model_selection import train_test_split
import pickle
import xgboost as xgb


import awkward
import matplotlib.pyplot as plt
import pandas as pd

import ROOT
from ROOT import TH1F, TDirectory

import os




##############################
###### file information ######

fold = "nuselection"
tree = "NeutrinoSelectionFilter"

path = '/uboone/data/users/kmiller/systematics/detvar/run1/slimmed_loosecuts/'
plots_path = '/uboone/data/users/kmiller/searchingfornues_v33/v08_00_00_33/plots/fhc/'

variables = [
    "selected", "nu_pdg", "shr_theta", "true_e_visible", 
    "trk_score_v", 
    "shr_tkfit_dedx_Y", "ccnc", "n_tracks_contained", 
    "NeutrinoEnergy2",
    "reco_nu_vtx_sce_x","reco_nu_vtx_sce_y","reco_nu_vtx_sce_z",
    "shrsubclusters0","shrsubclusters1","shrsubclusters2", # number of sub-clusters in shower
    "trkshrhitdist2",
    "nproton", "nu_e", "n_showers_contained", "nu_purity_from_pfp", 
    "shr_phi", "trk_phi", "trk_theta",
    "shr_score", 
    "trk_energy", "tksh_distance", "tksh_angle",
    "npi0",
    "shr_energy_tot_cali",  
    "nslice", 
    "contained_fraction",
    "true_nu_vtx_x", "true_nu_vtx_y" , "true_nu_vtx_z", 
    "npion", "shr_energy_cali", 
    "shrmoliereavg", "shr_px", "shr_py", "shr_pz",
    "true_nu_px", "true_nu_py", "true_nu_pz", 
    "elec_e", "proton_e", "mc_px", "mc_py", "mc_pz", "elec_px", "elec_py", "elec_pz", 
    "swtrig_pre", "ppfx_cv", "mc_pdg", "trkpid", "subcluster", "weightSplineTimesTune"
    
]

##############################
#### nue intrinsic stuff #####

# intrinsic sample contains in AV TPC events ONLY, & only CC events (overlay is entire cryo)
in_AV_query = "-1.55<=true_nu_vtx_x<=254.8 and -116.5<=true_nu_vtx_y<=116.5 and 0<=true_nu_vtx_z<=1036.8"
nueCC_query = 'abs(nu_pdg)==12 and ccnc==0 and '+in_AV_query

##############################
###### input parameters ######

xsec_variables = ["NeutrinoEnergy2_GeV", "nu_e", "shr_energy_cali", "n_tracks_contained", "tksh_angle"]
bins = [500, 500, 500, 12, 200]
xlow = [0, 0, 0, 0, -1]
xhigh = [5, 5, 5, 12, 1]

# create dictionary with POT values
variations = {
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

intrinsic_variations = {
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


class NuMIDetSys: 
     
    ##########################################################################################
    # Create BDT-selected event rate histograms for all variations & store to ROOT file 
    def makehist_detsys(self, variation, query, intrinsic=False): 
        
        if not intrinsic: 
            d = variations
        else: 
            d = intrinsic_variations
    
        fout = ROOT.TFile.Open("/uboone/data/users/kmiller/systematics/NuMI_FHC_BDTSelected_Detector_Variations.root","UPDATE")
        
        f = uproot.open(path+"neutrinoselection_filt_run1_overlay_"+variation+".root")[fold][tree]
        print("Opening neutrinoselection_filt_run1_overlay_"+variation+".root:")
    
        df = f.pandas.df(variables, flatten=False)
        df['NeutrinoEnergy2_GeV'] = df['NeutrinoEnergy2']/1000
    
        df.loc[ df['weightSplineTimesTune'] <= 0, 'weightSplineTimesTune' ] = 1.
        df.loc[ df['weightSplineTimesTune'] == np.inf, 'weightSplineTimesTune' ] = 1.
        df.loc[ df['weightSplineTimesTune'] > 100, 'weightSplineTimesTune' ] = 1.
        df.loc[ np.isnan(df['weightSplineTimesTune']) == True, 'weightSplineTimesTune' ] = 1.
        
        # add POT normalization to data - 2.0 x 10^20 POT (update for RHC later)
        beamon_pot = 2.0E20
        df['pot_scale'] = beamon_pot/d.get(variation)
        print("POT scale to data = " + str(beamon_pot/d.get(variation)))
        
        # remove nue CC events - only non-nueCC backgrounds  
        if intrinsic==False: 
            print("# of nueCC in AV in overlay det. sys. sample = "+str(len(df.query(nueCC_query))))
            len1 = len(df)
    
            idx = df.query(nueCC_query).index
            df.drop(idx, inplace=True)
            len2 = len(df) 
    
            print("# of nueCC in AV removed in overlay det. sys. sample = "+str(len1-len2)) # should be same as above
    
        # load BDT model 
        bdt_model = xgb.Booster({'nthread': 4})
        bdt_model.load_model('bdt_model_feb2021.model')

        varlist = ["shr_score", "shrmoliereavg", "trkpid",
            "n_showers_contained", "shr_tkfit_dedx_Y", "tksh_distance",
            "tksh_angle", "subcluster", "trkshrhitdist2"]
 
        df_bdt = df.copy()

    
        # clean datasets & apply BDT model  
        for column in varlist:
            df_bdt.loc[(df_bdt[column] < -1.0e37) | (df_bdt[column] > 1.0e37), column] = np.nan
    
        df_test = xgb.DMatrix(data=df_bdt[varlist])
        preds = bdt_model.predict(df_test)
        df_bdt['BDT_score'] = preds

    
        # create TH1F plots 
        for i in range(len(xsec_variables)): 
        
            print("Creating "+xsec_variables[i]+" ("+variation+").")
        
            h = TH1F(variation, xsec_variables[i]+" ("+variation+")", bins[i], xlow[i], xhigh[i])
            w = list(df_bdt.query(query)['ppfx_cv']*df_bdt.query(query)['weightSplineTimesTune']*df_bdt.query(query)['pot_scale'])
    
            for j in range(len(df_bdt.query(query))): 
                h.Fill(list(df_bdt.query(query)[xsec_variables[i]])[j], w[j])
        
            h.SetDirectory(0)
    
            # save to file
            fout.cd()
        
            if fout.GetDirectory(xsec_variables[i]): 
                fout.cd(xsec_variables[i])
            else: 
                fout.mkdir(xsec_variables[i])
                fout.cd(xsec_variables[i])
        
            h.Write()
            h.Reset()
    
        fout.Close()
    
    ##########################################################################################
    # Pull varied event rates & plot against CV (detector systematic samples)
    def plot_variations(self, x, bin_edges, axis_label=None, save=False, pot=None, intrinsic=False): 

        
        f = uproot.open("/uboone/data/users/kmiller/systematics/NuMI_FHC_BDTSelected_Detector_Variations.root")[x]
        
        if not intrinsic: 
            d = variations
        else: 
            d = intrinsic_variations
        
        fig = plt.figure(figsize=(8, 5))
        
        for v in list(d.keys()): 
            
            # grab histogram 
            h = f[v]
            b = [round(x, 2) for x in h.edges] # old bin edges
            counts = list(h.values)
            
            # store counts for new binning 
            y = []
            
            # now re-bin wider 
            for i in range(1, len(bin_edges)): 
                
                start = b.index(bin_edges[i-1])
                stop = b.index(bin_edges[i])
                
                y.append(sum(counts[start:stop]))
                
            # plot 
            if "CV" in v:
                # print("Plotting CV")
                plt.step(bin_edges+[bin_edges[-1]], [0]+y+[0], linewidth=2, color="black")
                
            else: 
               # print("Plotting "+str(v))
                plt.step(bin_edges+[bin_edges[-1]], [0]+y+[0], linewidth=0.5, color="cornflowerblue")

        if pot: 
            plt.ylabel("$\\nu$ / "+pot, fontsize=15)
        
        if axis_label: 
            plt.xlabel(axis_label, fontsize=14)
        else: 
            plt.xlabel(x, fontsize=14)
        
        plt.ylim(bottom=0)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        
        if not intrinsic: 
            plt.title("Detector Variations (non-$\\nu_e$ CC events)", fontsize=14)
        else: 
            plt.title("Detector Variations ($\\nu_e$ CC events)", fontsize=14)
        
        if save and intrinsic==True: 
            plt.savefig(plots_path+x+"_DetSys_Intrinsic.pdf", transparent=True, bbox_inches='tight') 
            print('saving to: '+plots_path)
            
        elif save and intrinsic==False: 
            plt.savefig(plots_path+x+"_DetSys.pdf", transparent=True, bbox_inches='tight') 
            print('saving to: '+plots_path)
        
        plt.show()
            
    ##########################################################################################
    # return a list of lists, of ratio-to-CV, for each bin, in the order of the variation list 
    def ratio_to_CV(self, x, bin_edges, intrinsic=False):
        
        if not intrinsic: 
            d = variations
        else: 
            d = intrinsic_variations
        
        # list of lists for ratios to CV 
        v_counts = []
        weights = []
 
        f = uproot.open("/uboone/data/users/kmiller/systematics/NuMI_FHC_BDTSelected_Detector_Variations.root")[x]
        
        for v in list(d.keys()): 
            
            # counts in each widened bin 
            y = []
            
            # grab histogram 
            h = f[v]
            b = [round(x, 2) for x in h.edges] # old bin edges
            counts = list(h.values)
            
            # store counts for new binning 
            for i in range(1, len(bin_edges)): 
                
                start = b.index(bin_edges[i-1])
                stop = b.index(bin_edges[i])
                y.append(sum(counts[start:stop]))
                
            if "CV" not in v: 
                v_counts.append(y)
                
            elif "CV" in v:
                # divide through by central value event rate 
                for j in range(len(v_counts)): 
                    weights.append([a/b for a, b in zip(v_counts[j], y)]) # where y is now the CV event rate
                ncv = y
                
        return v_counts, ncv, weights
        
    
    
    
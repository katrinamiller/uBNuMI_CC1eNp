'''''
Detector Systematics

'''''

import sys
sys.path.insert(0, '../BDT_models')

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
from ROOT import TH1F, TDirectory, TH1D

import os


import top 
importlib.reload(top)
from top import *


##############################
###### file information ######


fold = "nuselection"
tree = "NeutrinoSelectionFilter"

variables = [
    "nu_pdg", "ccnc",
    "true_nu_vtx_x", "true_nu_vtx_y" , "true_nu_vtx_z", 
    "swtrig_pre", "npion", "nproton", "npi0",
    "reco_nu_vtx_sce_x","reco_nu_vtx_sce_y","reco_nu_vtx_sce_z",
    "n_tracks_contained", 
    "shr_tkfit_dedx_Y",  
    "shrsubclusters0","shrsubclusters1","shrsubclusters2", # number of sub-clusters in shower
    "trkshrhitdist2",
    "n_showers_contained",
    "shr_score", "tksh_angle", 
    "trk_energy", 
    "tksh_distance",
    "shr_energy_tot_cali", 
    "shr_energy_cali", 
    "nslice", 
    "contained_fraction", 
    "shrmoliereavg", 
    "ppfx_cv", "weightSplineTimesTune", "NeutrinoEnergy2"
]


class NuMIDetSys: 
     
    ##########################################################################################
    # Create selected event rate histograms for all variations & store to ROOT file 
    # POT scaling is to the beam on POT
    def makehist_detsys(self, variation, isrun3, output_file, xvar, bins, cut=None, useBDT=False, 
                       background_subtraction=False): 
        
        print("Make sure to update for Run 2 detsys, and to store background subtraction!")

        if not isrun3: 
            run = 'run1'
            standard_dict = detvar_run1_fhc
            intrinsic_dict = intrinsic_detvar_run1_fhc
            
        else:
            run = 'run3b'
            standard_dict = detvar_run3_rhc
            intrinsic_dict = intrinsic_detvar_run3_rhc
           
            
        input_path = "/uboone/data/users/kmiller/uBNuMI_CCNp/ntuples/"+run+"/systematics/detvar/" #+ "slimmed/qualcuts/"
    
        beamon_pot = parameters(isrun3)['beamon_pot']
       
        print("Opening output file: "+ output_file) 
        fout = ROOT.TFile.Open(input_path+'makehist_detsys_output/'+output_file, "UPDATE")
        
        # grab detector variaton files
        standard_input_file = input_path + "standard_overlay/neutrinoselection_filt_"+run+"_overlay_"+variation+".root"
        intrinsic_input_file = input_path + "intrinsic/neutrinoselection_filt_"+run+"_overlay_"+variation+"_intrinsic.root"
        
        f_standard = uproot.open(standard_input_file)[fold][tree]
        f_intrinsic = uproot.open(intrinsic_input_file)[fold][tree]
    
        uproot_v = [f_standard, f_intrinsic]
        
        print("Opening files for "+variation+ " using xvar: " + xvar)
        
        # create dataframe 
        df_standard = f_standard.pandas.df(variables, flatten=False)
        df_intrinsic = f_intrinsic.pandas.df(variables, flatten=False)
        
        for i, df in enumerate([df_standard, df_intrinsic]):
            
            up = uproot_v[i]
            trk_llr_pid_v = up.array('trk_llr_pid_score_v')
            trk_id = up.array('trk_id')-1 
            trk_llr_pid_v_sel = awkward.fromiter([pidv[tid] if tid<len(pidv) else 9999. for pidv,tid in zip(trk_llr_pid_v,trk_id)])
            df['trkpid'] = trk_llr_pid_v_sel
            df['subcluster'] = df['shrsubclusters0'] + df['shrsubclusters1'] + df['shrsubclusters2']
        
            df['NeutrinoEnergy2_GeV'] = df['NeutrinoEnergy2']/1000

            df.loc[ df['weightSplineTimesTune'] <= 0, 'weightSplineTimesTune' ] = 1.
            df.loc[ df['weightSplineTimesTune'] == np.inf, 'weightSplineTimesTune' ] = 1.
            df.loc[ df['weightSplineTimesTune'] > 100, 'weightSplineTimesTune' ] = 1.
            df.loc[ np.isnan(df['weightSplineTimesTune']) == True, 'weightSplineTimesTune' ] = 1.
            
            # bool for is signal vs is not signal 
            df['is_signal'] = np.where((df.swtrig_pre == 1) 
                             & (df.nu_pdg==12) & (df.ccnc==0) & (df.nproton>0) & (df.npion==0) & (df.npi0==0)
                             & (10 <= df.true_nu_vtx_x) & (df.true_nu_vtx_x <= 246)
                             & (-106 <= df.true_nu_vtx_y) & (df.true_nu_vtx_y <= 106)
                             & (10 <= df.true_nu_vtx_z) & (df.true_nu_vtx_z <= 1026), True, False)
            
            print('is_signal check:', len(df) == len(df.query('is_signal==True')) + len(df.query('is_signal==False')))
        
        # software trigger 
        df_standard = df_standard.query('swtrig_pre==1')
        df_intrinsic = df_intrinsic.query('swtrig_pre==1')
        
        # scale to beam on POT
        df_standard['pot_scale'] = beamon_pot/standard_dict.get(variation)    
        df_intrinsic['pot_scale'] = beamon_pot/intrinsic_dict.get(variation+'_intrinsic')  
    
        # remove nue CC events 
        print("# nueCC in AV in standard overlay det. sys. sample = "+str(len(df_standard.query(nueCC_query))))
        len1 = len(df_standard)
    
        idx = df_standard.query(nueCC_query).index
        df_standard.drop(idx, inplace=True)
        len2 = len(df_standard) 
    
        print("# of nueCC in AV removed = "+str(len1-len2)) # should be same as above
        
        overlay = pd.concat([df_standard,df_intrinsic], ignore_index=True)
    
        # load BDT model 
        if useBDT: 
            
            print('Using BDT')
            
            # load bdt model 
            bdt_model = xgb.Booster({'nthread': 4})
            bdt_model.load_model(parameters(isrun3)['bdt_model'])

            df_bdt = overlay.copy()

            # clean datasets 
            for column in training_parameters:
                df_bdt.loc[(df_bdt[column] < -1.0e37) | (df_bdt[column] > 1.0e37), column] = np.nan
    
            # apply BDT model  
            df_test = xgb.DMatrix(data=df_bdt[training_parameters])
            preds = bdt_model.predict(df_test)
            df_bdt['BDT_score'] = preds
            
            df_sel = df_bdt
    
        else: 
            df_sel = overlay
            
        if cut: 
            df_sel = df_sel.query(cut)
         
        # with GENIE & PPFX weights - full selected event rate
        h = TH1F(variation, xvar+" ("+variation+" - Full Selected Event Rate)", len(bins)-1, np.array(bins))
        w = list(df_sel['ppfx_cv']*df_sel['weightSplineTimesTune']*df_sel['pot_scale']) # scales to standard overlay POT
        
        # create python histogram
        print("Creating python histogram....")  
        counts = plt.hist(df_sel[xvar], bins=bins, range=[bins[0], bins[-1]], weights=w)[0]
        plt.close()
        

        # store to ROOT histo
        bincenters = 0.5*(np.array(bins)[1:]+np.array(bins)[:-1])
 
        print("Filling ROOT histogram....")  
        for j in range(len(counts)): 
                #h.Fill(bincenters[j], counts[j])
                h.SetBinContent(j+1, counts[j])
                print(h.GetBinContent(j+1))

        if background_subtraction: 
            if variation=="CV": # store CV background for subtraction later on 
            
                print('Storing CV Background for Subtraction ....')
                print('is_signal check:', len(df_sel) == len(df_sel.query('is_signal==True')) + len(df_sel.query('is_signal==False')))

                df_bkgd_sel = df_sel.query('is_signal==False')

                h_cv_bkgd = TH1D(variation+'_Bkgd', xvar+" ("+variation+" - Selected Background)", len(bins)-1, np.array(bins))
                w_cv_bkgd = list(df_bkgd_sel['ppfx_cv']*df_bkgd_sel['weightSplineTimesTune']*df_bkgd_sel['pot_scale']) 

                bkgd_counts = plt.hist(df_bkgd_sel[xvar], bins=bins, range=[bins[0], bins[-1]], weights=w_cv_bkgd)[0]
                plt.close()

                for j in range(len(bkgd_counts)): 
                    h_cv_bkgd.SetBinContent(j+1, bkgd_counts[j])
                    print(h_cv_bkgd.GetBinContent(j+1))
            
        
        h.SetDirectory(0)
    
        # save to file
        fout.cd()
        
        if fout.GetDirectory(xvar): 
            fout.cd(xvar)
        else: 
            fout.mkdir(xvar)
            fout.cd(xvar)
        
        h.Write()
        h.Reset()
        
        if background_subtraction and variation=='CV': 
            h_cv_bkgd.Write()
            h_cv_bkgd.Reset()
          

        fout.Close()
    
    
    ##########################################################################################
    # Pull varied event rates & plot against CV (for the detector systematic samples)
    # return variation counts for covariance matrix calculation 
    def plot_variations(self, var, bins, file, ISRUN3, axis_label=None, save=False, plot=False, background_subtraction=False): 
        
        if ISRUN3: 
            run = 'run3b'
            d = detvar_run3_rhc
        else: 
            run = 'run1'
            d = detvar_run1_fhc

            
        path = "/uboone/data/users/kmiller/uBNuMI_CCNp/ntuples/"+run+"/systematics/detvar/" #+ "slimmed/qualcuts/"
        
        # always scale to data pot
        #pot_scaling = parameters(ISRUN3)['beamon_pot'] / d['CV']
        
        f = uproot.open(path+'makehist_detsys_output/'+file)[var]
        
        # dictionary for variation counts 
        variation_counts = {}
        
        # store CV 
        hcv = list(f['CV'].values)
        
        if background_subtraction: 
            hcv_bkgd = f['CV_Bkgd']
            cv_bkgd = list(hcv_bkgd.values)
            print('hcv', hcv)
            hcv = [y-z for y,z in zip(hcv, cv_bkgd)] # background subtracted 
            print('cv background', cv_bkgd)
            print('hcv', hcv)
            
            
        variation_counts['CV'] = hcv
        
        # bin width 
        x_err = [ round(abs(bins[x+1]-bins[x])/2, 3) for x in range(len(bins)-1) ]
        bin_centers = []
        for a in range(len(bins)-1): 
            bin_centers.append(round(bins[a] + (bins[a+1]-bins[a])/2, 3))
        

        # loop over the variations 
        for v in list(d.keys()):
            
            if "CV" in v: 
                continue
    
            # grab histograms
            h = f[v]
            
            if background_subtraction: 
                variation_counts[v] = [ y-z for y,z in zip(list(h.values), cv_bkgd) ]
            else: 
                variation_counts[v] = list(h.values)
            
            # h_uw = f[v+"_UW"]
            
            #b = [round(var, 2) for var in h.edges] # old bin edges
            # print(b)
            # counts = list(h.values)
            #counts_uw = list(h_uw.values)
            
            # store counts for new binning 
            #y = []
            #y_uw = []
            #stat_err = []
            
            # now re-bin wider 
            #for i in range(1, len(bin_edges)): 
                
            #    start = b.index(bin_edges[i-1])
            #    stop = b.index(bin_edges[i])
                
            #    y.append(sum(counts[start:stop]))
            #    y_uw.append(sum(counts_uw[start:stop]))

            # POT scaling for the weighted counts 
            #y_scaled = [z*pot_scaling for z in y]
            
            #if xsec_units: 
            #    y_scaled = [(1E39) * y/(n_target*flux) for y in y_scaled]
            
        
            if plot: 
                
                fig = plt.figure(figsize=(8, 5)) 
                
                # calculate MC stat error from (unweighted, unscaled) overlay POT - obsolete - use sum of weighted squares
                #frac_stat_err = [np.sqrt(k)/k for k in y_uw]
                #stat_err = [a*b for a,b in zip(frac_stat_err, y_scaled)]
 
                plt.hist(bin_centers, bins, histtype='step', 
                         range=[bins[0], bins[-1]], weights=variation_counts[v], 
                         color='cornflowerblue', linewidth=0.5, label='UV')
                
                plt.errorbar(bin_centers, hcv, 
                             xerr=x_err, #yerr=stat_err, 
                             fmt='none', color='black', linewidth=2, label='CV')

         
                plt.ylabel("$\\nu$ / "+str(parameters(ISRUN3)['beamon_pot'])+" POT ", fontsize=15)
        
                if axis_label: 
                    plt.xlabel(axis_label, fontsize=14)
                else: 
                    plt.xlabel(var, fontsize=14)

                plt.ylim(bottom=0)

                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                plt.legend(fontsize=14, frameon=False)

                if background_subtraction: 
                    plt.title(v + " (Background Subtracted)", fontsize=14)
                else: 
                    plt.title(v, fontsize=14)
        
            #if save : 
            #    plots_path = parameters(ISRUN3)['plots_path']
            #    plt.savefig(plots_path+str(var)+"_DetSys.pdf", transparent=True, bbox_inches='tight') 
            #    print('saving to: '+plots_path)
        
                plt.show()
        
        
        return variation_counts
            
        
    ############################################################################
    ############################################################################
    ############################################################################
    ############################################################################
    ############################################################################
    #
    # for detector systematics study or obsolete stuff 
    #
    ############################################################################
        ##########################################################################################
    # return a list of lists, of ratio-to-CV, for each bin, in the order of the variation list 
    # scales to overlay POT unless knob for data POT is turned on 
    def ratio_to_CV(self, x, bin_edges, file, ISRUN3, intrinsic=False, moreStats=False, data_pot=None, xsec_units=False):
        
        n_target = parameters(ISRUN3)['n_target']
        flux = parameters(ISRUN3)['integrated_flux_per_pot'] * parameters(ISRUN3)['beamon_pot']
        
        if not intrinsic: 
            d = detector_variations
            cv_pot = d['CV']
        else: 
            d = detector_intrinsic_variations
            cv_pot = d['CV_intrinsic']
        
        # list of lists for ratios to CV 
        v_counts = []
        weights = []
        
        f = uproot.open(path+'makehist_detsys_output/old/'+file)[x]
 
        #if moreStats: 
            #f = uproot.open(path+"NuMI_FHC_PreBDT_Detector_Variations.root")[x]
        #else: 
            #f = uproot.open(path+"NuMI_FHC_PostBDT_Detector_Variations.root")[x]
        

        for v in list(d.keys()):

            # counts in each widened bin 
            y = []
            
            # grab histogram 
            h = f[v]
            b = [round(x, 2) for x in h.edges] # old bin edges
            variation_pot = d[v] 
            
            if data_pot: 
                pot_scaling = data_pot / variation_pot
            else: 
                pot_scaling = cv_pot / variation_pot
                
            counts = [m*pot_scaling for m in list(h.values)]
            
            # replace inf or nan counts with 0            
            replace_counter = 0
            
            for k in range(len(counts)): 
                if np.isnan(counts[k]) or np.isinf(counts[k]): 
                    counts[k] = 0.0
                    replace_counter = replace_counter + 1
            
            # print('Replacing '+str(replace_counter)+' inf/nan events with 0.0')
                    
            # store counts for new binning 
            for i in range(1, len(bin_edges)): 
                
                start = b.index(bin_edges[i-1])
                stop = b.index(bin_edges[i])
                y.append( sum(counts[start:stop]) )
                
            #print(v)
            
            
            
            if "CV" not in v: 
                if xsec_units: 
                    y = [k/(n_target*flux) for k in y]
                v_counts.append(y)
                
            elif "CV" in v:
                if xsec_units: 
                    y = [k/(n_target*flux) for k in y]
                
                # divide through by central value event rate 
                for j in range(len(v_counts)): 
                    weights.append([a/b for a, b in zip(v_counts[j], y)]) # where y is now the CV event rate
                
                ncv = y
         
        return v_counts, ncv, weights # weights is the ratio to CV 
    
    #############################################################################
    def makehist_detsys_test(self, variation, sample, cut=None, useBDT=True): 
        
        
        cv_pot = intrinsic_variations.get("CV_intrinsic") # scale to det. sys. CV 
          
        # grab loose-cut-selected detector variations 
        #f = uproot.open(path+"neutrinoselection_filt_run1_overlay_"+variation+".root")[fold][tree]
        #print("Opening neutrinoselection_filt_run1_overlay_"+variation+".root:")
        
        f_intrinsic = uproot.open(path+"neutrinoselection_filt_run1_overlay_"+variation+"_intrinsic.root")[fold][tree]
        print("Opening neutrinoselection_filt_run1_overlay_"+variation+"_intrinsic.root:")
    
    
        # open new file to store det. var. histograms
        fout = ROOT.TFile.Open(path+"NuMI_FHC_PostBDT_Detector_Variations_"+sample+".root",
                               "UPDATE")

        
        # create pandas dataframes & fix weights
        #df_standard = f.pandas.df(variables, flatten=False)
        df_intrinsic = f_intrinsic.pandas.df(variables, flatten=False)
        
        #df_standard['pot_scale'] = cv_pot/variations.get(variation) # scale to standard det sys CV pot 
        #print("POT scale to det. sys. CV = " + str(cv_pot/variations.get(variation)))
        
        df_intrinsic['pot_scale'] = cv_pot/intrinsic_variations.get(variation+"_intrinsic") # scale to standard det sys CV pot 
        print("POT scale to det. sys. CV (intrinsic) = " + str(cv_pot/intrinsic_variations.get(variation+"_intrinsic")))


        ######  BACKGROUND SAMPLE #######
        # remove signal events from intrinsic if doing background only
        #print("# signal in intrinsic overlay det. sys. sample = "+str(len(df_intrinsic.query(signal))))
        #len1 = len(df_intrinsic)
    
        #idx = df_intrinsic.query(signal).index
        #df_intrinsic.drop(idx, inplace=True)
        #len2 = len(df_intrinsic) 
    
        #print("# of signal removed from intrinsic sample = "+str(len1-len2)) # should be same as above
        
        # ALSO remove nueCC events from standard overlay -- those events get covered by intrinsic 
        #print("# signal in standard overlay det. sys. sample = "+str(len(df_standard.query(signal))))
        #len1 = len(df_standard)
    
        #idx = df_standard.query(signal).index
        #df_standard.drop(idx, inplace=True)
        #len2 = len(df_standard) 
    
        #print("# of nueCC removed from standard overlay sample = "+str(len1-len2)) # should be same as above
        
        
        ###### COMBINED SAMPLE ######
        # for combined sample: remove nueCC events from standard overlay sample 

        #print("# nueCC in AV in standard overlay det. sys. sample = "+str(len(df_standard.query(nueCC_query))))
        #len1 = len(df_standard)
    
        #idx = df_standard.query(nueCC_query).index
        #df_standard.drop(idx, inplace=True)
        #len2 = len(df_standard) 
    
        #print("# of nueCC in AV removed = "+str(len1-len2)) # should be same as above
        
        ###### NUE CC BACKGROUNDS #########
        
        print("# signal in intrinsic overlay det. sys. sample = "+str(len(df_intrinsic.query(signal))))
        len1 = len(df_intrinsic)
    
        idx = df_intrinsic.query(signal).index
        df_intrinsic.drop(idx, inplace=True)
        len2 = len(df_intrinsic) 
        
        print("# of signal removed from intrinsic sample = "+str(len1-len2)) # should be same as above
        
        ###################################
        

        # load bdt model 
        bdt_model = xgb.Booster({'nthread': 4})
        bdt_model.load_model(parameters(ISRUN3)['bdt_model'])
       # bdt_model.load_model('bdt_FHC_oct2021_v3.model') # includes 40 MeV cut 

        varlist = ["shr_score", "shrmoliereavg", "trkpid",
                "n_showers_contained", "shr_tkfit_dedx_Y", "tksh_distance",
                "tksh_angle", "subcluster", "trkshrhitdist2"]
        
    
        df_v = [df_intrinsic]
        
        for i,df in enumerate(df_v): 
            
            df['NeutrinoEnergy2_GeV'] = df['NeutrinoEnergy2']/1000
        
            df = df.query('trk_energy>0.04')
    
            df.loc[ df['weightSplineTimesTune'] <= 0, 'weightSplineTimesTune' ] = 1.
            df.loc[ df['weightSplineTimesTune'] == np.inf, 'weightSplineTimesTune' ] = 1.
            df.loc[ df['weightSplineTimesTune'] > 100, 'weightSplineTimesTune' ] = 1.
            df.loc[ np.isnan(df['weightSplineTimesTune']) == True, 'weightSplineTimesTune' ] = 1.

            
        df_full = pd.concat([df_intrinsic], ignore_index=True)

        # apply BDT 
        df_bdt = df_full.copy()

        # clean datasets & apply BDT model  
        for column in varlist:
            df_bdt.loc[(df_bdt[column] < -1.0e37) | (df_bdt[column] > 1.0e37), column] = np.nan
    
        df_test = xgb.DMatrix(data=df_bdt[varlist])
        preds = bdt_model.predict(df_test)
        df_bdt['BDT_score'] = preds
            
        print("Plotting BDT-selected histograms")
       
        if cut: # apply BDT score cut 
            df_sel = df_bdt.query(cut)
        else: 
            df_sel = df_bdt
        
        
        # create TH1F plots 
        for i in range(len(xsec_variables)): 

            # no GENIE or PPFX weights 
            h_uw = TH1F(variation+"_UW", xsec_variables[i]+" ("+variation+" Unweighted);;"+str(cv_pot), bins[i], xlow[i], xhigh[i])
        
            # with GENIE & PPFX weights
            h = TH1F(variation, xsec_variables[i]+" ("+variation+")", bins[i], xlow[i], xhigh[i])
            w = list(df_sel['ppfx_cv']*df_sel['weightSplineTimesTune']*df_sel['pot_scale'])
            
    
            for j in range(len(df_sel)): 
                h_uw.Fill(list(df_sel[xsec_variables[i]])[j], list(df_sel['pot_scale'])[j])
                h.Fill(list(df_sel[xsec_variables[i]])[j], w[j])
                
                #if np.isinf(list(df_sel[xsec_variables[i]])[j]) or np.isinf(list(df_sel['pot_scale'])[j]) or np.isinf(w[j]): 
                #    print(str(j)+' is inf')
                #if np.isnan(list(df_sel[xsec_variables[i]])[j]) or np.isnan(list(df_sel['pot_scale'])[j]) or np.isnan(w[j]): 
                #    print(str(j)+' is nan')
        
            h.SetDirectory(0)
    
            # save to file
            fout.cd()
        
            if fout.GetDirectory(xsec_variables[i]): 
                fout.cd(xsec_variables[i])
            else: 
                fout.mkdir(xsec_variables[i])
                fout.cd(xsec_variables[i])
        
            h.Write()
            h_uw.Write()
            
            h.Reset()
            h_uw.Reset()
    
        fout.Close()
        
    ############################################################################
    def ratio_to_CV_test(self, x, bin_edges, sample, intrinsic=False, moreStats=False):
        

        d = variations
        
        

        # list of lists for ratios to CV 
        v_counts = []
        weights = []
        

        f = uproot.open(path+"NuMI_FHC_PostBDT_Detector_Variations_"+sample+".root")[x]
 
        

        for v in list(d.keys()): 
            
            if intrinsic: 
                v = v+"_intrinsic"

            # counts in each widened bin 
            y = []
            
            # grab histogram 
            h = f[v]
            b = [round(x, 2) for x in h.edges] # old bin edges
            counts = list(h.values)
            
            # replace inf or nan counts with 0            
            replace_counter = 0
            for k in range(len(counts)): 
                if np.isnan(counts[k]) or np.isinf(counts[k]): 
                    counts[k] = 0.0
                    replace_counter = replace_counter + 1
            
            print('Replacing '+str(replace_counter)+' inf/nan events with 0.0')
                    
            # store counts for new binning 
            for i in range(1, len(bin_edges)): 
                
                start = b.index(bin_edges[i-1])
                stop = b.index(bin_edges[i])
                y.append(sum(counts[start:stop]))
                
            print(v)
            if "CV" not in v: 
                v_counts.append(y)
                # print(y)
                
            elif "CV" in v:
                ncv = y
                # print(y)
         
        return v_counts, ncv # weights is the ratio to CV (not included in test)
    
    #######################################################
    def plot_variations_test(self, var, bin_edges, sample, intrinsic=False, pot=None, axis_label=None, save=False):
        
        
        f = uproot.open(path+"NuMI_FHC_PostBDT_Detector_Variations_"+sample+".root")[var]

        d = variations
        cv_pot = intrinsic_variations['CV_intrinsic']
            
        if pot: 
            pot_scaling = pot/cv_pot
        else: 
            pot_scaling = 1
        
        fig = plt.figure(figsize=(8, 5))
        
        # loop over the variations 
        for v in list(d.keys()): 
            
            if intrinsic: 
                v = v+"_intrinsic"
            
            # grab histograms
            h = f[v]
            h_uw = f[v+"_UW"]
            
            b = [round(var, 2) for var in h.edges] # old bin edges
            counts = list(h.values)
            counts_uw = list(h_uw.values)
            
            # replace inf or nan counts with 0            
            for k in range(len(counts)): 
                if np.isnan(counts[k]) or np.isinf(counts[k]): 
                    counts[k] = 0.0
                if np.isnan(counts_uw[k]) or np.isinf(counts_uw[k]): 
                    counts_uw[k] = 0.0

            
            # store counts for new binning 
            y = []
            y_uw = []

            
            # now re-bin wider 
            for i in range(1, len(bin_edges)): 
                
                start = b.index(bin_edges[i-1])
                stop = b.index(bin_edges[i])
                
                y.append(sum(counts[start:stop]))
                y_uw.append(sum(counts_uw[start:stop]))

            # POT scaling for the weighted counts 
            y_scaled = [z*pot_scaling for z in y]
                    
            # compute bin centers 
            bin_centers = []
            
            for a in range(len(bin_edges)-1): 
                bin_centers.append(round(bin_edges[a] + (bin_edges[a+1]-bin_edges[a])/2, 3))

            # plot 
            if "CV" in v:
                
                # calculate MC stat error from (unweighted, unscaled) overlay POT 
                frac_stat_err = [np.sqrt(k)/k for k in y_uw]
                stat_err = [a*b for a,b in zip(frac_stat_err, y_scaled)]
                    
                plt.step(bin_edges+[bin_edges[-1]], [0]+y_scaled+[0], linewidth=2, color="black")
                plt.errorbar(bin_centers, y_scaled, yerr=stat_err, fmt='none', color='black', linewidth=2)
                
            else: 
                plt.step(bin_edges+[bin_edges[-1]], [0]+y_scaled+[0], linewidth=0.5, color="cornflowerblue")

        if pot: 
            plt.ylabel("$\\nu$ / "+str(pot)+" POT ", fontsize=15)
        else: 
            plt.ylabel("$\\nu$ / "+str(cv_pot)+" POT ", fontsize=15)
        
        if axis_label: 
            plt.xlabel(axis_label, fontsize=14)
        else: 
            plt.xlabel(var, fontsize=14)
        
        plt.ylim(bottom=0)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        
        plt.title("Detector Variations ("+sample+")", fontsize=14)

        
        if save: 
            plots_path = parameters(ISRUN3)['plots_path']
            plt.savefig(plots_path+str(var)+"_DetSys.pdf", transparent=True, bbox_inches='tight') 
            print('saving to: '+plots_path)

        
        plt.show()
    


import sys

sys.path.insert(0, 'backend_functions')

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

import top 
from top import *

import uncertainty_functions 
from uncertainty_functions import *

import xsec_functions 
from xsec_functions import smear_matrix

from ROOT import TH1D, TH2D, TDirectory, TH1F, TH2F

from selection_functions import *

import NuMIDetSys
importlib.reload(NuMIDetSys)

NuMIDetSysWeights = NuMIDetSys.NuMIDetSys()


# Make the detector systematics variations

output_file = "NuMI_FHC_QualCuts_DetectorVariations_April2022.root"
q = "swtrig_pre==1 and nslice==1 and " + reco_in_fv_query + " and contained_fraction>0.9 and shr_energy_tot_cali>0.07"
ISRUN3 = False 

xvar = "n_showers_contained"
bins = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5] 

# #### Create ROOT file with BDT-selected detector variations 

recreate_file = True

# skip this step if it is already created
# should manually delete the file first 
# (located here: /uboone/data/users/kmiller/uBNuMI_CCNp/ntuples/runX/systematics/detvar/)

# scales to the det sys CV POT (standard overlay)

if recreate_file: 
	for v in list(detvar_run1_fhc.keys()):
		NuMIDetSysWeights.makehist_detsys(v, ISRUN3, output_file, xvar, bins, cut=q, useBDT=False)


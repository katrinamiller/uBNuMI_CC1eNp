'''
import NuMIGeoWeights as bgw
bgwDict,bgEnVarDict,bgAnVarDict = bgw.createGeoWeigthsDictionary()

def calculateGeoWeights(neutrino, energy, angle):
    key  = calculateGeoWeightKey(neutrino, energy, angle,  energyDict, angleDict, nameDictionary )
    return(key)
'''

import os
import uproot
import numpy as np
import time




class NuMIGeoWeights:
    def __init__(self, geoVarRootFileName="/uboone/data/users/kmiller/systematics/NuMI_Beamline_Variations_to_CV_Ratios.root", current="FHC"):
        self.nameDictionary = {14: 'numu', -14: 'numubar', 12: 'nue', -12:'nuebar'}
        self.fullDictionary = None 
        self.energyEdges    = None
        self.angleEdges     = None 
        self.createGeoWeigthsDictionary(current, geoVarRootFileName)

    # creates a dictionary with keys = (neutrino flavor + energy bin + angular bin) that
    def createGeoWeigthsDictionary(self, current="FHC",geoVarRootFileName="/uboone/data/users/kmiller/systematics/NuMI_Beamline_Variations_to_CV_Ratios.root"):
        self.fullDictionary = {}
        self.energyEdges    = {}
        self.angleEdges     = {}
                                                                                                                                            
        # loop over neutrino flavors 
        for neutrino in self.nameDictionary.values():
            weights        = []
            thisFolder = uproot.open(geoVarRootFileName)['EnergyTheta2D'] # pull the 2D ratio histogram file
            ####### Let's fix the parameters here                                                                                                         
            # pick correct neutrino type & horn mode 
            histoName      = "ratio_run"+str(1)+"_"+str(current)+"_"+str(neutrino)+"_CV_AV_TPC_2D"
            # grab histogram for run 1 (use to get bin edges)
            thisHisto = thisFolder[histoName]
            # energy bins
            energy_v = (thisHisto.edges)[0]
            # angle bins
            angle_v  = (thisHisto.edges)[1]
            self.energyEdges[neutrino] = energy_v
            self.angleEdges[neutrino]  = angle_v
            ###############################                                                                                                               
            # loop over energy bins 
            for e in energy_v[:-1]:
                
                # loop over angular bins 
                for a in angle_v[:-1]:
                    
                    # loop over each beamline variation 
                    for variation in range(1,21):
                        
                        # grab histogram 
                        histoName      = "ratio_run"+str(variation)+"_"+str(current)+"_"+str(neutrino)+"_CV_AV_TPC_2D"
                        thisHisto   = thisFolder[histoName]
                        # get weights (z axis) 
                        weightArray = thisHisto.values
                        # fetch right weight                                                                                                                                                               
                        itemindexE  = np.where(energy_v < e+0.001)
                        itemindexA  = np.where( angle_v < a+0.001)
                        energyBin   = np.max(itemindexE)
                        angleBin    = np.max(itemindexA)
                        # Take care of out of range events                                                                                                                                                   
                        weight = 1.
                        if energyBin < weightArray.shape[0] and angleBin < weightArray.shape[1]:
                            weight    = weightArray[energyBin,angleBin]
                        weights.append( weight )
                    fullDictionaryKey = neutrino+"_"+str(e)+"_"+str(a)
                    self.fullDictionary[fullDictionaryKey] = weights
                    weights = [] # list of weights for each variation run (for a specified flavor, energy, & angle)


    def calculateGeoWeightKey(self, neutrino, energy, angle):
        key  = self.nameDictionary[neutrino]
        en_v = self.energyEdges[key]
        an_v = self.angleEdges[key]
        itemindexE  = np.where(en_v[:-1] <= energy)
        itemindexA  = np.where(an_v[:-1] <= angle)
        try:
            energyBin   = np.max(itemindexE)
            angleBin    = np.max(itemindexA)
        except:
            print(key,energy,angle)
            print(itemindexE, itemindexA)
        key += "_"+str(en_v[energyBin])+"_"+str(an_v[angleBin])
        return(key)

    def calculateGeoWeight(self, neutrino, energy, angle):
        key  = self.calculateGeoWeightKey(neutrino, energy, angle)
        w    = self.fullDictionary[key]
        return(w)
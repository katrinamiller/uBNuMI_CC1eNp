
This is a repository for MicroBooNE's NuMI nue CCNp cross section analysis using the PELEE searchingfornues framework. 


TK: instructions on how to setup Jupyter notebooks


Wiener SVD unfolding instructions can be found here: https://github.com/BNLIF/Wiener-SVD-Unfolding. 


~~ Primary Jupyter notebooks ~~

selection.ipynb: To train & evaluate the performance of the BDT selection, including comparison to a corresponding linear selection & cross validation. 


uncertainty.ipynb: To evaluate all sources of uncertainty on the differential cross section measurements. Saves variations and CV event rates to a json file in unfolding/variations/. Also used to create downstream data/MC comparison plots (after quality cuts).


smearing.ipynb: Uses the nue intrinsic sample to construct an efficiency-corrected response matrix. Saves true/reco generated & true/reco selected MC signal event rates to a json file in unfolding/smearing/. 


combined.ipynb: Pulls json files from unfolding/variations/ to produce a full, combined (FHC+RHC) covariance matrix for the WSVD unfolding tool. Also re-computes a combined efficiency, combined CV event rate, and combined response matrix. Saves the FHC, RHC, and FHC+RHC distributions to separate WSVD input files in /uboone/data/users/kmiller/uBNuMI_CCNp/unfolding/input/. 


unfolding.ipynb: Plotter for the WSVD unfolding output which converts results to cross section units. Saves output distributions to a json file in unfolding/results/ to compare between FHC, RHC, & FHC+RHC.



~~ Other Jupyter notebooks ~~

fakedata.ipynb: Runs the NuWro fake data samples through the BDT selection & compares the selected event rate to GENIE v3. Adds the NuWro CV reco selected event rate to the corresponding json file in unfolding/variations/. 


binning.ipynb: To determine final binning for FHC+RHC. 


detsys_checks.ipynb: Study of the statistical limitations for the detector systematics, to determine a flat average value to apply for cross section measurement. 


Run2Validation.ipynb: To validate Run 2 searchingfornues ntuples & resulting distributions for use in this analysis. 


pandas_slim.ipynb: To produce upstream data/MC comparisons by creating slimmed versions of the pandas dataframes. 


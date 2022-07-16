
/* Script to create run subrun list 
 * for POT counting tool
 */

#include <fstream> 
#include <iostream>
#include <cstdlib>
#include <string>
#include <vector>
#include <typeinfo>
#include "TDirectory.h"
#include "TROOT.h"
#include "TH1F.h"
#include "TH1D.h"
#include "TGraph.h"
#include "TH2D.h"
#include "TFile.h"
#include "TMath.h"
#include "TRandom.h"
#include "TMatrixT.h"
#include "TObject.h" 

void get_run_subrun_list(){ 

	TFile* f = new TFile("/uboone/data/users/kmistry/work/MCC9/searchingfornues/ntuple_files_v5/neutrinoselection_filt_run3b_beamon_beamgood.root", "READ");
	TTree*t = (TTree*)f->Get("nuselection/SubRun");

	// store new text file 
	//TFile* fout = new TFile("run_subrun_list.txt", "RECREATE"); 

	int run; 
	int subrun; 

	t->SetBranchAddress("subRun", &subrun);
	t->SetBranchAddress("run", &run);

	std::ofstream fout("run_subrun_list.txt");

	for (int i=0;i<t->GetEntries();i++) {
		t->GetEntry(i);
		//std::ofstream fout("run_subrun_list.txt"); 
		fout << std::to_string(run) + " " +  std::to_string(subrun) + "\n"; 
		std::cout << run << " " << subrun << std::endl; 
	} 

	std::cout << "Total entries in tree: " << t->GetEntries() << std::endl; 
	
	fout.close(); 
	f->Close(); 


} 

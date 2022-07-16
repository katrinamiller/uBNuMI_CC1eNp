#include <stdio.h>
#include <string.h>
#include "TChain.h"
#include "TFile.h"
#include "TH1.h"
#include "TTree.h"
#include "TKey.h"
#include "Riostream.h"
#include "TSystemDirectory.h"


void get_pot(std::string sample) {

	TFile* f = new TFile(Form("%s.root", sample.c_str()), "READ"); 

	TTree*t = (TTree*)f->Get("nuselection/SubRun"); 
	
	int run; 
	float pot;
	float total_pot = 0.0; 
	int n_skip = 0; 
	
	t->SetBranchAddress("pot", &pot); 
	t->SetBranchAddress("run", &run); 
	for (int i=0;i<t->GetEntries();i++) { 
		t->GetEntry(i); 
		/*if (run>10000) { 
			n_skip = n_skip + 1; 
			continue;
		}
		else { 
	//	std::cout << "pot of entry " << i << ": " << pot << std::endl; 
	*/      total_pot = total_pot + pot; 
			
	} 

	std::cout << "total POT of sample: " << total_pot << std::endl; 
	std::cout << "skipped " << n_skip << " runs " << std::endl; 
	f->Close(); 

}

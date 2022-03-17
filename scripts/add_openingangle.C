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

// compute the magnitude of all  MCParticles
std::vector<float> get_mcp(std::vector<float> *mc_px, std::vector<float> *mc_py, std::vector<float> *mc_pz){ 

	std::vector<float> mc_p;
        float p_mag;

	for (int j=0; j<mc_px->size(); j++) {

		p_mag = TMath::Sqrt(mc_px->at(j)*mc_px->at(j) + mc_py->at(j)*mc_py->at(j) + mc_pz->at(j)*mc_pz->at(j));
		mc_p.push_back(p_mag);

		// check 
		//std::cout << mc_px->at(j) << ", " << mc_py->at(j) << ", " << mc_pz->at(j) << ", " << p_mag << std::endl; 

		// clear for next iteration
		p_mag = 0; 
	}

	// RETURN a vector of magnitudes
	return mc_p; 
} 

// get the index of the  maximum proton momentum vector 
float get_max_proton_idx(std::vector<float> *mc_pdg, std::vector<float> mc_p){ 

	float p_proton = 0;
        float p_max_proton = 0;
        float p_max_proton_idx; 

	// for each particle
	for (int j=0; j<mc_p.size(); j++) {

		// if that particle is a proton 
		if (mc_pdg->at(j) == 2212) { 
			p_proton = mc_p.at(j);
			
			// if the momentum is larger than current max, replace
			if (p_proton > p_max_proton) { 
				p_max_proton = p_proton;
                        	p_max_proton_idx = j;
			}
		} 
	
	}	

	return p_max_proton_idx; 

} 


// construct the true opening angle & add it to the ntuples
void add_openingangle() { 

	
	TFile*f = new TFile("/uboone/data/users/kmiller/ntuples/run2/neutrinoselection_filt_55f2f593-7f5d-4b16-91ba-ea0abcc48e22.root", "UPDATE"); 
	TTree* t = (TTree*)f->Get("nuselection/NeutrinoSelectionFilter"); 
	TDirectory* d = (TDirectory*)f->Get("nuselection"); 
	
	std::vector<float> *mc_px = 0; 
	std::vector<float> *mc_py = 0; 
	std::vector<float> *mc_pz = 0;
 
	t->SetBranchAddress("mc_px", &mc_px); 
        t->SetBranchAddress("mc_py", &mc_py);
        t->SetBranchAddress("mc_pz", &mc_pz);

	std::vector<float> *mc_pdg = 0; 
	t->SetBranchAddress("mc_pdg", &mc_pdg); 


	float elec_px = 0; 
	float elec_py = 0; 
	float elec_pz = 0;

	t->SetBranchAddress("elec_px", &elec_px); 
	t->SetBranchAddress("elec_py", &elec_py); 
	t->SetBranchAddress("elec_pz", &elec_pz);  
	
	int nu_pdg = 0; 
	t->SetBranchAddress("nu_pdg", &nu_pdg); 

	std::vector<float> mc_p; 
	float p_idx; 
	
	float elec_p; 
	float proton_p;

	// add new branch to TTree
	float opening_angle = std::numeric_limits<float>::lowest(); // between the electron & proton 
	TBranch* b = t->Branch("opening_angle",&opening_angle,"opening_angle/F"); 

	// total number of events
	std::cout << "total number of events: " << t->GetEntries() << std::endl;

	// draw for events with an opening angle 
	TH1D* h = new TH1D("h", "", 140, -1.2, 1.2); 

	for (int i=0; i<t->GetEntries(); i++) { 
	  
		t->GetEntry(i); 
		
		//std::cout << "event " << i << std::endl; 
		
		// get a vector of momentum magnitudes		
		mc_p = get_mcp(mc_px, mc_py, mc_pz); 

		// print vector of pdg
		//for (std::vector<float>::const_iterator i = mc_pdg->begin(); i != mc_pdg->end(); ++i) { 
    		//	std::cout << *i << ' ';
		//}


		// if there is a true electron 
		if (nu_pdg==12 or nu_pdg==-12) {
			//std::cout << "PDG code: " << nu_pdg << std::endl; 
			elec_p = TMath::Sqrt(elec_px*elec_px + elec_py*elec_py + elec_pz *elec_pz);

		 	// if there is a proton
		 	if (std::find(mc_pdg->begin(), mc_pdg->end(), 2212) != mc_pdg->end()) {
				p_idx = get_max_proton_idx(mc_pdg, mc_p);
				//std::cout << "max proton index = " << p_max_proton_idx << std::endl; 
				
				proton_p = TMath::Sqrt(mc_px->at(p_idx)*mc_px->at(p_idx) + mc_py->at(p_idx)*mc_py->at(p_idx) + mc_pz->at(p_idx)*mc_pz->at(p_idx));

				if (elec_p != 0 and proton_p !=0) {
                                        opening_angle = (elec_px*mc_px->at(p_idx) + elec_py*mc_py->at(p_idx) + elec_pz*mc_pz->at(p_idx)) / (elec_p*proton_p);
                                	h->Fill(opening_angle); 
				}
			//std::cout << "opening angle: " << opening_angle << std::endl; 
		
			}

			else { // no proton 
				opening_angle = std::numeric_limits<float>::lowest();
				//std::cout << "no proton in event!" << std::endl; 
			} 

		} 

		else { // no electron 
			opening_angle = std::numeric_limits<float>::lowest();
			//std::cout << "no electron in event!" << std::endl; 
		} 

		//std::cout << opening_angle << std::endl; 
		b->Fill(); 
				 
		// clear everything
	
		mc_p.clear(); 

		mc_px->clear(); 
		mc_py->clear(); 
		mc_pz->clear(); 

		elec_p = 0; 
		proton_p = 0; 
		opening_angle = std::numeric_limits<float>::lowest(); 


	} 

	 	
	//t->Draw("opening_angle", "-1<opening_angle<1");

	//h->Draw(); 	 

	f->cd(); 
	d->cd(); 
	t->Write("", TObject::kOverwrite); 

	f->Close(); 	


}

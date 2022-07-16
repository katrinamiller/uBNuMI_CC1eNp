/****
 *
 * Back-end script to make histograms for the detector systematics
 * at different stages of the nue CCNp selection 
 *
 * last modified: April 2022 (K. Miller krm29@uchicago.edu) 
 *
 * ****/

#include "Riostream.h"
#include <map>
#include "TH1F.h"
#include "TH1D.h"
#include "TDirectory.h" 
#include <vector>


// dictionary for POT values 
std::map<std::string, double> get_pot(std::string dataset) { 

	std::map<std::string, double> detvar_run1_fhc; 

    	detvar_run1_fhc["LYRayleigh"] = 7.60573E20; 
    	detvar_run1_fhc["LYDown"] = 7.43109E20; 
    	detvar_run1_fhc["SCE"] = 7.39875E20;
    	detvar_run1_fhc["Recomb2"] = 7.59105E20; 
    	detvar_run1_fhc["WireModX"] =  7.64918E20; 
    	detvar_run1_fhc["WireModYZ"] =  7.532E20; 
    	detvar_run1_fhc["WireModThetaXZ"] = 7.64282E20;
    	detvar_run1_fhc["WireModThetaYZ_withSigmaSplines"] =  7.64543E20;  
    	detvar_run1_fhc["CV"] =  7.59732E20; 

	std::map<std::string, double> intrinsic_detvar_run1_fhc;

        intrinsic_detvar_run1_fhc["LYRayleigh_intrinsic"] = 2.38081E22; 
        intrinsic_detvar_run1_fhc["LYDown_intrinsic"] = 2.24505E22; 
        intrinsic_detvar_run1_fhc["SCE_intrinsic"] = 2.39023E22;
        intrinsic_detvar_run1_fhc["Recomb2_intrinsic"] = 2.38193E22;
        intrinsic_detvar_run1_fhc["WireModX_intrinsic"] = 2.38318E22;
        intrinsic_detvar_run1_fhc["WireModYZ_intrinsic"] = 2.38416E22;
        intrinsic_detvar_run1_fhc["WireModThetaXZ_intrinsic"] = 2.31518E22;
        intrinsic_detvar_run1_fhc["WireModThetaYZ_withSigmaSplines_intrinsic"] =  2.31421E22;
        intrinsic_detvar_run1_fhc["CV_intrinsic"] =  2.37261E22;

	std::map<std::string, double> pot_values; 
	if (dataset=="detvar_run1_fhc") { 
		pot_values = detvar_run1_fhc; 
	} 

        elif (dataset=="intrinsic_detvar_run1_fhc") {
                pot_values = intrinsic_detvar_run1_fhc;
        }

	return pot_values

}



void makehist_detsys(std::string variation, std::string run, std::string output_file, std::string xvar, vector<double> bins){

	std::string run = "run1"; 
	std::string output_file = ""; // make sure to update the cuts ! 
	std::string xvar_str = ""; 
	vector<double> bins = {}; 
	
	std::map<std::string, double> pot_values_standard = "detvar_run1_fhc"; 
	std::map<std::string, double> pot_values_intrinsic = "intrinsic_detvar_run1_fhc";	

	double cv_pot = pot_values_standard.find("CV"); 


	if ( run != "run1" ) { 
		std::cout << "need to update for Run 2 and Run 3!" << std::endl; 
		break; 
	} 


	std::string path = "/uboone/data/users/kmiller/uBNuMI_CCNp/ntuples/"+run+"/systematics/detvar/"; 
	
	TFile* f_standard = new TFile(path + "standard_overlay/" + "neutrinoselection_filt_" + run + "_overlay_" + variation + ".root", "READ"); 
	TFile* f_intrinsic = new TFile(path + "intrinsic/" + "neutrinoselection_filt_" + run + "_overlay_" + variation + "_intrinsic.root", "READ");

	std::cout << "Opening files for " << variation << std::endl; 

	// new file to store detvar histograms 	
	TFile* fout = new TFile(path+"makehist_detsys_output/"+output_file, "UPDATE"); 

	TTree* standard_tree; 
	f_standard.GetObject("nuselection/NeutrinoSelectionFilter", standard_tree); 
	standard_pot_scale = cv_pot/pot_values_standard.find(variation); 

        TTree* intrinsic_tree;
        f_intrinsic.GetObject("nuselection/NeutrinoSelectionFilter", intrinsic_tree);
	intrinsic_pot_scale = cv_pot/pot_values_intrinsic.find(variation+"_intrinsic"); 
   
   	int swtrig_pre; 

   	int nslice;
   	float reco_nu_vtx_sce_x, reco_nu_vtx_sce_y, reco_nu_vtx_sce_z;
   	float contained_fraction; 
   	
	UInt_t n_tracks_contained; 
   	UInt_t n_showers_contained; 
   	float shr_energy_tot_cali; 
   	float trk_energy; 

   	float shr_score; 
   	float trkpid; 
   	float shrmoliereavg; 
   	float shr_tkfit_dedx_Y; 
   	float tksh_distance; 

	float weightSplineTimesTune, ppfx_cv; 
	float nu_pdg; 
	float ccnc; 
	float true_nu_vtx_x, true_nu_vtx_y, true_nu_vtx_z; 

	float xvar; 

   	standard_tree->SetBranchAddress("swtrig_pre", &swtrig_pre); 

   	standard_tree->SetBranchAddress("nslice", &nslice);
   	standard_tree->SetBranchAddress("reco_nu_vtx_sce_x", &reco_nu_vtx_sce_x);
   	standard_tree->SetBranchAddress("reco_nu_vtx_sce_y", &reco_nu_vtx_sce_y);
   	standard_tree->SetBranchAddress("reco_nu_vtx_sce_z", &reco_nu_vtx_sce_z);
   	standard_tree->SetBranchAddress("contained_fraction", &contained_fraction);
   	
	standard_tree->SetBranchAddress("n_tracks_contained", &n_tracks_contained);
   	standard_tree->SetBranchAddress("n_showers_contained", &n_showers_contained);
   	standard_tree->SetBranchAddress("shr_energy_tot_cali", &shr_energy_tot_cali);
   	standard_tree->SetBranchAddress("trk_energy", &trk_energy);

   	standard_tree->SetBranchAddress("shr_score", &shr_score); 
   	standard_tree->SetBranchAddress("trkpid", &trkpid); 
   	standard_tree->SetBranchAddress("shrmoliereavg", &shrmoliereavg); 
   	standard_tree->SetBranchAddress("shr_tkfit_dedx_Y", &shr_tkfit_dedx_Y); 
   	standard_tree->SetBranchAddress("tksh_distance", &tksh_distance);  

	standard_tree->SetBranchAddress("weightSplineTimesTune", &weightSplineTimesTune);
	standard_tree->SetBranchAddress("ppfx_cv", &ppfx_cv); 
	standard_tree->SetBranchAddress("nu_pdg", &nu_pdg);
	standard_tree->SetBranchAddress("ccnc", &ccnc);
	standard_tree->SetBranchAddress("true_nu_vtx_x", &true_nu_vtx_x); 
	standard_tree->SetBranchAddress("true_nu_vtx_y", &true_nu_vtx_y);
	standard_tree->SetBranchAddress("true_nu_vtx_z", &true_nu_vtx_z);
	standard_tree->SetBranchAddress(xvar_str, &xvar); 


        intrinsic_tree->SetBranchAddress("swtrig_pre", &swtrig_pre);

        intrinsic_tree->SetBranchAddress("nslice", &nslice);
        intrinsic_tree->SetBranchAddress("reco_nu_vtx_sce_x", &reco_nu_vtx_sce_x);
        intrinsic_tree->SetBranchAddress("reco_nu_vtx_sce_y", &reco_nu_vtx_sce_y);
        intrinsic_tree->SetBranchAddress("reco_nu_vtx_sce_z", &reco_nu_vtx_sce_z);
        intrinsic_tree->SetBranchAddress("contained_fraction", &contained_fraction);
        
        intrinsic_tree->SetBranchAddress("n_tracks_contained", &n_tracks_contained);
        intrinsic_tree->SetBranchAddress("n_showers_contained", &n_showers_contained);
        intrinsic_tree->SetBranchAddress("shr_energy_tot_cali", &shr_energy_tot_cali);
        intrinsic_tree->SetBranchAddress("trk_energy", &trk_energy);

        intrinsic_tree->SetBranchAddress("shr_score", &shr_score);
        intrinsic_tree->SetBranchAddress("trkpid", &trkpid);
        intrinsic_tree->SetBranchAddress("shrmoliereavg", &shrmoliereavg);
        intrinsic_tree->SetBranchAddress("shr_tkfit_dedx_Y", &shr_tkfit_dedx_Y);
        intrinsic_tree->SetBranchAddress("tksh_distance", &tksh_distance);

	intrinsic_tree->SetBranchAddress("weightSplineTimesTune", &weightSplineTimesTune);
        intrinsic_tree->SetBranchAddress("ppfx_cv", &ppfx_cv);
        intrinsic_tree->SetBranchAddress("nu_pdg", &nu_pdg);
        intrinsic_tree->SetBranchAddress("ccnc", &ccnc);
        intrinsic_tree->SetBranchAddress("true_nu_vtx_x", &true_nu_vtx_x); 
        intrinsic_tree->SetBranchAddress("true_nu_vtx_y", &true_nu_vtx_y);
        intrinsic_tree->SetBranchAddress("true_nu_vtx_z", &true_nu_vtx_z);
	intrinsic_tree->SetBranchAddress(xvar_str, &xvar);	

	
	// get rid of nueCC events in the standard overlay sample 
	std::cout << "Removing nue CC events in standard overlay ...." << std::endl; 

	int nentries = standard_tree->GetEntries(); 
	int isnueCC; 
	auto nueCC_branch = standard_tree->("isnueCC", &isnueCC, "isnueCC/i"); 

	for (auto i : ROOT::TSeqI(nentries)) { 
		standard_tree->GetEntry(i); 

		if ( abs(nu_pdg==12) && (ccnc==0) && (-1.55<=true_nu_vtx_x<=254.8) && (-116.5<=true_nu_vtx_y<=116.5) && (0<=true_nu_vtx_z<=1036.8) ) { 
			isnueCC = 1; 
		} 	
		else { 
			isnueCC = 0; 
		} 

		nueCC_branch->Fill(); 
		isnueCC = NULL; 
	} 
	
	// fill histograms based on chosen set of cuts
	TH1D * h = new TH1D(variation, xvar_str+" ("+variation+" - Full Selected Event Rate)", bins.size(), np.array(bins)); 
	
	std::cout << "Now filling histogram ...." << std::endl;  
	for (auto i : ROOT::TSeqI(nentries)) {

		standard_tree->GetEntry(i); 
		intrinsic_tree
		
}





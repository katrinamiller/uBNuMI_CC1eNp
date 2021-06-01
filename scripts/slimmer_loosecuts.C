#include "Riostream.h"
#include <map>
#include <vector>

void slimmer_loosecuts(TString fname)
{
  
   // Get old file, old tree and set top branch address
   TString dirorigin = "/uboone/data/users/kmistry/work/MCC9/searchingfornues/ntuple_files_detvar_newtune/run1/intrinsic/"; 
    //"/uboone/data/users/kmiller/systematics/detvar/run1/";
   
   TString dir = "/uboone/data/users/kmiller/systematics/detvar/run1/";
   TString fullpath = dirorigin + fname + ".root";
   std::cout<<"Name "<< fullpath<<" "<<fname <<"\n"; 
   TString foutname = dir + "slimmed_loosecuts/" + fname + "_slim.root";
   gSystem->ExpandPathName(dir);
 
   TFile oldfile(fullpath, "READ");
   TTree *oldtree;
   oldfile.GetObject("nuselection/NeutrinoSelectionFilter", oldtree);

   int numevts = 0;
   
   int nentries = oldtree->GetEntries();
   
   printf("There are %i entries \n",nentries);
   
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

   oldtree->SetBranchAddress("swtrig_pre", &swtrig_pre); 

   oldtree->SetBranchAddress("nslice", &nslice);
   oldtree->SetBranchAddress("reco_nu_vtx_sce_x", &reco_nu_vtx_sce_x);
   oldtree->SetBranchAddress("reco_nu_vtx_sce_y", &reco_nu_vtx_sce_y);
   oldtree->SetBranchAddress("reco_nu_vtx_sce_z", &reco_nu_vtx_sce_z);
   oldtree->SetBranchAddress("contained_fraction", &contained_fraction);
   oldtree->SetBranchAddress("n_tracks_contained", &n_tracks_contained);
   oldtree->SetBranchAddress("n_showers_contained", &n_showers_contained);
   oldtree->SetBranchAddress("shr_energy_tot_cali", &shr_energy_tot_cali);
   oldtree->SetBranchAddress("trk_energy", &trk_energy);

   oldtree->SetBranchAddress("shr_score", &shr_score); 
   oldtree->SetBranchAddress("trkpid", &trkpid); 
   oldtree->SetBranchAddress("shrmoliereavg", &shrmoliereavg); 
   oldtree->SetBranchAddress("shr_tkfit_dedx_Y", &shr_tkfit_dedx_Y); 
   oldtree->SetBranchAddress("tksh_distance", &tksh_distance);  

   std::cout << "Create a new file + a clone of old tree in new file.... " << std::endl; 
   TFile newfile(foutname, "recreate");
   TDirectory *searchingfornues = newfile.mkdir("nuselection");
   searchingfornues->cd();
   
   auto newtree = oldtree->CloneTree(0);
  
   std::cout << "Beginning slimming process...." << std::endl; 
   for (auto i : ROOT::TSeqI(nentries)) {

      oldtree->GetEntry(i);

     if (i%10000 == 0){
       printf("\t entry %i \n",i);
       //std::cout << "weightGenie : " << weightsGenie->at(0) << std::endl;
     }


     if ( (swtrig_pre==1) && (nslice == 1) && (reco_nu_vtx_sce_x>=10)  && (reco_nu_vtx_sce_x<=246) && (reco_nu_vtx_sce_y>=-106) && (reco_nu_vtx_sce_y<=106) && (reco_nu_vtx_sce_z>=10)  && (reco_nu_vtx_sce_z<=1026) && (contained_fraction > 0.9) && (n_tracks_contained>0) && (n_showers_contained==1) && (shr_energy_tot_cali > 0.07) && (shr_score<0.3) && (trkpid<0.35) && (shrmoliereavg<15) && (shr_tkfit_dedx_Y<7) && (tksh_distance<12) ) { 
       newtree->Fill();
     } 
     //std::cout << "end query" << std::endl; 
     
   }// for all entries
   //newtree->Print();
   
  TTree *subrunTree;
  oldfile.GetObject("nuselection/SubRun", subrunTree);
  auto newSubrunTree = subrunTree->CloneTree();


   newfile.Write();
}

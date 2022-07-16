#include "Riostream.h"
#include <map>
#include <vector>

void slimmer_qualcuts(TString fname)
{
  
   // Get old file, old tree and set top branch address
   TString dirorigin = "/uboone/data/users/kmiller/uBNuMI_CCNp/ntuples/run3b/cv/";
   TString dir = "/uboone/data/users/kmiller/uBNuMI_CCNp/ntuples/run3b/cv_slimmed/qualcuts/";
   TString fullpath = dirorigin + fname + ".root";
   std::cout<<"Name "<< fullpath<<" "<<fname <<"\n"; 
   TString foutname = dir + fname + ".root";
   gSystem->ExpandPathName(dir);
 
   TFile oldfile(fullpath);
   TTree *oldtree;
   oldfile.GetObject("nuselection/NeutrinoSelectionFilter", oldtree);
   
   std::cout << "Disabling some branches...." << std::endl; 
   for (auto b : *(oldtree->GetListOfBranches())) 
     {
       if (std::strncmp(b->GetName(),"trk_pid_chi",11)==0) {
	 //std::cout << "skip " << b->GetName() << std::endl;
	 oldtree->SetBranchStatus(b->GetName(), 0);
	 continue;
       }
       if (std::strncmp(b->GetName(),"trk_pida",8)==0) {
	 //std::cout << "skip " << b->GetName() << std::endl;
	 oldtree->SetBranchStatus(b->GetName(), 0);
	 continue;
       }
       if (std::strncmp(b->GetName(),"trk_bragg",9)==0) {
	 //std::cout << "skip " << b->GetName() << std::endl;
	 oldtree->SetBranchStatus(b->GetName(), 0);
	 continue;
       }
       if (std::strncmp(b->GetName(),"backtracked_start",17)==0) {
	 //std::cout << "skip " << b->GetName() << std::endl;
	 oldtree->SetBranchStatus(b->GetName(), 0);
	 continue;
       }
       if (std::strncmp(b->GetName(),"backtracked_sce_start",21)==0) {
	 //std::cout << "skip " << b->GetName() << std::endl;
	 oldtree->SetBranchStatus(b->GetName(), 0);
	 continue;
       }
       /*
       if (std::strncmp(b->GetName(),"pfpplane",8)==0) {
	 //std::cout << "skip " << b->GetName() << std::endl;
	 oldtree->SetBranchStatus(b->GetName(), 0);
	 continue;
	 }
       */
       if (std::strncmp(b->GetName(),"weights",10)==0) { // for MC
	 //if (std::strncmp(b->GetName(),"weights",7)==0) { // for data
	 //std::cout << "skip " << b->GetName() << std::endl;
	 oldtree->SetBranchStatus(b->GetName(), 0);
	 continue;
       }
       oldtree->SetBranchStatus(b->GetName(), 1);
     }


//   std::vector<unsigned short> *weightsGenie = 0;
//   std::vector<unsigned short> *weightsFlux = 0;
//   std::vector<unsigned short> *weightsReint = 0;

//   oldtree->SetBranchAddress("weightsGenie",&weightsGenie);
//   oldtree->SetBranchAddress("weightsFlux",&weightsFlux);
//   oldtree->SetBranchAddress("weightsReint",&weightsReint);

   int numevts = 0;
   
   int nentries = oldtree->GetEntries();
   
   printf("There are %i entries \n",nentries);
   
   // Deactivate all branches
   //oldtree->SetBranchStatus("*", 1);
   
   int nslice;
   float topological_score;
   float reco_nu_vtx_sce_x, reco_nu_vtx_sce_y, reco_nu_vtx_sce_z;
   int selected;
   float shr_energy_tot_cali;
   float contained_fraction; 

   oldtree->SetBranchAddress("nslice", &nslice);
   oldtree->SetBranchAddress("topological_score", &topological_score);
   oldtree->SetBranchAddress("reco_nu_vtx_sce_x", &reco_nu_vtx_sce_x);
   oldtree->SetBranchAddress("reco_nu_vtx_sce_y", &reco_nu_vtx_sce_y);
   oldtree->SetBranchAddress("reco_nu_vtx_sce_z", &reco_nu_vtx_sce_z);
   oldtree->SetBranchAddress("selected", &selected);
   oldtree->SetBranchAddress("shr_energy_tot_cali", &shr_energy_tot_cali);
   oldtree->SetBranchAddress("contained_fraction", &contained_fraction); 

   std::cout << "Create a new file + a clone of old tree in new file.... " << std::endl; 
   TFile newfile(foutname, "recreate");
   TDirectory *searchingfornues = newfile.mkdir("nuselection");
   searchingfornues->cd();
   
   auto newtree = oldtree->CloneTree(0);
   //oldtree->SetBranchStatus("weightsGenie", 0);
   //oldtree->SetBranchStatus("weightsFlux ", 0);
   //oldtree->SetBranchStatus("weightsReint", 0);

   //newtree->Branch("weightsGenieSub", "std::vector<unsigned short>", &weightsGenie);
   //newtree->Branch("weightsReintSub", "std::vector<unsigned short>", &weightsReint);
   //newtree->Branch("weightsFluxSub" , "std::vector<unsigned short>", &weightsFlux );

   std::cout << "Beginning slimming process...." << std::endl; 
   for (auto i : ROOT::TSeqI(nentries)) {

      oldtree->GetEntry(i);

     if (i%10000 == 0){
       printf("\t entry %i \n",i);
       //std::cout << "weightGenie : " << weightsGenie->at(0) << std::endl;
     }


    /* if (weightsGenie->size() > 0) {
       weightsGenie->resize(100);
       weightsReint->resize(100);
       weightsFlux->resize(100);
     }*/

     //std::cout << "query" << std::endl; 
     if ( (nslice == 1) && (reco_nu_vtx_sce_x>=10)  && (reco_nu_vtx_sce_x<=246) && (reco_nu_vtx_sce_y>=-106) && (reco_nu_vtx_sce_y<=106) && (reco_nu_vtx_sce_z>=10)  && (reco_nu_vtx_sce_z<=1026) && (contained_fraction > 0.9) ) { 
       newtree->Fill();
     } 
     //std::cout << "end query" << std::endl; 
     
   }// for all entries
   //newtree->Print();
   
  TTree *subrunTree;
  oldfile.GetObject("nuselection/SubRun", subrunTree);
  auto newSubrunTree = subrunTree->CloneTree();
  //newSubrunTree->Print();


   newfile.Write();
}

#include "Riostream.h"
#include <map>
#include <vector>

void split_run3b(TString fname)
{
  
   // Get old file, old tree and set top branch address
   TString dirorigin = "/uboone/data/users/kmiller/uBNuMI_CCNp/ntuples/run3b/cv/"; 
   TString fullpath = dirorigin + fname + ".root";
   std::cout<<"Name "<< fullpath<<" "<<fname <<"\n"; 

   TString fout1 = dirorigin + fname + "_split1.root";
   TString fout2 = dirorigin + fname + "_split2.root";
 
   TFile oldfile(fullpath);
   //TTree *oldtree;
   //oldfile.GetObject("nuselection/NeutrinoSelectionFilter", oldtree);
   
   TTree *subrunTree;
   oldfile.GetObject("nuselection/SubRun", subrunTree);

   int numevts = 0;
   int nentries = subrunTree->GetEntries();
   
   printf("There are %i entries \n",nentries);
   
   int run;

   subrunTree->SetBranchAddress("run", &run);
   
   std::cout << "Create news file + a clone of old trees in new files.... " << std::endl; 
   TFile newfile1(fout1, "recreate");
   TFile newfile2(fout2, "recreate");

   TDirectory *searchingfornues1 = newfile1.mkdir("nuselection");
   TDirectory *searchingfornues2 = newfile2.mkdir("nuselection");

   searchingfornues1->cd();
   
   //auto newtree1 = oldtree->CloneTree(0);
   auto subruntree1 = subrunTree->CloneTree(0);

   searchingfornues2->cd(); 
   //auto newtree2 = oldtree->CloneTree(0);
   auto subruntree2 = subrunTree->CloneTree(0);
    
   for (auto i : ROOT::TSeqI(nentries)) {

      subrunTree->GetEntry(i);

     if (i%10000 == 0){
       printf("\t entry %i \n",i);
     }

     if ( run<16880 ) { 
       //newtree1->Fill();
       subruntree1->Fill(); 
     } 
     else { 
       //newtree2->Fill(); 
       subruntree2->Fill();
     }
    
     
   }

   newfile1.Write();
   newfile2.Write(); 
   oldfile.Close(); 

}

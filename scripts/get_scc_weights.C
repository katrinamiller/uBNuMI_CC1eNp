#include <iostream>
#include <string>
#include <vector>

  // NOTE: For each of the Fv3 and Fa3 universes, multiply
  //   //   spline * tune * fv3_weight (or fa3_weight) to get the overall
  //     //   weight for the event. The CV to compare the result to is still
  //       //   spline * tune. This procedure is necessary for weird technical
  //         //   reasons (in short, the SCC weight calculator doesn't "know" about
  //           //   the tuned central value).


void get_scc_weights(TString filename) { 

	// Open the Run 1 BNB CV MC ntuple file
	TFile* f = new TFile(filename, "READ"); 

	// Retrieve the TTree containing the ntuple branches
	TTree* nsf = nullptr;
  	f->GetObject( "nuselection/NeutrinoSelectionFilter", nsf );

	// Set the branch address for reading the map of event weights
	std::map< std::string, std::vector<double> >* wgts = nullptr;
  	nsf->SetBranchAddress("weights", &wgts );

	// Get an arbitrary ntuple entry for testing purposes
	nsf->GetEntry(5);


	// Print out all of the keys in the map. These are the names of all of the
   	// weights that are available. Also include the size of the vector of weights
	// for each one. The vector sizes are the same as the number of universes.

	std::cout << "\n\n***AVAILABLE WEIGHTS***\n";
	for ( const auto& pair : *wgts ) std::cout << pair.first
		<< ' ' << pair.second.size() << '\n';

	// Access the vectors of weights for the Fv3 and Fa3 multisims
	const auto& fv3_wgts = wgts->at( "xsr_scc_Fv3_SCC" );
	const auto& fa3_wgts = wgts->at( "xsr_scc_Fa3_SCC" );

	std::cout << "Fv3 weights:";
  	for ( const double& w : fv3_wgts ) std::cout << ' ' << w;
  	std::cout << '\n';
 
  	std::cout << "Fa3 weights:";
  	for ( const double& w : fa3_wgts ) std::cout << ' ' << w;
  	std::cout << '\n';

}

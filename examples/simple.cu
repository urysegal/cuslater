//
// Created by gkluhana on 26/03/24.
//

#include "../include/evalIntegral.h"
#include <iomanip> // for std::setprecision
#include <iostream>
#include <chrono>
#include <string>
using namespace std;

int
main(int argc, const char *argv[])
{
	// Default Parameter Values
	cuslater::ProgramParameters sys;
	// Process Input Parameters
	cuslater::handleArguments(argc, argv, sys);

	int nr = sys.nr;
	int nl = sys.nl;
	int nx = sys.nx;
	int ny = sys.ny;
	int nz = sys.nz;
	float alpha[4];
	float c[12];

	for (int i = 0; i < 4; ++i) {
    		alpha[i] = sys.alpha[i];
	}	
	for (int i = 0; i < 12; ++i) {
    		c[i] = sys.c[i];
	}	

	double tol = sys.tol;
	int check_zero_cond = sys.check_zero_cond;

        const std::string x1_type = "legendre"; //legendre or simpson

        auto start = std::chrono::high_resolution_clock::now();
	double sum = 0;
        sum = cuslater::evaluateFourCenterIntegral(c, alpha, nr, nl, nx, ny, nz, x1_type, tol, check_zero_cond);
        auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "nr: " << nr << " nl: " << nl << " nx: " << nx << " ny: " << ny << " nz: " << nz << std::endl;
        std::cout << "result: " << std::fixed << std::setprecision(std::numeric_limits<double>::max_digits10) << sum << std::endl;
        std::cout << "Time Elapsed: " << duration.count()/1e6 << " seconds" << std::endl;
}

//
// Created by gkluhana on 26/03/24.
//

#include "../include/evalIntegral.h"
#include <iomanip> // for std::setprecision
#include <iostream>
#include <chrono>
using namespace std;

int
main(int argc, const char *argv[])
{
        int nr = 56;
        int nl = 74;
        int nx = 600;
        double c[] = {0,0,0,
                      1,0,0,
                      2,0,0,
                      3,0,0};

        const std::string x1_type = "legendre"; //legendre or simpson

	auto start = std::chrono::high_resolution_clock::now();
        auto sum = cuslater::evaluateTotalIntegral(c,nr,nl,nx,x1_type);
//        auto sum = cuslater::evaluateTotalIntegral(c,nr,nl,nx,x1_type,4);
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "nr: " << nr << " nl: " << nl << " nx: " << nx << std::endl;
        std::cout << "result: " << std::fixed << std::setprecision(std::numeric_limits<double>::max_digits10) << sum << std::endl;
        std::cout << "Time Elapsed: " << duration.count()/1e6 << " seconds" << std::endl;
//	for (int i=0; i<10; ++i){
//		nx += 25;
//		start = std::chrono::high_resolution_clock::now();
//		sum = cuslater::evaluateTotalIntegral(c,nr,nl,nx,x1_type);
//		end = std::chrono::high_resolution_clock::now();
//		duration = std::chrono::duration_cast<std::chrono::microseconds>(end-start);
//        	std::cout << "nr: " << nr << " nl: " << nl << " nx: " << nx << std::endl;
//        	std::cout << "result: " << std::fixed << std::setprecision(std::numeric_limits<double>::max_digits10) << sum << std::endl;
//        	std::cout << "Time Elapsed: " << duration.count()/1e6 << " seconds" << std::endl;
//	}	
}

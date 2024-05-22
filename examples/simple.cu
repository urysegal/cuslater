//
// Created by gkluhana on 26/03/24.
//
// editted by MarkEwert03 on 13/05/24

#include <chrono>
#include <iomanip>  // for std::setprecision
#include <iostream>

#include "../include/evalIntegral.h"
using namespace std;

int main(int argc, const char *argv[]) {
    int nr = 89;
    int nl = 590;
    int nx = 375;
    int ny = 375;
    int nz = 375;
    double tol = 1e-10;
    if (argc == 2 and argv[1][0] == 'd') {
        nr = 2;
        nl = 6;
        nx = 21;
        ny = 21;
        nz = 21;
    }
    if (argc == 3 and argv[1][0] == 't') {
        tol = std::stod(argv[2]);
    }
    float c[] = {0, 0, 0, 1, 0, 0, 2, 0, 0, 3, 0, 0};
    float alphas[] = {1, 1, 1, 1};

    const std::string x1_type = "legendre";  // legendre or simpson

    auto start = std::chrono::high_resolution_clock::now();
    auto sum = cuslater::evaluateFourCenterIntegral(c, alphas, nr, nl, nx, ny,
                                                    nz, x1_type, tol);
    // auto sum = cuslater::evaluateFourCenterIntegral(c,nr,nl,nx,x1_type,4);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "nr: " << nr << " nl: " << nl << " nx: " << nx << std::endl;
    std::cout << "result: " << std::fixed
              << std::setprecision(std::numeric_limits<double>::max_digits10)
              << sum << std::endl;
    std::cout << "Time Elapsed: " << duration.count() / 1e6 << " seconds"
              << std::endl;
}

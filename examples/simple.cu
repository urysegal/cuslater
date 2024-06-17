// Created by gkluhana on 26/03/24.
//
// editted by MarkEwert03 on 13/05/24

#include <chrono>
#include <iomanip> // for std::setprecision
#include <iostream>

#include "../include/evalIntegral.h"
using namespace std;

int main(int argc, const char *argv[]) {
    // represents number of points to calculate in the x,y,z direction
    int nx = 375;
    int ny = 375;
    int nz = 375;
    int nr = 89;
    int nl = 590;
    double tol = 1e-10;

    // α1 = alpha[0], α2 = alpha[1], α3 = alpha[2], α4 = alpha[3]
    float alpha[] = {1, 1, 1, 1};

    if (argc == 3) {
        std::string argv1 = std::string(argv[1]);
        if (argv1 == "-a") {
            float val = std::atoi(argv[2]);
            alpha[0] = val;
            alpha[1] = val;
            alpha[2] = val;
            alpha[3] = val;
        } else if (argv1 == "-t") {
            tol = std::stod(argv[2]);
        }
    }

    // c1 = (c1.x, c1.y, c1.z) = (c[0], c[1], c[2])
    // c2 = (c2.x, c2.y, c2.z) = (c[3], c[4], c[5])
    // c3 = (c3.x, c3.y, c3.z) = (c[6], c[7], c[8])
    // c4 = (c4.x, c4.y, c4.z) = (c[9], c[10], c[11])
    float c[] = {0, 0, 0, 1, 0, 0, 2, 0, 0, 3, 0, 0};

    const std::string x1_type = "legendre"; // legendre or simpson

    auto start = std::chrono::high_resolution_clock::now();

    // this is where all the computation time takes place
    double sum = cuslater::evaluateFourCenterIntegral(c, alpha, nr, nl, nx, ny, nz, x1_type, tol);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::stringstream message;
    message << "nr=" << nr;
    message << " nl=" << nl;
    message << " nx=" << nx;
    message << " ny=" << ny;
    message << " nz=" << nz << "\n";
    message << "result: ";
    message << std::fixed << std::setprecision(std::numeric_limits<double>::max_digits10);
    message << sum << "\n";
    message << "Time Elapsed: " << duration.count() / 1e6 << " seconds";

    std::cout << message.str() << std::endl;
}

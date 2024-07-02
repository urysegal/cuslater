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

    // tolerance value for sum (if partial sum value < tol, then value is discarded)
    double tol = 1e-10;

    // α1 = alpha[0], α2 = alpha[1], α3 = alpha[2], α4 = alpha[3]
    float alpha[] = {1, 1, 1, 1};

    // c1 = (c1.x, c1.y, c1.z) = (c[0], c[1], c[2])
    // c2 = (c2.x, c2.y, c2.z) = (c[3], c[4], c[5])
    // c3 = (c3.x, c3.y, c3.z) = (c[6], c[7], c[8])
    // c4 = (c4.x, c4.y, c4.z) = (c[9], c[10], c[11])
    float c[] = {0, 0, 0, 1, 0, 0, 2, 0, 0, 3, 0, 0};

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--help") == 0) {
            // Print help message
            std::cout << "Usage: ./simple [OPTION] [PARAMETERS ...]\n";
            std::cout << "Options:\n";
            std::cout << "  --help\t\tDisplay this help message and exit\n";
            std::cout << "  -a a1 a2 a3 a4\tSet alpha values\n";
            std::cout << "  -t tol\t\tSet tolerance value\n";
            exit(EXIT_SUCCESS); // Exit after printing help message

        } else if (std::strcmp(argv[i], "-a") == 0) {
            // Read next 4 values as alphas
            if (i + 4 >= argc) {
                std::cerr << "Error: Fewer than 4 alpha parameters provided.\n";
                exit(EXIT_FAILURE);
            }
            for (int j = 0; j < 4; ++j) {
                // a_j = i+1, i+2, i+3, i+4
                try {
                    alpha[j] = std::atof(argv[j + 1 + i]);
                } catch (...) {
                    std::cerr << "Error: Insufficient values provided for -a option.\n";
                    exit(EXIT_FAILURE);
                }
            }
            i += 4; // Skip over processed alpha values

        } else if (std::strcmp(argv[i], "-t") == 0) {
            if (i + 1 >= argc) {
                std::cerr << "Error: No tolerance parameter provided.\n";
                exit(EXIT_FAILURE);
            } else {
                tol = std::atof(argv[i + 1]);
                ++i; // Skip over tol value
            }
        }
    }

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

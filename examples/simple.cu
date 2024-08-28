// Created by gkluhana on 26/03/24.
//
// editted by MarkEwert03 on 13/05/24

#include <chrono>
#include <iomanip> // for std::setprecision
#include <iostream>
#include <string>

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
            std::cout << "  -c1 x1 y1 z1\t\tSets coordinates for c1 (c1.x, c1.y, c1.z)\n";
            std::cout << "  -c2 x2 y2 z2\t\tSets coordinates for c2 (c2.x, c2.y, c2.z)\n";
            std::cout << "  -c3 x3 y3 z3\t\tSets coordinates for c3 (c3.x, c3.y, c3.z)\n";
            std::cout << "  -c4 x4 y4 z4\t\tSets coordinates for c4 (c4.x, c4.y, c4.z)\n";
            std::cout << "  -t tol\t\tSet tolerance value\n";
            std::cout << "  -l nl\t\tSet nl\n";
            std::cout << "  -r nr\t\tSet nr\n";
            std::cout << "  -n nx\t\tSet nx, ny, nz\n";
            exit(EXIT_SUCCESS); // Exit after printing help message

        } else if (std::strcmp(argv[i], "-a") == 0) {
            // Read next 4 values as alphas
            if (i + 4 >= argc) {
                std::cerr << "Error: Fewer than 4 alpha parameters provided.\n";
                exit(EXIT_FAILURE);
            }
            for (int j = 1; j <= 4; ++j) {
                // a_j = i+1, i+2, i+3, i+4
                try {
                    alpha[j - 1] = std::atof(argv[i + j]);
                } catch (...) {
                    std::cerr << "Error: Insufficient numerical values provided for -a option.\n";
                    exit(EXIT_FAILURE);
                }
            }
            i += 4; // Skip over processed alpha values

        } else if (argv[i][0] == '-' && argv[i][1] == 'c') {
            int centNum = 0;
            std::string argi = argv[i];
            if (argi.length() != 3) {
                std::cerr << "Error: Please provide 1 digit for the center number.\n";
                exit(EXIT_FAILURE);
            } else if (argi.substr(2, 1) != "1" && argi.substr(2, 1) != "2" &&
                       argi.substr(2, 1) != "3" && argi.substr(2, 1) != "4") {
                std::cerr << "Error: Please provide `-ci` where i = 1,2,3,4.\n";
                exit(EXIT_FAILURE);
            }
            // shift by '0' to convert char->int
            centNum = argi[2] - '0';
            // Read next 3 values coordinates for ci
            if (i + 3 >= argc) {
                std::cerr << "Error: Fewer than 3 coordinates provided for c" << centNum << ".\n";
                exit(EXIT_FAILURE);
            }
            for (int j = 0; j < 3; ++j) {
                // ci_j = i+1, i+2, i+3
                try {
                    c[(centNum - 1) * 3 + j] = std::atof(argv[i + 1 + j]);
                } catch (...) {
                    std::cerr << "Error: Insufficient numerical values provided for -c" << centNum
                              << " option.\n";
                    exit(EXIT_FAILURE);
                }
            }
            i += 3; // Skip over processed ci values

        } else if (std::strcmp(argv[i], "-t") == 0) {
            if (i + 1 >= argc) {
                std::cerr << "Error: No tolerance parameter provided.\n";
                exit(EXIT_FAILURE);
            } else {
                tol = std::atof(argv[i + 1]);
                ++i; // Skip over tol value
            }
        } else if (std::strcmp(argv[i], "-l") == 0) {
            if (i + 1 >= argc) {
                std::cerr << "Error: No nl parameter provided.\n";
                exit(EXIT_FAILURE);
            } else {
                nl = std::atof(argv[i + 1]);
                ++i; // Skip over tol value
            }
        } else if (std::strcmp(argv[i], "-r") == 0) {
            if (i + 1 >= argc) {
                std::cerr << "Error: No nr parameter provided.\n";
                exit(EXIT_FAILURE);
            } else {
                nr = std::atof(argv[i + 1]);
                ++i; // Skip over tol value
            }
        } else if (std::strcmp(argv[i], "-n") == 0) {
            if (i + 1 >= argc) {
                std::cerr << "Error: No nx parameter provided.\n";
                exit(EXIT_FAILURE);
            } else {
                nx = std::atof(argv[i + 1]);
                ny = nx;
                nz = nx;
                ++i; // Skip over tol value
            }
        } else {
            std::cerr << "Error: Invalid command line parameter. Use ./simple --help for more "
                         "information.\n";
            exit(EXIT_FAILURE);
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

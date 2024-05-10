
#include "../include/grids.h"

#include <iostream>
using namespace std;

int
main(int argc, const char *argv[])
{
        double a = -10;
        double b = 11;
        int N = 200;
        std::vector<double> grid;
        std::vector<double> weights;
        cuslater::make_1d_grid_legendre(a,b,N, &grid, &weights);

}

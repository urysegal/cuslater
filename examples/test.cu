#include "../stocalculator.h"

using namespace cuslater;

void calculate(const std::array<STO_Basis_Function, 4> &basis_functions)
{
    Equidistance_1D_Grid x_grid(-10,10,100);
    Equidistance_1D_Grid y_grid(x_grid);
    Equidistance_1D_Grid z_grid(x_grid);
    Logarithmic_1D_Grid s_grid(1000, 50);


    Four_Center_STO_Integrals_Calculator calculator(x_grid, y_grid, z_grid, s_grid);
    calculator.calculate(basis_functions);
}


int
main(int argc, const char *argv[])
{

}

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
    Quantum_Numbers quantum_numbers1 = {1,0,0};
    Quantum_Numbers quantum_numbers2 = {1,0,0};
    Quantum_Numbers quantum_numbers3 = {1,0,0};
    Quantum_Numbers quantum_numbers4 = {1,0,0};

    STO_Basis_Function_Info fi1(1, quantum_numbers1);
    STO_Basis_Function_Info fi2(1, quantum_numbers2);
    STO_Basis_Function_Info fi3(1, quantum_numbers3);
    STO_Basis_Function_Info fi4(1, quantum_numbers4);

    STO_Basis_Function f1(fi1, {0,0,0});
    STO_Basis_Function f2(fi2, {0,0,0});
    STO_Basis_Function f3(fi3, {0,0,0});
    STO_Basis_Function f4(fi4, {0,0,0});
    calculate ({f1, f2, f3, f4});
}

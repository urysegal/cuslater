#pragma once

#include <assert.h>
#include <string>
#include <algorithm>
#include <array>
#include <unordered_map>
#include "include/grids.h"
#include "include/sto.h"
#include "include/tensors.cuh"
#include "include/gputensors.h"

namespace cuslater {

typedef double real_t ;
class Four_Center_STO_Integrals_Calculator {

    MAKE_INDEX(X1)
    MAKE_INDEX(X2)
    MAKE_INDEX(Y1)
    MAKE_INDEX(Y2)
    MAKE_INDEX(Z1)
    MAKE_INDEX(Z2)
    MAKE_INDEX(S)

    General_3D_Grid r_grid;
    Grid_1D &s_grid;

    General_3D_Grid ex_grid;
    General_3D_Grid ey_grid;
    General_3D_Grid ez_grid;

    General_2D_Grid e_slice_grid_x;
    General_2D_Grid e_slice_grid_y;
    General_2D_Grid e_slice_grid_z;


public:
    Four_Center_STO_Integrals_Calculator(Grid_1D &x_grid, Grid_1D &y_grid, Grid_1D &z_grid, Grid_1D &_s_grid) :
        r_grid(x_grid, y_grid, z_grid), s_grid(_s_grid),
        ex_grid(x_grid, x_grid, s_grid), ey_grid(y_grid, y_grid, s_grid), ez_grid(z_grid, z_grid, s_grid),
        e_slice_grid_x(x_grid, x_grid),   e_slice_grid_y(y_grid, y_grid),   e_slice_grid_z(z_grid, z_grid)

    {

    }

    real_t calculate(const std::array<STO_Basis_Function, 4> &basis_functions);

};
}

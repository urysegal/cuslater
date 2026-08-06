#include "number.cuh"
namespace cuslater {
    __constant__ real_t d_c[12];
    __constant__ real_t d_alpha[4];

    __constant__ real_t d_x_grid[500];
    __constant__ real_t d_y_grid[500];
    __constant__ real_t d_z_grid[500];
} // namespace cuslater
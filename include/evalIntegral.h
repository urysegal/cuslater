//
// Created by gkluhana on 26/03/24.
//
// editted by MarkEwert03 on 13/05/24

#include "utilities.h"
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
namespace cuslater {
    double evaluateFourCenterIntegral(real_t* c, real_t* alphas, int nr, int nl, int n, double tol,
                                      bool check_zero_cond);
    double evaluateFourCenterIntegral(real_t* c, real_t* alphas, int nr, int nl, int nx, int ny,
                                      int nz, const std::string x1_type, int num_gpus);

} // namespace cuslater

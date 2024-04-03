//
// Created by gkluhana on 26/03/24.
//
#include <vector>
#include "cuslater.cuh"
#include "utilities.h"
#include "grids.h"
#include "evalInnerIntegral.h"
namespace cuslater{
    double evaluateFourCenterIntegral( double* c,
                                int nr,  int nl,  int nx,
                                const std::string x1_type);
    double evaluateFourCenterIntegral( double* c,
                                  int nr,  int nl,  int nx,
                                  const std::string x1_type,
                                  int num_gpus);

}

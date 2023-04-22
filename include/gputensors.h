#pragma once
#include <vector>
#include <unordered_map>

#include "tensors.h"

namespace cuslater {
int hadamar(std::vector<int> &modes, std::unordered_map<int, int64_t> &extent, const real_t *A, const real_t *C,
            real_t *D);
}
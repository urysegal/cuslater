
#include "../include/evalIntegral.h"
#include "cuslater.cuh"
#include "grids.h"
#include <cassert>
#include <chrono>
#include <iomanip> // for std::setprecision
#include <iostream>
#include <vector>
using namespace std;

#define F3(x, y, z) (make_real3(x, y, z))

// -c1 0 0 0 -c2 1 0 0 -c3 2 0 0 -c4 3 0 0
// -c1 1 0 0 -c2 0 1 0 -c3 0 0 1 -c4 1 1 1
// -c1 -1 0 0 -c2 0 1 0 -c3 0 0 -1 -c4 1 -1 1
// -c1 -1 0 1 -c2 2 -1 1 -c3 0 2 -1 -c4 1 1 1
// -c1 1 2 3 -c2 -2 1 3 -c3 3 -2 1 -c4 2 3 -1
// -c1 0 0 0 -c2 1 0 0 -c3 2 0 0 -c4 3 0 0
// -c1 1 0 0 -c2 0 1 0 -c3 0 0 1 -c4 1 1 1
// -c1 -1 0 0 -c2 0 1 0 -c3 0 0 -1 -c4 1 -1 1
// -c1 -1 0 1 -c2 2 -1 1 -c3 0 2 -1 -c4 1 1 1
// -c1 1 2 3 -c2 -2 1 3 -c3 3 -2 1 -c4 2 3 -1
// newly added random points
vector<std::array<float3, 4>> centers = {
    {F3(0, 0, 0), F3(1, 0, 0), F3(2, 0, 0), F3(3, 0, 0)},
    {F3(1, 0, 0), F3(0, 1, 0), F3(0, 0, 1), F3(1, 1, 1)},
    {F3(-1, 0, 0), F3(0, 1, 0), F3(0, 0, -1), F3(1, -1, 1)},
    {F3(-1, 0, 1), F3(2, -1, 1), F3(0, 2, -1), F3(1, 1, 1)},
    {F3(1, 2, 3), F3(-2, 1, 3), F3(3, -2, 1), F3(2, 3, -1)},

    {F3(0, 0, 0), F3(1, 0, 0), F3(2, 0, 0), F3(3, 0, 0)},
    {F3(1, 0, 0), F3(0, 1, 0), F3(0, 0, 1), F3(1, 1, 1)},
    {F3(-1, 0, 0), F3(0, 1, 0), F3(0, 0, -1), F3(1, -1, 1)},
    {F3(-1, 0, 1), F3(2, -1, 1), F3(0, 2, -1), F3(1, 1, 1)},
    {F3(1, 2, 3), F3(-2, 1, 3), F3(3, -2, 1), F3(2, 3, -1)},
    // newly added randome points
    {F3(0.5, 1.5, -0.6), F3(0.6, 1.4, -0.5), F3(0.4, 1.6, -0.4), F3(0.5, 1.4, -0.4)},
    {F3(-1.1, 2.5, 0.3), F3(-1.9, 1.7, 0.6), F3(-1.4, 2.2, 0.0), F3(-1.6, 2.1, 0.9)},
    {F3(2.7, -0.2, 2.0), F3(1.3, -0.8, 2.2), F3(2.4, -1.1, 3.3), F3(2.0, 0.1, 2.8)},
    {F3(-1.5, -1.7, -0.2), F3(-3.6, -2.3, -1.3), F3(-2.2, -3.6, -2.1), F3(-2.9, -2.0, 0.1)},
    {F3(3.0, 0.8, -1.7), F3(3.5, 1.3, -1.3), F3(3.3, 0.7, -1.2), F3(3.1, 1.2, -1.8)},
    {F3(-1.3, 2.4, 2.3), F3(0.3, 3.5, 3.4), F3(-0.8, 3.7, 2.6), F3(-0.2, 2.8, 3.6)},
    {F3(1.5, -1.2, 0.9), F3(-1.4, 1.6, -0.8), F3(1.7, 0.8, -1.3), F3(-0.9, -1.5, 1.2)},
    {F3(2.4, 2.5, 2.3), F3(1.5, 2.1, 1.7), F3(2.2, 1.6, 2.4), F3(1.8, 2.3, 1.5)},
    {F3(-3.1, 0.6, -2.1), F3(-2.9, 0.4, -2.0), F3(-3.0, 0.5, -1.8), F3(-3.2, 0.6, -2.0)},
    {F3(1.4, -2.2, 1.9), F3(-0.2, -2.7, 0.7), F3(0.7, -3.8, 1.0), F3(-0.4, -3.2, 2.3)},

    {F3(0.5, 1.5, -0.6), F3(0.6, 1.4, -0.5), F3(0.4, 1.6, -0.4), F3(0.5, 1.4, -0.4)},
    {F3(-1.1, 2.5, 0.3), F3(-1.9, 1.7, 0.6), F3(-1.4, 2.2, 0.0), F3(-1.6, 2.1, 0.9)},
    {F3(2.7, -0.2, 2.0), F3(1.3, -0.8, 2.2), F3(2.4, -1.1, 3.3), F3(2.0, 0.1, 2.8)},
    {F3(-1.5, -1.7, -0.2), F3(-3.6, -2.3, -1.3), F3(-2.2, -3.6, -2.1), F3(-2.9, -2.0, 0.1)},
    {F3(3.0, 0.8, -1.7), F3(3.5, 1.3, -1.3), F3(3.3, 0.7, -1.2), F3(3.1, 1.2, -1.8)},
    {F3(-1.3, 2.4, 2.3), F3(0.3, 3.5, 3.4), F3(-0.8, 3.7, 2.6), F3(-0.2, 2.8, 3.6)},
    {F3(1.5, -1.2, 0.9), F3(-1.4, 1.6, -0.8), F3(1.7, 0.8, -1.3), F3(-0.9, -1.5, 1.2)},
    {F3(2.4, 2.5, 2.3), F3(1.5, 2.1, 1.7), F3(2.2, 1.6, 2.4), F3(1.8, 2.3, 1.5)},
    {F3(-3.1, 0.6, -2.1), F3(-2.9, 0.4, -2.0), F3(-3.0, 0.5, -1.8), F3(-3.2, 0.6, -2.0)},
    {F3(1.4, -2.2, 1.9), F3(-0.2, -2.7, 0.7), F3(0.7, -3.8, 1.0), F3(-0.4, -3.2, 2.3)},
};

// -a 1 1 2 2
// -a 1 2 3 4
// -a 1.2 2.3 3.2 3.1
// -a 4 3 2 1
// -a 2 1 2 1
// -a 1 1 1 1
// -a 1 1 1 1
// -a 1 1 1 1
// -a 1 1 1 1
// -a 1 1 1 1
// -a 2 1 1.5 0.5
// -a 3 2 1 2.5
// -a 1.5 0.5 3 3
// -a 1 1.5 2 3.5
// -a 2 1 0.5 3
// -a 2.5 3.5 1 0.5
// -a 3 2 1.5 1
// -a 0.5 2 3 0.5
// -a 1.5 3 2.5 1
// -a 2.5 1 2 3.5
vector<std::array<real_t, 4>> alphas = {
    {1, 1, 2, 2},       {1, 2, 3, 4},   {1.2, 2.3, 3.2, 3.1}, {4, 3, 2, 1},     {2, 1, 2, 1},
    {1, 1, 1, 1},       {1, 1, 1, 1},   {1, 1, 1, 1},         {1, 1, 1, 1},     {1, 1, 1, 1},
    {2, 1, 1.5, 0.5},   {3, 2, 1, 2.5}, {1.5, 0.5, 3, 3},     {1, 1.5, 2, 3.5}, {2, 1, 0.5, 3},
    {2.5, 3.5, 1, 0.5}, {3, 2, 1.5, 1}, {0.5, 2, 3, 0.5},     {1.5, 3, 2.5, 1}, {2.5, 1, 2, 3.5},
    {1, 1, 1, 1},       {1, 1, 1, 1},   {1, 1, 1, 1},         {1, 1, 1, 1},     {1, 1, 1, 1},
    {1, 1, 1, 1},       {1, 1, 1, 1},   {1, 1, 1, 1},         {1, 1, 1, 1},     {1, 1, 1, 1},
};

vector<int>   r_nodes = {2, 16, 39, 48, 56, 63, 69, 74, 79, 84, 89};
vector<int>   l_nodes = {146, 170, 194, 230, 266, 302, 350, 434, 590, 770};
vector<int>   n_nodes = {5,  10, 15, 20,  25,  30,  35,  40,  45,  50,  55,  60,  65,  70,  75, 80,
                         85, 90, 95, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375};
vector<float> tols    = {1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10};

real_t c[12]    = {0, 0, 0, 1, 0, 0, 2, 0, 0, 3, 0, 0};
real_t alpha[4] = {1, 1, 1, 1};

void test_A() {

    vector<real2_t> r = cuslater::read_r_grid(39);
    vector<real4_t> l = cuslater::read_l_grid(302);

    vector<double>           results = {};
    vector<cuslater::Metric> metrics(centers.size());
    for (size_t i = 0; i < centers.size(); ++i) {
        real_t c[12];
        for (int j = 0; j < 4; ++j) {
            c[j * 3 + 0] = centers[i][j].x;
            c[j * 3 + 1] = centers[i][j].y;
            c[j * 3 + 2] = centers[i][j].z;
        }
        real_t* alpha = alphas[i].data();
        double sum = cuslater::evaluateFourCenterIntegral(c, alpha, r, l, 200, 1e-8, true, &metrics[i]);
        cout << "Result for center " << i << ": " << sum << endl;
        results.push_back(sum);
    }
    for (size_t i = 0; i < results.size(); ++i) {
        cout << results[i] << "\truntime: " << metrics[i].totalTime.count() / 1e6 << " seconds"
             << std::endl;
    }
}

void test_N() {
    vector<real2_t> r = cuslater::read_r_grid(39);
    vector<real4_t> l = cuslater::read_l_grid(302);

    vector<real_t>           results = {};
    vector<cuslater::Metric> metrics(n_nodes.size());
    for (size_t i = 0; i < n_nodes.size(); ++i) {
        real_t sum = cuslater::evaluateFourCenterIntegral(c, alpha, r, l, n_nodes[i], 1e-8, true,
                                                          &metrics[i]);
        cout << "Result for N size " << n_nodes[i] << ": " << sum << endl;
        results.push_back(sum);
    }
    for (size_t i = 0; i < results.size(); ++i) {
        cout << "N size: " << (n_nodes[i]) << ": " << results[i]
             << "\truntime: " << metrics[i].totalTime.count() / 1e6 << " seconds" << std::endl;
    }
}

void test_L() {
    vector<real2_t> r = cuslater::read_r_grid(39);

    vector<real_t>           results = {};
    vector<cuslater::Metric> metrics(l_nodes.size());
    for (size_t i = 0; i < l_nodes.size(); ++i) {
        vector<real4_t> l = cuslater::read_l_grid(l_nodes[i]);
        double sum = cuslater::evaluateFourCenterIntegral(c, alpha, r, l, 200, 1e-8, true, &metrics[i]);
        cout << "Result for L node " << l_nodes[i] << ": " << sum << endl;
        results.push_back(sum);
    }
    for (size_t i = 0; i < results.size(); ++i) {
        cout << "l = " << l_nodes[i] << ": " << results[i]
             << "\truntime: " << metrics[i].totalTime.count() / 1e6 << " seconds" << std::endl;
    }
}

void test_R() {
    vector<real4_t> l = cuslater::read_l_grid(302);

    vector<real_t>           results = {};
    vector<cuslater::Metric> metrics(r_nodes.size());
    for (size_t i = 0; i < r_nodes.size(); ++i) {
        vector<real2_t> r = cuslater::read_r_grid(r_nodes[i]);
        double sum = cuslater::evaluateFourCenterIntegral(c, alpha, r, l, 200, 1e-8, true, &metrics[i]);
        cout << "Result for R node " << r_nodes[i] << ": " << sum << endl;
        results.push_back(sum);
    }
    for (size_t i = 0; i < results.size(); ++i) {
        cout << "r = " << r_nodes[i] << ": " << results[i]
             << " \truntime: " << metrics[i].totalTime.count() / 1e6 << " seconds" << std::endl;
    }
}

void test_Tol() {
    vector<real2_t> r = cuslater::read_r_grid(39);
    vector<real4_t> l = cuslater::read_l_grid(302);

    vector<real_t>           results = {};
    vector<cuslater::Metric> metrics(tols.size());
    for (size_t i = 0; i < tols.size(); ++i) {
        double tol = tols[i];
        double sum = cuslater::evaluateFourCenterIntegral(c, alpha, r, l, 200, tol, true, &metrics[i]);
        cout << "Result for tol " << tol << ": " << sum << endl;
        results.push_back(sum);
    }
    for (size_t i = 0; i < results.size(); ++i) {
        int total = metrics[i].totalKernelCalls + metrics[i].skippedLebdevNodes;
        cout << "tol = 1e-" << (i + 3) << ": " << results[i]
             << "\truntime: " << metrics[i].totalTime.count() / 1e6 << " seconds"
             << " skipped: " << metrics[i].skippedLebdevNodes << "/" << total << std::endl;
    }
}

int main(int argc, const char* argv[]) {

    HANDLE_CUDA_ERROR(cudaSetDevice(0));

    std::cout << std::fixed << std::setprecision(std::numeric_limits<double>::max_digits10);
    assert(centers.size() == alphas.size());

    cout << "Testing Cuslater Integral Evaluation" << endl;
    cout << "========================================" << endl;
    cout << "Testing Centers/Alphas" << endl;
    cout << "Number of Centers/Alphas: " << centers.size() << endl;
    cout << "----------------------------------------" << endl;
    test_A();
    cout << "----------------------------------------" << endl;
    cout << "Testing N Nodes" << endl;
    cout << "Number of N Nodes: " << n_nodes.size() << endl;
    cout << "----------------------------------------" << endl;
    test_N();
    cout << "----------------------------------------" << endl;
    cout << "Testing L Nodes" << endl;
    cout << "Number of L Nodes: " << l_nodes.size() << endl;
    cout << "----------------------------------------" << endl;
    test_L();
    cout << "----------------------------------------" << endl;
    cout << "Testing R Nodes" << endl;
    cout << "Number of R Nodes: " << r_nodes.size() << endl;
    cout << "----------------------------------------" << endl;
    test_R();
    cout << "----------------------------------------" << endl;
    cout << "Testing Tolerance" << endl;
    cout << "Number of Tolerances: " << tols.size() << endl;
    cout << "----------------------------------------" << endl;
    test_Tol();
}

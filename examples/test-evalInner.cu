#include "../include/evalInnerIntegral.h"
#include <iostream>
#include <chrono>

using namespace std;

int
main(int argc, const char *argv[])
{

	double c[] = {0,0,0,
                    1,0,0,
                    2,0,0,
                    3,0,0};
	double r = 1.0;
        double xrange[] = {-5,6};
        double yrange[] = {-5,6};
        double zrange[] = {-5,6};
        unsigned int x_axis_points = 501;
        unsigned int y_axis_points = 501;
        unsigned int z_axis_points = 501;



        std::cout << "num_grids: 1" <<std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        double w[] = {0.1,0.2,0.3};
        double* d_results = nullptr;
        double** d_results_ptr = &d_results;
        assert(d_results==nullptr);
        auto sum0 = cuslater::evaluateInnerSumX1_rl( c,r, w, xrange,  yrange, zrange,
                                             x_axis_points, y_axis_points, z_axis_points,
                                             d_results_ptr,0);
        std::cout << "nx,ny,nz: " << x_axis_points << std::endl;
        std::cout << "Sum from evaluateInner: " << sum0 << std::endl;
        d_results = *d_results_ptr;
        sum0 = cuslater::evaluateInnerSumX1_rl( c,r, w, xrange,  yrange, zrange,
                                        x_axis_points, y_axis_points, z_axis_points,
                                        d_results_ptr,0);
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "nx,ny,nz: " << x_axis_points << std::endl;
        std::cout << "Execution time Sum: " << duration.count() << " microseconds" << std::endl;
        std::cout << "Sum from evaluateInner: " << sum0 << std::endl;
	return 0;
}

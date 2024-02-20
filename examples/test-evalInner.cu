#include "../include/evalInnerIntegral.h"
<<<<<<< HEAD
#include <iostream>
#include <chrono>
=======

>>>>>>> f54066d (fix and test file)
using namespace std;

#define N (100)

int
main(int argc, const char *argv[])
{
<<<<<<< HEAD

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
        auto sum0 = cuslater::evaluateInner( c,r, w, xrange,  yrange, zrange,
                                             x_axis_points, y_axis_points, z_axis_points,
                                             d_results_ptr,0);
        std::cout << "nx,ny,nz: " << x_axis_points << std::endl;
        std::cout << "Sum from evaluateInner: " << sum0 << std::endl;
        d_results = *d_results_ptr;
        sum0 = cuslater::evaluateInner( c,r, w, xrange,  yrange, zrange,
                                        x_axis_points, y_axis_points, z_axis_points,
                                        d_results_ptr,0);
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "nx,ny,nz: " << x_axis_points << std::endl;
        std::cout << "Execution time Sum: " << duration.count() << " microseconds" << std::endl;
        std::cout << "Sum from evaluateInner: " << sum0 << std::endl;
//
//        int Nl = 500 ;
//	int ws_length = Nl*3;
//	double ws[ws_length];
//	double w_wts[Nl];
//	for (int i = 0; i< Nl; ++i){
//		w_wts[i] = 1.0;
//		ws[i*3] = 0.1;
//		ws[i*3+1] = 0.2;
//		ws[i*3+2] = 0.3;
//	}
//
//	std::cout << "num_grids: " << Nl <<std::endl;
//
//        auto sum = cuslater::evaluateInnerStreams( c,r,
//                                                   ws,w_wts, Nl,
//                                                   xrange,  yrange, zrange,
//                                                   x_axis_points, y_axis_points, z_axis_points,
//                                                    0);
//
//        std::cout << "nx,ny,nz: " << x_axis_points << std::endl;
//        std::cout << "Execution time Sum1: " << duration.count() << " microseconds" << std::endl;
//        std::cout << "Sum from evaluateInnerStreams: " << sum << std::endl;
//
//
//
//        auto start2 = std::chrono::high_resolution_clock::now();
//	auto sum2 = cuslater::evaluateInnerStreams( c,r,
//                                                   ws,w_wts, Nl,
//                                                   xrange,  yrange, zrange,
//                                                   x_axis_points, y_axis_points, z_axis_points,
//                                                    1);
//
//	auto end2 = std::chrono::high_resolution_clock::now();
//	auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2);
//        std::cout << "nx,ny,nz: " << x_axis_points << std::endl;
//        std::cout << "Sum2 from evaluateInnerStreams: " << sum << std::endl;

        //d_results = nullptr;
	//Nl = num_grids;
        //sum = cuslater::evaluateInnerStreams( c,r,
        //                                           ws,w_wts, Nl,
        //                                           xrange,  yrange, zrange,
        //                                           x_axis_points, y_axis_points, z_axis_points,
        //                                           d_results, 0);


        //std::cout << "nx,ny,nz: " << x_axis_points << std::endl;
        //std::cout << "Sum from evaluateInnerStreams: " << sum << std::endl;
//	x_axis_points = 69;
//	y_axis_points = 69;
//	z_axis_points = 69;
//	sum = cuslater::evaluateInner( d_c,r, w, xrange,  yrange, zrange, x_axis_points, y_axis_points, z_axis_points, result_array,0);
//	std::cout << "nx,ny,nz: " << x_axis_points << std::endl;
//	std::cout << "Sum from evaluateInner: " << sum << std::endl;
////
//	x_axis_points = 450;
//	y_axis_points = 450;
//	z_axis_points = 450;
//	sum = cuslater::evaluateInner(d_c, r, w, xrange,  yrange, zrange, x_axis_points, y_axis_points, z_axis_points, result_array,0);
//	std::cout << "nx,ny,nz: " << x_axis_points << std::endl;
//	std::cout << "Sum from evaluateInner: " << sum << std::endl;
	return 0;
=======
	double c1[] = {0,0,0};
	double c2[] = {1,0,0};
	double c3[] = {2,0,0};
	double c4[] = {3,0,0};
	double r = 1.0;
	double w[] = {0.1,0.2,0.3};
	double xrange[] = {-5,10};
	double yrange[] = {-5,10};
	double zrange[] = {-5,10};
	unsigned int x_axis_points = 1000;
	unsigned int y_axis_points = 1000;
	unsigned int z_axis_points = 1000;
	double *result_array=NULL;
	auto sum = cuslater::evaluateInner( c1, c2, c3,c4, r, w, xrange,  yrange, zrange, x_axis_points, y_axis_points, z_axis_points, result_array);

>>>>>>> f54066d (fix and test file)
}

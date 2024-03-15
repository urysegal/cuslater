#include "../include/evalInnerIntegral.h"
#include <iostream>
using namespace std;

#define N (100)

int
main(int argc, const char *argv[])
{
	double c[] = {0,0,0,
                    1,0,0,
                    2,0,0,
                    3,0,0};
	double r = 1.0;
	double w[] = {0.1,0.2,0.3};
	double xrange[] = {-5,6};
	double yrange[] = {-5,6};
	double zrange[] = {-5,6};
	unsigned int x_axis_points = 501;
	unsigned int y_axis_points = 501;
	unsigned int z_axis_points = 501;
	double *result_array=NULL;
//	auto d_c = cuslater::preProcessIntegral( c);
	auto sum = cuslater::evaluateInner( c,r, w, xrange,  yrange, zrange, x_axis_points, y_axis_points, z_axis_points, result_array,0);
	std::cout << "nx,ny,nz: " << x_axis_points << std::endl;
	std::cout << "Sum from evaluateInner: " << sum << std::endl;

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
}

#include "../include/evalInnerIntegral.h"

using namespace std;

#define N (100)

int
main(int argc, const char *argv[])
{
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

}

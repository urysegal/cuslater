#include "../include/gputensors.h"

using namespace std;

#define N (100)

int
main(int argc, const char *argv[])
{
    std::vector<int> modes = { 'x', 'y', 'z'};
    std::unordered_map<int, int64_t> extent;
    extent['x'] = N;
    extent['y'] = N;
    extent['z'] = N;
    double *A;
    double *C;
    double *D;

    A = new double [ N*N*N];
    C = new double [ N*N*N];
    D = new double [ N*N*N];

    for ( int i = 0U ; i < N ; ++i ) {
        for (int j = 0U; j < N; ++j) {
            for (int k = 0U; k < N; ++k) {
                A[i*N*N+N*j+k]=2;
                C[i*N*N+N*j+k]=3;
                D[i*N*N+N*j+k]=0;
            }
        }
    }

    cuslater::hadamard(modes, extent, (const double *) A, (const double *) C, (double *) D);

    delete A;
    delete C;
    delete D;
    return 0;
}

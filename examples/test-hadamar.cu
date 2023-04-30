#include "../include/gputensors.h"

using namespace std;

#define N (3)

int
main(int argc, const char *argv[])
{
    std::vector<int> modes = { 'x', 'y', 'z'};
    std::unordered_map<int, int64_t> extent;
    extent['x'] = N;
    extent['y'] = N;
    extent['z'] = N;
    double A[N][N][N];
    double C[N][N][N];
    double D[N][N][N];

    for ( int i = 0U ; i < N ; ++i ) {
        for (int j = 0U; j < N; ++j) {
            for (int k = 0U; k < N; ++k) {
                A[i][j][k]=2;
                C[i][j][k]=3;
                D[i][j][k]=0;
            }
        }
    }

    cuslater::hadamar(modes, extent, (const double *)A, (const double *)C, (double *)D);

    return 0;
}

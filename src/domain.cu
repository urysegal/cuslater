#include "evalIntegral.cuh"
namespace cuslater {
    Domain::Domain(std::span<real_t, 12> c, int n) {
        real_t dx = std::abs(c[0] - c[3]);
        real_t dy = std::abs(c[1] - c[4]);
        real_t dz = std::abs(c[2] - c[5]);
        real_t lx = 18.0 + dx;
        real_t ly = 18.0 + dy;
        real_t lz = 18.0 + dz;
        real_t mx = (c[0] + c[3]) / 2.0;
        real_t my = (c[1] + c[4]) / 2.0;
        real_t mz = (c[2] + c[5]) / 2.0;
        real_t ax = mx - (lx / 2.0);
        real_t bx = mx + (lx / 2.0);
        real_t ay = my - (ly / 2.0);
        real_t by = my + (ly / 2.0);
        real_t az = mz - (lz / 2.0);
        real_t bz = mz + (lz / 2.0);

        real_t hx = (bx - ax) / (n - 1);
        real_t hy = (by - ay) / (n - 1);
        real_t hz = (bz - az) / (n - 1);

        this->ax = ax;
        this->bx = bx;
        this->ay = ay;
        this->by = by;
        this->az = az;
        this->bz = bz;
        this->hx = hx;
        this->hy = hy;
        this->hz = hz;
    }
    // using namespace std;
    // ostream& operator<<(ostream& os, const Domain& m) {

    //     os << "   Legendre Grid Parameters: \n"
    //        << "   xgrid (a.x , b.x) : (" << m.ax << " , " << m.bx << ")\n"
    //        << "   ygrid (a.y , b.y) : (" << m.ay << " , " << m.by << ")\n"
    //        << "   zgrid (a.z , b.z) : (" << m.az << " , " << m.bz << ")\n";
    //     return os;
    // }
} // namespace cuslater
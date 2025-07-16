#include "evalIntegral.h"
#include "iomanip"
#include <sstream>

namespace cuslater {
    using namespace std;
    ostream& operator<<(ostream& os, const Metric& m) {
        int                total = m.totalKernelCalls + m.skippedLebdevNodes;
        std::ostringstream skipped;
        skipped << std::fixed << std::setprecision(2)
                << (m.skippedLebdevNodes * 100.0 / total);
        os << "Cuslater Metric Report:\n";
        os << "  Total Threads: " << m.totalThreads << "\n"
           << "  Total Blocks: " << m.totalBlocks << "\n"
           << "  Total Grid Points: " << m.totalGridPoints << "\n"
           << "  Total Time: " << m.totalTime.count() / 1e6 << " seconds\n"
           << "  Total Kernel Calls: " << m.totalKernelCalls << "\n"
           << "  Total Kernel Time: " << m.totalKernelTime.count() / 1e6 << " seconds\n"
           << "  Average Kernel Time: " << m.avgKernelTime.count() / 1e6 << " seconds\n"
           << "  Skipped Lebedev Nodes: " << m.skippedLebdevNodes << "/" << total << " ("
           << skipped.str() << "%)\n"
           << "  Effective Bandwidth: " << m.effectiveBandwidth << " GB/s\n"
           << "   Legendre Grid Parameters: \n"
           << "   xgrid (a.x , b.x) : (" << m.a.x << " , " << m.b.x << ")\n"
           << "   ygrid (a.y , b.y) : (" << m.a.y << " , " << m.b.y << ")\n"
           << "   zgrid (a.z , b.z) : (" << m.a.z << " , " << m.b.z << ")\n";
        return os;
    }
} // namespace cuslater

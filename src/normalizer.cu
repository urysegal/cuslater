#include "number.cuh"
#include <array>
#include <iomanip>
#include <iostream>
#include <numbers>
namespace cuslater {
    using namespace std;

    auto orbital_label_sp(int i) -> std::string {
        switch (i) {
            case 0:
                return "1x"; // 1s, px
            case 1:
                return "1y"; // 1s, py
            case 2:
                return "1z"; // 1s, pz
            case 3:
                return "2x"; // 2s, px
            case 4:
                return "2y"; // 2s, py
            case 5:
                return "2z"; // 2s, pz
            default:
                return "??";
        }
    }

    auto orbital_label_ps(int i) -> std::string {
        switch (i) {
            case 0:
                return "x1"; // px, 1s
            case 1:
                return "y1"; // py, 1s
            case 2:
                return "z1"; // pz, 1s
            case 3:
                return "x2"; // px, 2s
            case 4:
                return "y2"; // py, 2s
            case 5:
                return "z2"; // pz, 2s
            default:
                return "??";
        }
    }
    auto orbital_label_p(int i) -> std::string {
        switch (i) {
            case 0:
                return "1x";
            case 1:
                return "1y";
            case 2:
                return "1z";
            case 3:
                return "2x";
            case 4:
                return "2y";
            case 5:
                return "2z";
            default:
                return "??";
        }
    }
    auto orbital_label_s(int i) -> std::string {
        switch (i) {
            case 0:
                return "11";
            case 1:
                return "21";
            case 2:
                return "12";
            case 3:
                return "22";
            default:
                return "??";
        }
    }

    auto ps_pattern(int pair_idx) -> std::tuple<int, int, int> {
        switch (pair_idx) {
            case 0:
                return {1, 0, 0}; // px, 1s
            case 1:
                return {1, 0, 1}; // py, 1s
            case 2:
                return {1, 0, 2}; // pz, 1s
            case 3:
                return {1, 1, 0}; // px, 2s
            case 4:
                return {1, 1, 1}; // py, 2s
            case 5:
                return {1, 1, 2}; // pz, 2s
            default:
                return {1, 0, 0};
        }
    }

    auto sp_pattern(int pair_idx) -> std::tuple<int, int, int> {
        switch (pair_idx) {
            case 0:
                return {0, 1, 0}; // 1s, px
            case 1:
                return {0, 1, 1}; // 1s, py
            case 2:
                return {0, 1, 2}; // 1s, pz
            case 3:
                return {1, 1, 0}; // 2s, px
            case 4:
                return {1, 1, 1}; // 2s, py
            case 5:
                return {1, 1, 2}; // 2s, pz
            default:
                return {0, 1, 0};
        }
    }

    auto sp_left_pattern(int i) -> std::pair<int, int> {
        switch (i) {
            case 0:
                return {0, 0}; // x1  => (orb1=1s, orb2=px)
            case 1:
                return {0, 1}; // y1
            case 2:
                return {0, 2}; // z1
            case 3:
                return {1, 0}; // x2  => (orb1=2s, orb2=px)
            case 4:
                return {1, 1}; // y2
            case 5:
                return {1, 2}; // z2
            default:
                return {0, 0};
        }
    }

    auto is2s_pattern = [](int pair_idx) -> std::array<int, 2> {
        switch (pair_idx) {
            case 0:
                return {0, 0}; // 1s,1s
            case 1:
                return {1, 0}; // 2s,1s
            case 2:
                return {0, 1}; // 1s,2s
            case 3:
                return {1, 1}; // 2s,2s
            default:
                return {0, 0};
        }
    };

    void print_results(const std::vector<double>& sums) {
        // SS|SS
        for (int idx = 0; idx < 16 && idx < sums.size(); ++idx) {
            int li = idx / 4;
            int ri = idx % 4;
            printf("[%3d] %s|%s : %.10f\n", idx, orbital_label_s(li).c_str(),
                   orbital_label_s(ri).c_str(), sums[idx]);
        }

        // SS|SP
        for (int idx = 16; idx < 40 && idx < sums.size(); ++idx) {
            int rel = idx - 16;
            int li  = rel / 6;
            int ri  = rel % 6;
            printf("[%3d] %s|%s : %.10f\n", idx, orbital_label_s(li).c_str(),
                   orbital_label_sp(ri).c_str(), sums[idx]);
        }

        // SS|PS
        for (int idx = 40; idx < 64 && idx < sums.size(); ++idx) {
            int rel = idx - 40;
            int li  = rel / 6;
            int ri  = rel % 6;
            printf("[%3d] %s|%s : %.10f\n", idx, orbital_label_s(li).c_str(),
                   orbital_label_ps(ri).c_str(), sums[idx]);
        }

        // SP|SS
        for (int idx = 64; idx < 88 && idx < sums.size(); ++idx) {
            int rel = idx - 64;
            int li  = rel / 4;
            int ri  = rel % 4;
            printf("[%3d] %s|%s : %.10f\n", idx, orbital_label_sp(li).c_str(),
                   orbital_label_s(ri).c_str(), sums[idx]);
        }

        // PS|SS, only if this block exists in some other kernel.
        for (int idx = 88; idx < 112 && idx < sums.size(); ++idx) {
            int rel = idx - 88;
            int li  = rel / 4;
            int ri  = rel % 4;
            printf("[%3d] %s|%s : %.10f\n", idx, orbital_label_ps(li).c_str(),
                   orbital_label_s(ri).c_str(), sums[idx]);
        }
    }

    void normalize_sums(std::vector<double>& sums, const std::array<double, 4>& a) {

        constexpr double factor = 4.0 / std::numbers::pi;

        // Block 1: SS|SS  -> idx 0..15
        for (int idx = 0; idx < 16 && idx < sums.size(); ++idx) {
            const int li = idx / 4;
            const int ri = idx % 4;

            const auto left  = is2s_pattern(li); // orbitals 1,2
            const auto right = is2s_pattern(ri); // orbitals 3,4

            const int b1 = left[0];
            const int b2 = left[1];
            const int b3 = right[0];
            const int b4 = right[1];

            const int total_2s = b1 + b2 + b3 + b4;

            const double norm = factor * std::pow(a[0], 1.5 + b1) * std::pow(a[1], 1.5 + b2)
                              * std::pow(a[2], 1.5 + b3) * std::pow(a[3], 1.5 + b4)
                              / std::pow(3.0, 0.5 * total_2s);

            sums[idx] *= norm;
        }

        // Block 2: SS|SP -> idx 16..39
        for (int idx = 16; idx < 40 && idx < sums.size(); ++idx) {
            const int rel = idx - 16;

            const int li = rel / 6; // left SS index: 0..3
            const int ri = rel % 6; // right SP index: 0..5

            const auto left  = is2s_pattern(li); // orbitals 1,2
            const auto right = sp_pattern(ri);   // orbital 3 is s/2s, orbital 4 is p

            const int b1 = left[0];
            const int b2 = left[1];

            const int b3 = std::get<0>(right); // orbital 3: 0->1s, 1->2s
            const int p4 = std::get<1>(right); // orbital 4: always p (=1)

            const int total_2s = b1 + b2 + b3;

            const double norm = factor * std::pow(a[0], 1.5 + b1) * std::pow(a[1], 1.5 + b2)
                              * std::pow(a[2], 1.5 + b3) * std::pow(a[3], 2.5)
                              / std::pow(3.0, 0.5 * total_2s);

            sums[idx] *= norm;
        }

        // Block 3: SS|PS -> idx 40..63
        for (int idx = 40; idx < 64 && idx < sums.size(); ++idx) {
            const int rel = idx - 40;

            const int li = rel / 6; // left SS index: 0..3
            const int ri = rel % 6; // right PS index: 0..5

            const auto left  = is2s_pattern(li); // orbitals 1,2
            const auto right = ps_pattern(ri);   // orbital 3 is p, orbital 4 is s/2s

            const int b1 = left[0];
            const int b2 = left[1];

            const int p3 = std::get<0>(right); // orbital 3: always p (=1)
            const int b4 = std::get<1>(right); // orbital 4: 0->1s, 1->2s

            const int total_2s = b1 + b2 + b4;

            const double norm = factor * std::pow(a[0], 1.5 + b1) * std::pow(a[1], 1.5 + b2)
                              * std::pow(a[2], 2.5) * std::pow(a[3], 1.5 + b4)
                              / std::pow(3.0, 0.5 * total_2s);

            sums[idx] *= norm;
        }
        for (int idx = 64; idx < 88 && idx < sums.size(); ++idx) {
            const int rel = idx - 64;

            const int li = rel / 4; // left SP index: 0..5
            const int ri = rel % 4; // right SS index: 0..3

            const auto left  = sp_left_pattern(li);
            const auto right = is2s_pattern(ri);

            const int b1 = left.first; // orbital 1: 0->1s, 1->2s
            const int b3 = right[0];
            const int b4 = right[1];

            const int total_2s = b1 + b3 + b4;

            const double norm = factor * std::pow(a[0], 1.5 + b1) * std::pow(a[1], 2.5)
                              * std::pow(a[2], 1.5 + b3) * std::pow(a[3], 1.5 + b4)
                              / std::pow(3.0, 0.5 * total_2s);

            sums[idx] *= norm;
        }
        for (int idx = 88; idx < 112 && idx < sums.size(); ++idx) {
            const int rel = idx - 88;

            const int li = rel / 4; // left PS index: 0..5
            const int ri = rel % 4; // right SS index: 0..3

            const auto left  = ps_pattern(li);   // orbital 1 is p, orbital 2 is s/2s
            const auto right = is2s_pattern(ri); // orbitals 3,4

            const int b2 = std::get<1>(left); // orbital 2: 0->1s, 1->2s
            const int b3 = right[0];
            const int b4 = right[1];

            const int total_2s = b2 + b3 + b4;

            const double norm = factor * std::pow(a[0], 2.5) * std::pow(a[1], 1.5 + b2)
                              * std::pow(a[2], 1.5 + b3) * std::pow(a[3], 1.5 + b4)
                              / std::pow(3.0, 0.5 * total_2s);

            sums[idx] *= norm;
        }
    }

    auto orbital_label_d(int i) -> std::string {
        switch (i) {
            case 0:
                return "xy";
            case 1:
                return "xz";
            case 2:
                return "yz";
            case 3:
                return "x2y2";
            case 4:
                return "z2";
            default:
                return "??";
        }
    }

    auto d_norm_coeff(int i) -> double {
        switch (i) {
            case 0:
                return 2.0 / 3.0; // xy
            case 1:
                return 2.0 / 3.0; // xz
            case 2:
                return 2.0 / 3.0; // yz
            case 3:
                return 1.0 / 6.0; // x^2 - y^2
            case 4:
                return 1.0 / 18.0; // 2z^2 - x^2 - y^2
            default:
                return 1.0;
        }
    }

    void print_results_d_subset81(const std::vector<double>& sums, const std::array<int, 3>& comps) {
        for (int idx = 0; idx < 81; ++idx) {
            const int left_pair  = idx / 9;
            const int right_pair = idx % 9;

            const int local_d1 = left_pair / 3;
            const int local_d2 = left_pair % 3;

            const int local_d3 = right_pair / 3;
            const int local_d4 = right_pair % 3;

            const int d1 = comps[local_d1];
            const int d2 = comps[local_d2];
            const int d3 = comps[local_d3];
            const int d4 = comps[local_d4];

            printf("[%3d] (%s,%s)|(%s,%s) : %.10f\n", idx, orbital_label_d(d1).c_str(),
                   orbital_label_d(d2).c_str(), orbital_label_d(d3).c_str(),
                   orbital_label_d(d4).c_str(), sums[idx]);
        }
    }

    void normalize_sums_d_subset81(std::vector<double>& sums, const std::array<double, 4>& a,
                                   const std::array<int, 3>& comps) {
        constexpr double factor = 4.0 / std::numbers::pi;

        for (int idx = 0; idx < 81; ++idx) {
            const int left_pair  = idx / 9;
            const int right_pair = idx % 9;

            const int local_d1 = left_pair / 3;
            const int local_d2 = left_pair % 3;

            const int local_d3 = right_pair / 3;
            const int local_d4 = right_pair % 3;

            const int d1 = comps[local_d1];
            const int d2 = comps[local_d2];
            const int d3 = comps[local_d3];
            const int d4 = comps[local_d4];

            const double c1 = d_norm_coeff(d1);
            const double c2 = d_norm_coeff(d2);
            const double c3 = d_norm_coeff(d3);
            const double c4 = d_norm_coeff(d4);

            const double norm = factor * std::pow(a[0], 3.5) * std::pow(a[1], 3.5)
                              * std::pow(a[2], 3.5) * std::pow(a[3], 3.5)
                              * std::sqrt(c1 * c2 * c3 * c4);

            sums[idx] *= norm;
        }
    }

    void normalize_sums_d_tile(std::vector<double>& tile_vals, const std::array<double, 4>& a,
                               int left_begin, int left_count, int right_begin, int right_count) {
        const int out_count = left_count * right_count;

        if (static_cast<int>(tile_vals.size()) < out_count) {
            throw std::runtime_error("normalize_sums_d_tile: tile_vals too small.");
        }

        constexpr double factor = 4.0 / std::numbers::pi;

        const double a_factor = std::pow(a[0], 3.5) * std::pow(a[1], 3.5) * std::pow(a[2], 3.5)
                              * std::pow(a[3], 3.5);

        for (int local_idx = 0; local_idx < out_count; ++local_idx) {
            const int local_left  = local_idx / right_count;
            const int local_right = local_idx % right_count;

            const int left_pair  = left_begin + local_left;
            const int right_pair = right_begin + local_right;

            const int d1 = left_pair / 5;
            const int d2 = left_pair % 5;

            const int d3 = right_pair / 5;
            const int d4 = right_pair % 5;

            const double c1 = d_norm_coeff(d1);
            const double c2 = d_norm_coeff(d2);
            const double c3 = d_norm_coeff(d3);
            const double c4 = d_norm_coeff(d4);

            const double norm = factor * a_factor * std::sqrt(c1 * c2 * c3 * c4);

            tile_vals[local_idx] *= norm;
        }
    }

    void print_results_d_tile(const std::vector<double>& tile_vals, int left_begin, int left_count,
                              int right_begin, int right_count) {
        const int out_count = left_count * right_count;

        if (static_cast<int>(tile_vals.size()) < out_count) {
            throw std::runtime_error("print_results_d_tile: tile_vals too small.");
        }

        for (int local_idx = 0; local_idx < out_count; ++local_idx) {
            const int local_left  = local_idx / right_count;
            const int local_right = local_idx % right_count;

            const int left_pair  = left_begin + local_left;
            const int right_pair = right_begin + local_right;

            const int d1 = left_pair / 5;
            const int d2 = left_pair % 5;

            const int d3 = right_pair / 5;
            const int d4 = right_pair % 5;

            const int global_idx = left_pair * 25 + right_pair;

            printf("[local %2d | global %3d] (%s,%s)|(%s,%s) : %.10f\n", local_idx, global_idx,
                   orbital_label_d(d1).c_str(), orbital_label_d(d2).c_str(),
                   orbital_label_d(d3).c_str(), orbital_label_d(d4).c_str(), tile_vals[local_idx]);
        }
    }
} // namespace cuslater
#include "cuslater.h"

#define ASSERT assert

namespace cuslater {


    real_t Four_Center_STO_Integrals_Calculator::calculate(const std::array<STO_Basis_Function, 4> &basis_functions)
    {

        Tensor_3D<X1, Y1, Z1> P1(r_grid);
        Tensor_3D<X2, Y2, Z2> P2(r_grid);
        Tensor_3D<X1, Y1, Z1> P3(r_grid);
        Tensor_3D<X2, Y2, Z2> P4(r_grid);

        calculate_basis_function_values(basis_functions[0], P1);
        calculate_basis_function_values(basis_functions[2], P2);
        calculate_basis_function_values(basis_functions[1], P3);
        calculate_basis_function_values(basis_functions[3], P4);

        auto P13 = Hadamard<X1, Y1, Z1>().calculate(P1, P3);
        auto P24 = Hadamard<X2, Y2, Z2>().calculate(P2, P4);

        Tensor_1D<S> wIs(s_grid);

        calculate_s_values(wIs);

        Tensor_3D<X1, X2, S> wEx(ex_grid);
        Tensor_3D<Y1, Y2, S> wEy(ey_grid);
        Tensor_3D<Z1, Z2, S> wEz(ez_grid);

        calculate_exponent_part<X1, X2, S>(wEx);
        calculate_exponent_part<Y1, Y2, S>(wEy);
        calculate_exponent_part<Z1, Z2, S>(wEz);

        Tensor_1D<S> I(s_grid);

        for (auto l = 0U; l < s_grid.size(); ++l) {

            // Get rid of X1 dimension
            Tensor_2D<X1, X2> Ex_page(e_slice_grid_x, wEx, l);
            //Tensor_3D<X2, Y1, Z1> P13X(r_grid);
            auto P13X = tensor_product_3D_with_2D_Contract_1st<X1, Y1, Z1, X2>(P13, Ex_page);

            // Get rid of Y1 dimension
            //Tensor_3D<X2, Y2, Z1> P13XY(r_grid);
            Tensor_2D<Y1, Y2> Ey_page(e_slice_grid_y, wEy, l);
            auto P13XY = tensor_product_3D_with_2D_Contract_2nd<X2, Y1, Z1, Y2>(P13X, Ey_page);

            // Get rid of Z1 dimension
            Tensor_2D<Z1, Z2> Ez_page(e_slice_grid_z, wEz, l);
            //Tensor_3D<X2, Y2, Z1> P24Z(r_grid);
            auto P24Z = tensor_product_3D_with_2D_Contract_3rd<X2, Y2, Z2, Z1>(P24, Ez_page);

            real_t contract = full_3D_contract<X2, Y2, Z1>(P13XY, P24Z);
            I[l] = contract;
        }

        real_t sum_over_s = Tensor_1D_1D_product(wIs, I);
        return sum_over_s;
    }





real_t &Tensor_1D_Impl::operator[](int idx)
{
        ASSERT(idx < grid.size());
    return data.at(idx);
}

real_t Tensor_1D_Impl::operator[](int idx) const
{
    return data.at(idx);
}

template<class Index1, class Index2, class Index3>
Tensor_3D<Index1, Index2, Index3> Hadamard<Index1, Index2, Index3>::calculate(
    const Tensor_3D<Index1, Index2, Index3> &t1,
    const Tensor_3D<Index1, Index2, Index3> &t2
)
{
    auto grid = t1.get_grid();
    auto t2grid = t2.get_grid();
    auto grid_size = grid.get_sizes();
    auto grid2_size = grid.get_sizes();

        ASSERT(std::get<0>(grid_size) == std::get<0>(grid2_size));
        ASSERT(std::get<1>(grid_size) == std::get<1>(grid2_size));
        ASSERT(std::get<2>(grid_size) == std::get<2>(grid2_size));

    Tensor_3D<Index1, Index2, Index3> res(grid);

    const real_t *t1_data_ptr = t1.get_data();
    const real_t *t2_data_ptr = t2.get_data();
    real_t *result_data = res.get_data();

    std::vector<int> modes;
    std::unordered_map<int, int64_t> extent;

    modes = {'x', 'y', 'z'};

    extent.emplace('x', std::get<0>(grid_size));
    extent.emplace('y', std::get<1>(grid_size));
    extent.emplace('z', std::get<2>(grid_size));

    auto rv = hadamar(modes, extent, t1_data_ptr, t2_data_ptr, result_data);

    return res;
}
template<class I1, class I2, class S> void calculate_exponent_part(Tensor_3D<I1, I2, S> &ex) {}

template
    <class Contraction_Index, class A_Index2, class A_Index3, class B_Index2>
Tensor_3D<B_Index2, A_Index2, A_Index3>
tensor_product_3D_with_2D_Contract_1st (const Tensor_3D<Contraction_Index, A_Index2, A_Index3> &,
                                        const Tensor_2D<Contraction_Index, B_Index2> &);


template
    <class A_Index1, class Contraction_Index, class A_Index3, class B_Index2>
Tensor_3D<A_Index1, B_Index2, A_Index3>
tensor_product_3D_with_2D_Contract_2nd (const Tensor_3D<A_Index1, Contraction_Index, A_Index3> &,
                                        const Tensor_2D<Contraction_Index, B_Index2> &);

template
    <class A_Index1, class A_Index2, class Contraction_Index, class B_Index1>
Tensor_3D<A_Index1, A_Index2, B_Index1>
tensor_product_3D_with_2D_Contract_3rd (const Tensor_3D<A_Index1, A_Index2, Contraction_Index> &,
                                        const Tensor_2D<B_Index1, Contraction_Index> &);

template<class Index1, class Index2, class Index3>
real_t full_3D_contract(
    const Tensor_3D<Index1, Index2, Index3> &,
    const Tensor_3D<Index1, Index2, Index3> &
) { return 0;}

void calculate_s_values(Tensor_1D_Impl &t) {}
void calculate_basis_function_values(const STO_Basis_Function &, Tensor_3D_Impl &)  {}
template<class Index> real_t Tensor_1D_1D_product(const Tensor_1D<Index> &t1, const Tensor_1D<Index> &t2)
{
    return 0;
}



}
#pragma once
#include "grids.h"
#include <vector>
#include <memory>
#include <string>

namespace cuslater
{

class Tensor_3D_Impl;

class Tensor_Index  {
public:
    Tensor_Index(char const * name) : index_name(name)
    {
    }
private:
    const std::string index_name;
};

class Tensor_1D_Impl
{
public:
    Tensor_1D_Impl(const Grid_1D &_grid) : grid(_grid)
    {
        data.resize(grid.size());
    }

    real_t &operator[](int idx) ;
    real_t operator[](int idx) const ;

protected:
    const Grid_1D &grid;
private:
    std::vector<real_t> data;

};

class Tensor_2D_Impl
{
public:
    Tensor_2D_Impl(const Grid_2D &_grid) : grid(_grid)
    {
    }
    virtual ~Tensor_2D_Impl() = default;
    const Grid_1D &get_grid(int g) const { return grid.get_grid(g); }

protected:
    const Grid_2D &grid;
};

class Tensor_2D_Impl_Owner : public Tensor_2D_Impl
{
public:
    Tensor_2D_Impl_Owner(const Grid_2D &_grid) : Tensor_2D_Impl(_grid)
    {
        data.resize(grid.size().first*grid.size().second);
    }

private:
    std::vector<real_t> data;
};

class Tensor_2D_Impl_Ref : public Tensor_2D_Impl
{
public:
    Tensor_2D_Impl_Ref(const Grid_2D &_grid, const real_t *_data) : Tensor_2D_Impl(_grid), data(_data)
    {
    }

private:
    const real_t *data = nullptr;
};





template<class Index>
class Tensor_1D : public Tensor_1D_Impl {
public:
    Tensor_1D(const Grid_1D &_grid) : Tensor_1D_Impl(_grid)
    {
        static_assert(std::is_base_of<Tensor_Index, Index>::value, "Index not derived from Tensor_Index");
    }
    Tensor_1D(Tensor_1D<Index> &&) = delete;
    Tensor_1D operator=(Tensor_1D<Index> &&);

};



template<class Index1, class Index2>
class Tensor_2D  {
public:
    Tensor_2D(const Grid_2D &_grid)
    {
        static_assert(std::is_base_of<Tensor_Index, Index1>::value, "Index1 not derived from Tensor_Index");
        static_assert(std::is_base_of<Tensor_Index, Index2>::value, "Index2 not derived from Tensor_Index");
        _impl = std::make_unique<Tensor_2D_Impl_Owner>(_grid);
    }
    Tensor_2D(const Grid_2D &_grid, const Tensor_3D_Impl &tensor_3d, int page);
    const Grid_1D &get_grid(int g) const { return _impl->get_grid(g); }

private:
    std::unique_ptr<Tensor_2D_Impl> _impl;
};

class Tensor_3D_Impl
{
public:
    Tensor_3D_Impl(const Grid_3D &_grid) : grid(_grid)
    {
        auto sizes= grid.get_sizes();
        data.resize(std::get<0>(sizes) * std::get<1>(sizes) * std::get<2>(sizes) );
    }

    const real_t *get_page_data(unsigned int page_no) const
    {
        auto sizes= grid.get_sizes();
        return data.data() + ( sizeof (real_t) * std::get<0>(sizes) * std::get<1>(sizes) ) ;
    }

    const Grid_3D &get_grid() const { return grid; }
    const Grid_1D &get_grid(int g) const { return grid.get_grid(g); }

protected:
    const Grid_3D &grid;
    std::vector<real_t> data;

};
template<class Index1, class Index2, class Index3>
class Tensor_3D : public Tensor_3D_Impl {
public:
    Tensor_3D(const Grid_3D &_grid) : Tensor_3D_Impl(_grid)
    {
        static_assert(std::is_base_of<Tensor_Index, Index1>::value, "Index1 not derived from Tensor_Index");
        static_assert(std::is_base_of<Tensor_Index, Index2>::value, "Index2 not derived from Tensor_Index");
        static_assert(std::is_base_of<Tensor_Index, Index3>::value, "Index3 not derived from Tensor_Index");
    }

    Tensor_3D(const cuslater::Tensor_3D<Index1, Index2, Index3> &&other) : Tensor_3D_Impl(other.grid)
    {
        data = std::move(data);
    }

    //Tensor_3D<Index1, Index2, Index3> & operator=(Tensor_3D<Index1, Index2, Index3> &&) ;
    real_t *get_data() { return data.data(); }
    const real_t *get_data() const { return data.data(); }
};




#define MAKE_INDEX(n) \
class n : public Tensor_Index \
{\
public: \
    n () : Tensor_Index(#n) {}\
    static const char *get_name() { return #n ;} \
}  ;





template<class Index1, class Index2, class Index3>
class Hadamard {
public:
    Hadamard()
    {
        static_assert(std::is_base_of<Tensor_Index, Index1>::value,
                      "Index1 not derived from Tensor_Index");
        static_assert(std::is_base_of<Tensor_Index, Index2>::value,
                      "Index2 not derived from Tensor_Index");
        static_assert(std::is_base_of<Tensor_Index, Index3>::value,
                      "Index3 not derived from Tensor_Index");
    }

    Tensor_3D<Index1, Index2, Index3> calculate(
        const Tensor_3D<Index1, Index2, Index3> &t1,
        const Tensor_3D<Index1, Index2, Index3> &t2);
};




template<class I1, class I2, class S> void calculate_exponent_part(Tensor_3D<I1, I2, S> &ex);

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
);

void calculate_s_values(Tensor_1D_Impl &t);
void calculate_basis_function_values(const STO_Basis_Function &, Tensor_3D_Impl &) ;

template<class Index> real_t Tensor_1D_1D_product(const Tensor_1D<Index> &t1, const Tensor_1D<Index> &t2);


}
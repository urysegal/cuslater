#include <string>
#include <algorithm>
#include <array>

namespace cuslater {

typedef enum error_code_t : int
{
    SUCCESS = 0,

} error_code_t;

typedef double real_t;
typedef  int angular_quantum_number_t;
typedef  int principal_quantum_number_t;
typedef int magnetic_quantum_number_t;
enum class spin_quantum_number_t  { UNDEFINED, UP, DOWN } ;
typedef double sto_exponent_t;
/// A coordinate in 1D space
typedef double spatial_coordinate_t;
/// A coordinate in 3D space
typedef std::array<spatial_coordinate_t, 3> center_t;

/// A normalization_coefficient or Normalization normalization_coefficient
typedef double sto_coefficient_t;

struct Quantum_Numbers {
    principal_quantum_number_t n = 0; /// principal quantum number, also known as 'n'
    angular_quantum_number_t l = 0; /// Angular moment number, also known as 'l'
    magnetic_quantum_number_t m = 0; /// Magnetic/Orientation quantum number, also known as 'm' or 'ml'
    spin_quantum_number_t ms = spin_quantum_number_t::UNDEFINED; /// Spin Quantum number, a.k.a. 'ms'
    void validate() const;
};

class STO_Basis_Function_Info {

    sto_coefficient_t normalization_coefficient; /// Normalization Coefficient N(n,alpha) of the radial part. also known as N
    sto_exponent_t exponent; /// alpha of the radial part. also known as alpha
    Quantum_Numbers quantum_numbers;     /// Quantum numbers for this basis function

public:

    /// Get the set of quantum numbers for this basis function
    /// \return set of quantum numbers for this basis function
    const Quantum_Numbers &get_quantum_numbers() const;

    /// Set the quantum numbers for this basis function
    /// \param quantum_numbers set of quantum numbers to use
    void set_quantum_numbers(const Quantum_Numbers &quantum_numbers);

    /// Get the alpha used by this basis function
    /// \return the alpha used by this basis function
    sto_exponent_t get_exponent() const ;

    /// Set the alpha used by this basis function
    /// \param e alpha to use
    void set_exponent(sto_exponent_t e) ;

    /// Get the normalization_coefficient used by this basis function
    /// \return the normalization_coefficient used by this basis function
    sto_coefficient_t get_coefficient() const ;
    /// Set the normalization_coefficient used by this basis function
    /// \param c normalization_coefficient to use
    void set_coefficient(sto_coefficient_t c) ;


public:
    /// Construct an STO style basis function detail object.
    /// \param exponent_ alpha of the radial part.
    /// \param quantum_numbers Set of quantum numbers for this function
    STO_Basis_Function_Info( sto_exponent_t exponent_, const Quantum_Numbers &quantum_numbers);

};

class Tensor_3D_Impl;

class STO_Basis_Function {

    STO_Basis_Function_Info function_info; /// Basis function parameters
    center_t center; /// the center of the function

public:

    /// Construct a basis function located at a specific coordinates
    /// \param function_info Basis function information
    /// \param location Cartesian center of the function
    STO_Basis_Function(STO_Basis_Function_Info function_info_, center_t location_);

    /// Get the set of quantum numbers for this basis function
    /// \return set of quantum numbers
    const Quantum_Numbers &get_quantum_numbers() const;

    /// Get the alpha used by this basis function
    /// \return the alpha used by this basis function
    sto_exponent_t get_exponent() const;

    /// Get the normalization_coefficient used by this basis function
    /// \return the normalization_coefficient used by this basis function
    sto_coefficient_t get_normalization_coefficient() const;

    /// Get the spatial center of this basis function
    /// \return the spatial center of this basis function
    center_t get_center() const;

    void calculate(const Tensor_3D_Impl &) const;

};

class Grid {
public:
};



class Grid_1D : public Grid {
public:
    unsigned int size() const;
};

class Grid_2D : public Grid {
public:
    Grid_2D(const Grid_1D &, const Grid_1D &);
};



class Grid_3D : public Grid {
public:
    Grid_3D(const Grid_3D &);
    Grid_3D(const Grid_1D &, const Grid_1D &, const Grid_1D &);
};


class Logarithmic_1D_Grid : public Grid_1D
{
public:
    Logarithmic_1D_Grid(real_t from, real_t to, int steps);
    real_t operator[](unsigned int n) const;

};

class Equidistance_1D_Grid : public Grid_1D {
public:
    Equidistance_1D_Grid(real_t from, real_t to, int steps);
};

class Tensor_Index  {
public:
    Tensor_Index(char const * name) : index_name(name)
    {
    }
private:
    const std::string index_name;
};

class Tensor_3D_Impl
{
public:
    Tensor_3D_Impl(const Grid_3D &_grid) : grid(_grid) {}
protected:
    const Grid_3D &grid;

};

class Tensor_2D_Impl
{
public:
    Tensor_2D_Impl(const Grid_2D &_grid) : grid(_grid) {}
protected:
    const Grid_2D &grid;

};

template<class Index1, class Index2>
class Tensor_2D : public Tensor_2D_Impl {
public:
    Tensor_2D(const Grid_2D &_grid) : Tensor_2D_Impl(_grid)
    {
        static_assert(std::is_base_of<Tensor_Index, Index1>::value, "Index1 not derived from Tensor_Index");
        static_assert(std::is_base_of<Tensor_Index, Index2>::value, "Index2 not derived from Tensor_Index");
    }
    Tensor_2D(Tensor_2D<Index1, Index2> &&) = delete;
    Tensor_2D operator=(Tensor_2D<Index1, Index2> &&);
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
    Tensor_3D(Tensor_3D<Index1, Index2, Index3> &&) = delete;
    Tensor_3D operator=(Tensor_3D<Index1, Index2, Index3> &&);
    Tensor_2D<Index1, Index2> get_page(unsigned int page_no);
};



#define MAKE_INDEX(n) \
class n : public Tensor_Index \
{\
public: \
    n () : Tensor_Index(#n) {}\
}  ;



template<class Index1, class Index2, class Index3>
class General_3D_Grid : public Grid_3D {
public:
    General_3D_Grid(const General_3D_Grid &_grid) : Grid_3D(_grid)
    {
        static_assert(std::is_base_of<Tensor_Index, Index1>::value, "Index1 not derived from Tensor_Index");
        static_assert(std::is_base_of<Tensor_Index, Index2>::value, "Index2 not derived from Tensor_Index");
        static_assert(std::is_base_of<Tensor_Index, Index3>::value, "Index3 not derived from Tensor_Index");
    }
    General_3D_Grid(const Grid_1D &grid1,const Grid_1D &grid2, const Grid_1D &grid3 ) : Grid_3D(grid1,grid2,grid3)
    {
        static_assert(std::is_base_of<Tensor_Index, Index1>::value, "Index1 not derived from Tensor_Index");
        static_assert(std::is_base_of<Tensor_Index, Index2>::value, "Index2 not derived from Tensor_Index");
        static_assert(std::is_base_of<Tensor_Index, Index3>::value, "Index3 not derived from Tensor_Index");
    }

};

template<class Index1, class Index2>
class General_2D_Grid : public Grid_2D {
public:
    General_2D_Grid(const Grid_1D &grid1,const Grid_1D &grid2 ) : Grid_2D(grid1,grid2)
    {
        static_assert(std::is_base_of<Tensor_Index, Index1>::value, "Index1 not derived from Tensor_Index");
        static_assert(std::is_base_of<Tensor_Index, Index2>::value, "Index2 not derived from Tensor_Index");
    }

};

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
        const Tensor_3D<Index1, Index2, Index3> &t2
    );
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



error_code_t calculate(const std::array<STO_Basis_Function, 4> &basis_functions)
{

    MAKE_INDEX(X1)
    MAKE_INDEX(X2)
    MAKE_INDEX(Y1)
    MAKE_INDEX(Y2)
    MAKE_INDEX(Z1)
    MAKE_INDEX(Z2)
    MAKE_INDEX(S)

    Equidistance_1D_Grid x_grid(-10,10,100);
    Equidistance_1D_Grid y_grid(x_grid);
    Equidistance_1D_Grid z_grid(x_grid);

    General_3D_Grid<X1, Y1, Z1> r1_grid(x_grid, y_grid, z_grid);
    General_3D_Grid<X2, Y2, Z2> r2_grid(x_grid, y_grid, z_grid);


    Tensor_3D<X1, Y1, Z1> P1(r1_grid);
    Tensor_3D<X2, Y2, Z2> P2(r2_grid);
    Tensor_3D<X1, Y1, Z1> P3(r1_grid);
    Tensor_3D<X2, Y2, Z2> P4(r2_grid);



    basis_functions[0].calculate(P1);
    basis_functions[2].calculate(P3);
    basis_functions[1].calculate(P2);
    basis_functions[3].calculate(P4);


    Tensor_3D<X1,Y1,Z1> P13(r1_grid);
    P13 = Hadamard<X1,Y1,Z1>().calculate(P1, P3);
    Tensor_3D<X2,Y2,Z2> P24(r2_grid);
    P24 = Hadamard<X2,Y2,Z2>().calculate(P2, P4);

    Logarithmic_1D_Grid s_grid(0,1000,50);

    General_3D_Grid<X1, X2, S> ex_grid(x_grid, x_grid, s_grid);
    General_3D_Grid<Y1, Y2, S> ey_grid(y_grid, y_grid, s_grid);
    General_3D_Grid<Z1, Z2, S> ez_grid(z_grid, z_grid, s_grid);

    Tensor_3D<X1, X2, S> Ex(ex_grid);
    Tensor_3D<Y1, Y2, S> Ey(ey_grid);
    Tensor_3D<Z1, Z2, S> Ez(ez_grid);

    calculate_exponent_part<X1, X2, S>(Ex);
    calculate_exponent_part<Y1, Y2, S>(Ey);
    calculate_exponent_part<Z1, Z2, S>(Ez);

    General_2D_Grid<X1, X2> e_slice_grid_x(x_grid, x_grid);
    General_2D_Grid<X1, X2> e_slice_grid_y(y_grid, y_grid);

    General_3D_Grid<X2, Y1, Z1> p13x_grid(x_grid, y_grid, z_grid);
    General_3D_Grid<X2, Y2, Z1> p13xy_grid(x_grid, y_grid, z_grid);

    for ( auto l = 0U ; l < s_grid.size() ; ++l ) {
        // Get rid of X1 dimension

        Tensor_2D<X1, X2> Ex_page(e_slice_grid_x);
        Ex_page = Ex.get_page(l);
        Tensor_3D<X2, Y1, Z1> P13X(p13x_grid);
        P13X = tensor_product_3D_with_2D_Contract_1st<X1,Y1,Z1,X2>(P13, Ex_page);

        // Get rid of Y1 dimension
        Tensor_3D<X2, Y2, Z1> P13XY(p13xy_grid);
        Tensor_2D<Y1, Y2> Ey_page(e_slice_grid_y);
        Ey_page = Ey.get_page(l);
        P13XY = tensor_product_3D_with_2D_Contract_2nd<X2,Y1,Z1,Y2>(P13X, Ey_page);


    }
}

}
#pragma once

namespace cuslater {

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


};

}
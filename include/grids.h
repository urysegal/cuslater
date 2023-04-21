#pragma once

namespace cuslater {

typedef double real_t;

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

class General_3D_Grid : public Grid_3D {
public:

    General_3D_Grid(const Grid_1D &grid1,const Grid_1D &grid2, const Grid_1D &grid3 ) : Grid_3D(grid1,grid2,grid3)
    {
    }

};

class General_2D_Grid : public Grid_2D {
public:
    General_2D_Grid(const Grid_1D &grid1,const Grid_1D &grid2 ) : Grid_2D(grid1,grid2)
    {
    }

};
}
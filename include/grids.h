#pragma once
#include <map>

namespace cuslater {

typedef double real_t;

class Grid {
public:
};



class Grid_1D : public Grid {
    Grid_1D(int _size) : grid_size(_size) {}
public:
    unsigned int size() const { return grid_size; }
private:
    int grid_size = 0;
};

class Grid_2D : public Grid {
public:
    Grid_2D(const Grid_1D &g1, const Grid_1D &g2) : modes({g1,g2}) {}

    std::pair<int, int> size() const { return std::make_pair<int, int>(modes.first.size(), modes.second.size());}
private:
    std::pair<const Grid_1D &, const Grid_1D &> modes ;
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
#pragma once
#include <map>
#include <exception>
#include <vector>

namespace cuslater {

typedef double real_t;

class Grid {
public:
};



class Grid_1D : public Grid {
public:
    Grid_1D(int _size) : grid_size(_size) {}
    unsigned int size() const { return grid_size; }
    virtual const std::vector<real_t> & get_data() const  = 0;
private:
    int grid_size = 0;
};

class Grid_2D : public Grid {
public:
    Grid_2D(const Grid_1D &g1, const Grid_1D &g2) : modes({g1,g2}) {}

    std::pair<int, int> size() const { return std::make_pair<int, int>(modes.first.size(), modes.second.size());}
    const Grid_1D& get_grid(int g) const {
        switch (g) {
            case  0:
                return std::get<0>(modes);
            case  1:
                return std::get<1>(modes);
            default:
                throw std::exception();
        }
    }

private:
    std::pair<const Grid_1D &, const Grid_1D &> modes ;
};



class Grid_3D : public Grid {
public:
    Grid_3D(const Grid_1D &g1, const Grid_1D &g2, const Grid_1D &g3) :
        sizes ({ g1.size(), g2.size(), g3.size() }),
        grids({g1,g2,g3})
                  { }
    std::tuple<int, int, int> get_sizes() const { return sizes ;}
    const Grid_1D& get_grid(int g) const {
        switch (g) {
            case  0:
                return std::get<0>(grids);
            case  1:
                return std::get<1>(grids);
            case  2:
                return std::get<2>(grids);
            default:
                throw std::exception();
        }
    }
private:
    std::tuple<int, int, int> sizes;
    std::tuple<const Grid_1D &, const Grid_1D &, const Grid_1D &> grids;

};



class Logarithmic_1D_Grid : public Grid_1D
{
public:
    Logarithmic_1D_Grid(real_t _from, real_t _to, int _steps): Grid_1D(_steps),
    from(_from) , to(_to), steps(_steps)
    {}
    real_t operator[](unsigned int n) const;
    const std::vector<real_t> & get_data() const override { return points; }
private:
    real_t from=0;
    real_t to=0;
    int steps =0;
    std::vector<real_t> points;
};

class Equidistance_1D_Grid : public Grid_1D {
public:
    Equidistance_1D_Grid(real_t _from, real_t _to, int _steps) : Grid_1D(_steps),
     from(_from), to(_to), steps(_steps)
    {}
    real_t operator[](unsigned int n) const;
    const std::vector<real_t> & get_data() const override
    {
        if ( points.empty() ) {
            calculate_points();
        }
        return points;
    }

private:

    void calculate_points() const
    {
        auto s = (to-from)/double(steps);
        for ( int i = 0 ; i < steps ; ++ i) {
            points.emplace_back(from + i*s);
        }
    }
    real_t from = 0;
    real_t to = 0;
    int steps = 0;
    mutable std::vector<real_t> points;
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
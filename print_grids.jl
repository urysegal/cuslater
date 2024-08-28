
include("grids.jl")

nv = 25:25:500
for i in eachindex(nv)
    n = nv[i]
    generate_grid_x1_legendre(n)
    generate_grid_x1_legendre(n,Float32)
end

for i in 1:20
    generate_grid_l_lebedev(i)
    generate_grid_l_lebedev(i,Float32)
end

nv = 50:50:600
for i in eachindex(nv)
    n = nv[i]
    generate_grid_r_laguerre(n)
    generate_grid_r_laguerre(n,Float32)
end
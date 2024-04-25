using FastGaussQuadrature, Plots, LazyGrids, Lebedev
using DelimitedFiles

function making_zeros(x)
    ϵ=eps()
    for i in eachindex(x)
        if abs(x[i]) < ϵ
            x[i] = 0
        end
    end
    return x
end


f1(x,y,z)=exp.(-sqrt.( (x .^ 2) + (y .^ 2) + (z .^ 2) ))
f2(x,y,z)=exp.(-sqrt.(((x .- 1) .^ 2) + (y .^ 2) + (z.^ 2)))
g1(x,y,z,x2,y2,z2,r) = exp.(-sqrt.(((x .+ (r .* x2) .- 2) .^ 2) + ((y .+ (r .* y2)).^ 2) + ((z .+ (r .* z2)).^ 2)))
g2(x,y,z,x2,y2,z2,r) = exp.(-sqrt.(((x .+ (r .* x2) .- 3) .^ 2) + ((y .+ (r .* y2)) .^ 2) + ((z .+ (r .* z2)).^ 2)))
g(x,y,z,x2,y2,z2,r) = g1(x,y,z,x2,y2,z2,r) .* g2(x,y,z,x2,y2,z2,r)
n=375
nl= 590
#Unit Sphere Grid
# lnodes_x, lnodes_y, lnodes_z, lweights =  lebedev_by_points(nl)
# lweights=making_zeros(Float32.(lweights))
# lweights_new=Float32.(lweights[lweights .!= 0])
# lnodes_x_new=Float32.(lnodes_x[lweights .!= 0])
# lnodes_y_new=Float32.(lnodes_y[lweights .!= 0])
# lnodes_z_new=Float32.(lnodes_z[lweights .!= 0])
# println("number of zeroed:",length(lnodes_x)-length(lnodes_x_new))
# #Laguerre grid for infinite r and \alpha = 0
# rnodes, rweights = gausslaguerre(n+40)
# lr0=length(rnodes)
# rweights=making_zeros(Float32.(rweights))
# rweights_new=Float32.(rweights[rweights .!= 0])
# rnodes_new=Float32.(rnodes[rweights .!= 0])
# lr=length(rnodes_new)
# println("number of zeroed:",length(rnodes)-length(rnodes_new))

nodes, weights = gausslegendre(n+21)
a = -9
b = 10
nodes = ((b-a)/2)*nodes .+ ((a+b)/2)
weights = ((b-a)/2)*weights
weights=making_zeros(Float32.(weights))
weights_new=Float32.(weights[weights .!= 0])
nodes_new=Float32.(nodes[weights .!= 0])
println("number of zeroed:",length(nodes)-length(nodes_new))


(xg, yg, zg) = ndgrid(nodes_new,nodes_new,nodes_new)
(wxg, wyg, wzg) = ndgrid(weights_new,weights_new,weights_new)
F=(wxg .* wyg .* wzg) .* f1.(xg,yg,zg) .* f2.(xg,yg,zg)
ϵ=eps()
less= (abs.(F) .< ϵ)
nxg = xg[.!(less)]
nyg = yg[.!(less)]
nzg = zg[.!(less)]
nwxg = wxg[.!(less)]
nwyg = wyg[.!(less)]
nwzg = wzg[.!(less)]

x1_file = "nz_grids_files/ep64/x1_legendre_3d_ep64_$(n).grid"
open(x1_file, "w") do file
    # Write nodes_new and weights_new
    writedlm(file, [nxg  nwxg], ' ')
end
y1_file = "nz_grids_files/ep64/y1_legendre_3d_ep64_$(n).grid"
open(y1_file, "w") do file
    # Write nodes_new and weights_new
    writedlm(file, [nyg  nwyg], ' ')
end
z1_file = "nz_grids_files/ep64/z1_legendre_3d_ep64_$(n).grid"
open(z1_file, "w") do file
    # Write nodes_new and weights_new
    writedlm(file, [nzg  nwzg], ' ')
end




#code by Avleen for grid generation 

using FastGaussQuadrature, Plots, Lebedev
using DelimitedFiles

println("Current files are being stored in:")
println(pwd())

#function for zeroing out the numbers with magnitude less than machine precision
function making_zeros(x,precision_type=Float64)
    ϵ=eps(precision_type)
        for i in eachindex(x)
            if abs(x[i]) < ϵ
            x[i] = 0
            end
        end
    return x
end

#code for composite Simpson 3/8th
function comp_sim_new(a,b,n)
	#n must be multiple of 3
	if n%3 != 0
        n=3*ceil(n/3)
    end
	h=(b-a)/n
	#equidistant nodes
	nodes=a:h:b
	#weights for simpson 3/8th
	weights = ones(n+1) 
	weights[2:3:n] .= 3
	weights[3:3:n] .= 3
	weights[4:3:n] .= 2
	weights = weights * h * (3/8)
	return nodes,weights,h
end

#Gauss Generalized Laguerre grid for a given precision_type = Float64, Float32, so on...
function generate_grid_r_laguerre(nr_start::Int=10,precision_type=Float64)
	# r points
	#alpha = 1 for generalized Laguerre
	rnodes, rweights = gausslaguerre(nr_start,1) 
	rnodes=precision_type.(rnodes)
	rweights=precision_type.(rweights)
	rweights=making_zeros(rweights,precision_type)
	rweights_new=rweights[rweights .!= 0]
	rnodes_new=rnodes[rweights .!= 0]
	nr=length(rnodes_new)
	if precision_type == Float64
		r_file = "../grid_files/lag64/r_$(nr_start)_$(nr).grid"
	end
	if precision_type == Float32
		r_file = "../grid_files/lag32/r_$(nr_start)_$(nr).grid"
	end
    writedlm(r_file, [rnodes_new rweights_new])
	return nr
end

#Gauss Legendre grid for a given precision_type = Float64, Float32, so on...
function generate_grid_l_lebedev(nl_start::Int=2,precision_type=Float64)
	#lebedev points
    avail=getavailablepoints()
	nl= avail[nl_start]
	lnodes_x, lnodes_y, lnodes_z, lweights =  lebedev_by_points(nl)
	lnodes_x = precision_type.(lnodes_x)
	lnodes_y = precision_type.(lnodes_y)
	lnodes_z = precision_type.(lnodes_z)
	lweights = precision_type.(lweights)	
	lweights = making_zeros(lweights,precision_type)
	lweights_new = lweights[lweights .!= 0]
	lxgrid = lnodes_x[lweights .!= 0]
	lygrid = lnodes_y[lweights .!= 0]
	lzgrid = lnodes_z[lweights .!= 0]
	nl = length(lzgrid)
	if precision_type == Float64
		l_file = "../grid_files/leb64/l_$(nl).grid"
	end
	if precision_type == Float32
		l_file = "../grid_files/leb32/l_$(nl).grid"
	end
    writedlm(l_file, [lxgrid lygrid lzgrid lweights_new],' ')
	return nl
end

#generating grid file for Simpson for a given precision_type = Float64, Float32, so on...
function generate_grid_x1_simpson(a,b,n,precision_type=Float64)
	nodes, weights = comp_sim(a,b,n)
	nodes = precision_type.(nodes)
	weights = precision_type.(weights)
	weights=making_zeros(weights,precision_type)
	weights_new=weights[weights .!= 0]
	nodes_new=nodes[weights .!= 0]
	nx = length(weights_new)
	if precision_type == Float64
		x1_file = "../grid_files/sim64/x1_simpson_1d_$(nx-1).grid"
	end
	if precision_type == Float32
		x1_file = "../grid_files/sim32/x1_simpson_1d_$(nx-1).grid"
	end
	open(x1_file, "w") do file
			# Write a and b values in the first two lines
			println(file, a, " ", b)
			# Write nodes_new and weights_new
			writedlm(file, [nodes_new weights_new], ' ')
	end

	return nx
end

#generating grid file for Legendre for a given precision_type = Float64, Float32, so on...
function generate_grid_x1_legendre(n,precision_type=Float64)
	nodes, weights = gausslegendre(n)
	nodes = precision_type.(nodes)
	weights = precision_type.(weights)
	weights=making_zeros(weights,precision_type)
	weights_new=weights[weights .!= 0]
	nodes_new=nodes[weights .!= 0]
	nx = length(weights_new)
	if precision_type == Float64
		x1_file = "../grid_files/leg64/x1_legendre_1d_$(nx).grid"
	end
	if precision_type == Float32
		x1_file = "../grid_files/leg32/x1_legendre_1d_$(nx).grid"
	end
	open(x1_file, "w") do file
		# Write nodes_new and weights_new
		writedlm(file, [nodes_new weights_new], ' ')
	end
	return nx
end
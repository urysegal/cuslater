
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
function comp_sim(a,b,n)
    h=(b-a)/n
    nodes=a:h:b
    weights = h*(3/8)*ones(n+1)
    for i in eachindex(weights)
        if ((mod(i-1,3)==0 && i >1) && (i < (n+1)))
            weights[i]=weights[i]*2
        elseif ((i > 1) && (i < (n+1)))
            weights[i]=weights[i]*3
        end
    end
    return nodes,weights,h
end

function generate_grid_r_laguerre(nr_start::Int=10)
	# r points
	rnodes, rweights = gausslaguerre(nr_start,1)
	rweights=making_zeros(rweights)
	rweights_new=rweights[rweights .!= 0]
	rnodes_new=rnodes[rweights .!= 0]
	nr=length(rnodes_new)
	r_file = "grid_files/r_$(nr).grid"
    writedlm(r_file, [rnodes_new rweights_new])
	return nr
end

function generate_grid_l_lebedev(nl_start::Int=2)
	#lebedev points
    avail=getavailablepoints()
	nl= avail[nl_start]
	lnodes_x, lnodes_y, lnodes_z, lweights =  lebedev_by_points(nl)
	lweights=making_zeros(lweights)
	lweights_new=lweights[lweights .!= 0]
	lxgrid = lnodes_x[lweights .!= 0]
	lygrid = lnodes_y[lweights .!= 0]
	lzgrid = lnodes_z[lweights .!= 0]
	nl = length(lzgrid)
	l_file = "grid_files/l_$(nl).grid"
    writedlm(l_file, [lxgrid lygrid lzgrid lweights_new],' ')
	return nl
end

function generate_grid_x1_simpson(a,b,n)
	nodes, weights = comp_sim(a,b,n)
	weights=making_zeros(weights)
	weights_new=weights[weights .!= 0]
	nodes_new=nodes[weights .!= 0]
	nx = length(weights_new)
	x1_file = "grid_files/x1_simpson_1d_$(nx-1).grid"
	open(x1_file, "w") do file
			# Write a and b values in the first two lines
			println(file, a, " ", b)
			# Write nodes_new and weights_new
			writedlm(file, [nodes_new weights_new], ' ')
	end

	return nx
end

function generate_grid_x1_legendre(a,b,n)
		nodes, weights = gausslegendre(n)
        nodes = ((b-a)/2)*nodes .+ ((a+b)/2)
        weights = ((b-a)/2)*weights
        weights=making_zeros(weights)
        weights_new=weights[weights .!= 0]
        nodes_new=nodes[weights .!= 0]
		nx = length(weights_new)
		x1_file = "grid_files/x1_legendre_1d_$(nx).grid"
		open(x1_file, "w") do file
			# Write a and b values in the first two lines
			println(file, a, " ", b)
			# Write nodes_new and weights_new
			writedlm(file, [nodes_new weights_new], ' ')
    	end

		return nx
end

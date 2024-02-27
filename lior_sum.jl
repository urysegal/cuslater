# Set up the interface with the cuda code
include("interface_cuda.jl")
using FastGaussQuadrature, Plots, Lebedev, TensorOperations
using Base.Threads
using ChunkSplitters
function making_zeros(x)
    ϵ=eps()
    for i in eachindex(x)
        if abs(x[i]) < ϵ 
            x[i] = 0
        end
    end
    return x
end
# Define a function to create InputData instances
function create_InputData(rval)
    InputData(
        [0.0,0.0,0.0],      #c1
        [1.0,0.0,0.0],      #c2
        [2.0,0.0,0.0],      #c3
        [3.0,0.0,0.0],      #c4
        rval,               # r
	[0.1,0.2,0.3],      #w
        [-5.0,10.0],        #xrange
        [-5.0,10.0],        #yrange
        [-5.0,10.0],        #zrange
        300,              # x_axis_points
        300,              # y_axis_points
        300               # z_axis_points
    )
end
function evaluate_broadcast(nr_start)
	#initialize data
	rdata = InputData[]
	rnodes, rweights = gausslaguerre(nr_start,1)
	rweights=making_zeros(rweights)
	rweights_new=rweights[rweights .!= 0]
	rnodes_new=rnodes[rweights .!= 0]
	nr=length(rnodes_new)
	
	#lebedev points
        avail=getavailablepoints()
	nl= avail[2]
	lnodes_x, lnodes_y, lnodes_z, lweights =  lebedev_by_points(nl)
	lweights=making_zeros(lweights)
	lweights_new=lweights[lweights .!= 0]
	lxgrid = lnodes_x[lweights .!= 0]
	lygrid = lnodes_y[lweights .!= 0]
	lzgrid = lnodes_z[lweights .!= 0]
	rwdata = [create_InputData(args...) for args in Iterators.product(rnodes_new,)]
	integral_sum_r = evaluate_inner.(rwdata)
	println("nr = ",nr_start,", nl = ",nl," Result of evaluate_inner: ", sum(integral_sum_r))
end
function evaluate_broadcast_simple(nr_start)
	#initialize data
	#r points
	rnodes, rweights = gausslaguerre(nr_start,1)
	rweights=making_zeros(rweights)
	rweights_new=rweights[rweights .!= 0]
	rnodes_new=rnodes[rweights .!= 0]
	nr=length(rnodes_new)
	
	#lebedev points
        avail=getavailablepoints()
	nl= avail[2]
	lnodes_x, lnodes_y, lnodes_z, lweights =  lebedev_by_points(nl)
	lweights=making_zeros(lweights)
	lweights_new=lweights[lweights .!= 0]
	lxgrid = lnodes_x[lweights .!= 0]
	lygrid = lnodes_y[lweights .!= 0]
	lzgrid = lnodes_z[lweights .!= 0]
	nl = length(lzgrid)
	
	integral_sum_r = [evaluate_inner_simple(rnodes_new[i],lxgrid[l],lygrid[l],lzgrid[l]) for i in 1:nr for l in 1:nl] 
	println("nr = ",nr_start,", nl = ",nl," Result of evaluate_inner: ", sum(integral_sum_r))
end
@time evaluate_broadcast(10)

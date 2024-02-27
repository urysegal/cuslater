<<<<<<< HEAD
# Set up the interface with the cuda code11i
include("interface_cuda.jl")
include("grids.jl")

# Define a function to create InputData instances
function evaluate_broadcast(nr_start::Int=10,nl_start::Int=2,nx::Int=9)
=======
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
>>>>>>> 88cd7ee (lior sum)
	#initialize data
	rdata = InputData[]
	rnodes, rweights = gausslaguerre(nr_start,1)
	rweights=making_zeros(rweights)
	rweights_new=rweights[rweights .!= 0]
	rnodes_new=rnodes[rweights .!= 0]
	nr=length(rnodes_new)
<<<<<<< HEAD

	#lebedev points
	avail=getavailablepoints()
	nl= avail[nl_start]
=======
	
	#lebedev points
        avail=getavailablepoints()
	nl= avail[2]
>>>>>>> 88cd7ee (lior sum)
	lnodes_x, lnodes_y, lnodes_z, lweights =  lebedev_by_points(nl)
	lweights=making_zeros(lweights)
	lweights_new=lweights[lweights .!= 0]
	lxgrid = lnodes_x[lweights .!= 0]
	lygrid = lnodes_y[lweights .!= 0]
	lzgrid = lnodes_z[lweights .!= 0]
<<<<<<< HEAD
	#	 wdata = [create_InputData(rval,lxgrid[l],lygrid[l],lzgrid[l],nx) for l in 1:nl for rval in  rnodes_new ]
	nl = length(lzgrid)
	integral_sum = 0
	r_sums = zeros(nr,1)
	for l in 1:nl
		rdata = [create_InputData(rval,lxgrid[l],lygrid[l],lzgrid[l],nx) for rval in rnodes_new]
		r_sums .= evaluate_inner.(rdata,0)
		r_sums .= r_sums .* rweights_new
		r_sum  = sum(r_sums)
	#		integral_sum += evaluate_inner(rwdata[(l-1)*nr + r],0)*rweights_new[r]*lweights_new[l]
		integral_sum += r_sum * lweights_new[l]
		println("Computed for l = ",l)
	end
	result = 4/pi * integral_sum
	println("nr = ",nr_start,", nl = ",nl)
	return result
end

function parallel_integral_sum(rweights_new, lweights_new, lxgrid, lygrid, lzgrid, nx, rnodes_new)
    l_ctr = Atomic{Int}(0) # Atomic variable to safely track the number of threads executed
    nl = length(lxgrid)
	gpu_num = 2
	l_sum = zeros(nl,1)
	    @threads for l in 1:nl
            thread_id = threadid()
            r_sums = zeros(length(rnodes_new)) # Initialize an array to store intermediate sums
            rdata = [create_InputData(rval,lxgrid[l],lygrid[l],lzgrid[l],nx) for rval in rnodes_new]
            r_sums .= evaluate_inner.(rdata,mod(thread_id,gpu_num))
            r_sums .= r_sums .* rweights_new
            r_sum  = sum(r_sums)
            l_sum[l] = r_sum * lweights_new[l]
            atomic_add!(l_ctr, 1) # Update the ctr atomically
            println("computed for l: ",l, " l_ctr= ",l_ctr)
	   end
	integral_sum = sum(l_sum)
    println("l_sum_vec: ", l_sum)
    return integral_sum[]
end


function parallel_integral_sum_lr(rweights_new, lweights_new, lxgrid, lygrid, lzgrid, nx, rnodes_new,gpu_num)
     l_ctr = Atomic{Int}(0) # Atomic variable to safely track the number of threads executed
    nl = length(lxgrid)
	nr = length(rnodes_new)
	nlr = nl*nr
	lr_sum = zeros(nlr,1)
	@threads for i in 1:nlr
		device_id = mod(i,gpu_num)
		l_i = cld(i,nr)
		r_i = mod(i,nr) + 1
		rdata = create_InputData(rnodes_new[r_i],lxgrid[l_i],lygrid[l_i],lzgrid[l_i],nx)
		lr_sum[i] = evaluate_inner(rdata,device_id ) * rweights_new[r_i]	* lweights_new[l_i]
 	        atomic_add!(l_ctr, 1) # Update the ctr atomically
 		if ( mod(l_ctr[],1000)==0 )
 			println("computed for l_i: ",l_i," r_i: ",r_i, " computed= ",l_ctr[]," / ",nlr)
 		end
	end
	integral_sum = sum(lr_sum)
#    println("l_sum_vec: ", lr_sum)
    return integral_sum[]
end

function evaluate_broadcast_parallel(nr_start::Int=10,nl_start::Int=2,nx::Int=9,gpu_num::Int=1)
	#initialize data
	rdata = InputData[]
=======
	rwdata = [create_InputData(args...) for args in Iterators.product(rnodes_new,)]
	integral_sum_r = evaluate_inner.(rwdata)
	println("nr = ",nr_start,", nl = ",nl," Result of evaluate_inner: ", sum(integral_sum_r))
end
function evaluate_broadcast_simple(nr_start)
	#initialize data
	#r points
>>>>>>> 88cd7ee (lior sum)
	rnodes, rweights = gausslaguerre(nr_start,1)
	rweights=making_zeros(rweights)
	rweights_new=rweights[rweights .!= 0]
	rnodes_new=rnodes[rweights .!= 0]
	nr=length(rnodes_new)
<<<<<<< HEAD

	#lebedev points
    avail=getavailablepoints()
	nl= avail[nl_start]
=======
	
	#lebedev points
        avail=getavailablepoints()
	nl= avail[2]
>>>>>>> 88cd7ee (lior sum)
	lnodes_x, lnodes_y, lnodes_z, lweights =  lebedev_by_points(nl)
	lweights=making_zeros(lweights)
	lweights_new=lweights[lweights .!= 0]
	lxgrid = lnodes_x[lweights .!= 0]
	lygrid = lnodes_y[lweights .!= 0]
	lzgrid = lnodes_z[lweights .!= 0]
	nl = length(lzgrid)
<<<<<<< HEAD

# 	Compute Sum
	integral_sum = 0
	result = parallel_integral_sum_lr(rweights_new, lweights_new,
	                        lxgrid, lygrid, lzgrid,
	                        nx,
	                        rnodes_new,
	                        gpu_num)

	result = 4/pi * result
	println("nr = ",nr,", nl = ",nl, " nx= ",nx)
	println("Result= ", result)

	return result
end

function evaluate_streams(nr_start::Int=10,nl_start::Int=2,nx::Int=9, n_gpus::Int=1)
	# Create the Input Data with all l's
	#initialize data
	rnodes, rweights = gausslaguerre(nr_start,1)
	rweights=making_zeros(rweights)
	rweights_new=rweights[rweights .!= 0]
	rnodes_new=rnodes[rweights .!= 0]
	nr=length(rnodes_new)


	#lebedev points
    avail=getavailablepoints()
	nl= avail[nl_start]
	lnodes_x, lnodes_y, lnodes_z, lweights =  lebedev_by_points(nl)
	lweights=making_zeros(lweights)
	lweights_new=lweights[lweights .!= 0]
	lxgrid = lnodes_x[lweights .!= 0]
	lygrid = lnodes_y[lweights .!= 0]
	lzgrid = lnodes_z[lweights .!= 0]
	l_vector = grid_to_vector(lxgrid,lygrid, lzgrid)
	nl = length(lzgrid)

	# For all r, change r in input Data and send it to c function
	rsum = [0.0 for r in rnodes_new]
    	r_ctr = Atomic{Int}(0) # Atomic variable to safely track the number of threads executed
	num_threads = nthreads()
	@threads for i in 1:nr
		rdata = create_InputDataStreams(rnodes_new[i], l_vector, lweights_new, nl, nx)
		# Sum them up with the right weights
		rsum[i] = evaluate_inner_stream(rdata, mod(i,n_gpus) )*rweights[i]
	        atomic_add!(r_ctr, 1) # Update the ctr atomically
		if ( mod(r_ctr[],5)==0 )
			println("r_i: ",i, " computed= ",r_ctr[]," / ",nr)
		end
	end
	# multiply with factor
	result = 4/pi * sum(rsum)
	# display result
	println("nr: ", nr, " nl: ", nl," nx: ",nx)
	println("result = ",result)
end


#time_function(evaluate_broadcast_parallel,2,1,9,1)

#
# nr = 550
# nl_i = 18
# gpu_num = 4
# for nx = 200:2244\=
#     time_function(evaluate_broadcast_parallel,nr,nl_i,nx,gpu_num)
# end


=======
	
	integral_sum_r = [evaluate_inner_simple(rnodes_new[i],lxgrid[l],lygrid[l],lzgrid[l]) for i in 1:nr for l in 1:nl] 
	println("nr = ",nr_start,", nl = ",nl," Result of evaluate_inner: ", sum(integral_sum_r))
end
@time evaluate_broadcast(10)
>>>>>>> 88cd7ee (lior sum)

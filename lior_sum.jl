# Set up the interface with the cuda code11i
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
function create_InputData(rval,wx,wy,wz,nx)
InputData(
	  [0.0,0.0,0.0],      #c1
	  [1.0,0.0,0.0],      #c2
	  [2.0,0.0,0.0],      #c3
	  [3.0,0.0,0.0],      #c4
	  rval,               # r
	  [wx,wy,wz],      #w
	  [-7.0,8.0],        #xrange
	  [-7.0,8.0],        #yrange
	  [-7.0,8.0],        #zrange
	  nx,              # x_axis_points
	  nx,              # y_axis_points
	  nx               # z_axis_points
	  )
end
function evaluate_broadcast(nr_start::Int=10,nl_start::Int=2,nx::Int=9)
	#initialize data
	rdata = InputData[]
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
	gpu_num = 1
	l_sum = zeros(nl,1)
	    @threads for l in 1:nl
		thread_id = threadid()
	        r_sums = zeros(length(rnodes_new)) # Initialize an array to store intermediate sums
		rdata = [create_InputData(rval,lxgrid[l],lygrid[l],lzgrid[l],nx) for rval in rnodes_new]
		r_sums .= evaluate_inner.(rdata,thread_id)
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
function evaluate_broadcast_parallel(nr_start::Int=10,nl_start::Int=2,nx::Int=9)
	#initialize data
	rdata = InputData[]
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
	#rwdata = [create_InputData(rval,lxgrid[l],lygrid[l],lzgrid[l],nx) for l in 1:nl for rval in  rnodes_new ]
	nl = length(lzgrid)
	integral_sum = 0
	result = parallel_integral_sum(rweights_new, lweights_new, lxgrid, lygrid, lzgrid, nx, rnodes_new)
	result = 4/pi * result 
	println("nr = ",nr_start,", nl = ",nl)
	return result 
end


#nv=3*(73:10:200)
#nr = 192 
#l_pick = 17
#for n in nv
#        println("Computing for n=($n)")
#        println("For n=($n), liortensors gives:",evaluate_broadcast(nr,l_pick,n))
#end


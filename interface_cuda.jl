using FastGaussQuadrature, Plots, Lebedev, TensorOperations
using Base.Threads
using ChunkSplitters

# Load the shared library
const libSlater = "/arc/project/st-greif-1/gkluhana/cuslater/libcuSlater.so"  # Update the path


# define the structs and function defined in C
struct InputData
    c::Vector{Float64}
    r::Float64
    w::Vector{Float64}
    xrange::Vector{Float64}
    yrange::Vector{Float64}
    zrange::Vector{Float64}
    x_axis_points::Int64
    z_axis_points::Int64
    y_axis_points::Int64
end
function create_InputData(rval,wx,wy,wz,nx)
InputData(
	  [0.0,0.0,0.0,     #c1
	  1.0,0.0,0.0,      #c2
	  2.0,0.0,0.0,      #c3
	  3.0,0.0,0.0],     #c4
	  rval,               # r
	  [wx,wy,wz],      #w
	  [-10.0,11.0],        #xrange
	  [-10.0,11.0],        #yrange
	  [-10.0,11.0],        #zrange
	  nx,              # x_axis_points
	  nx,              # y_axis_points
	  nx               # z_axis_points
	  )
end
struct InputDataStreams
    c::Vector{Float64}
    r::Float64
    w::Vector{Float64}
    w_wts::Vector{Float64}
    nl::Int64
    xrange::Vector{Float64}
    yrange::Vector{Float64}
    zrange::Vector{Float64}
    x_axis_points::Int64
    z_axis_points::Int64
    y_axis_points::Int64
end
function create_InputDataStreams(rval,ws,w_wts,nl,nx)
InputDataStreams(
	  [0.0,0.0,0.0,     #c1
	  1.0,0.0,0.0,      #c2
	  2.0,0.0,0.0,      #c3
	  3.0,0.0,0.0],     #c4
	  rval,               	#r
	  ws,      		#ws
	  w_wts,		#w_wts
	  nl,
	  [-10.0,11.0],        #xrange
	  [-10.0,11.0],        #yrange
	  [-10.0,11.0],        #zrange
	  nx,              # x_axis_points
	  nx,              # y_axis_points
	  nx               # z_axis_points
	  )
end
function evaluate_inner(input_data::InputData, gpu_num::Int)
    d_results_ptr = Ref{Ptr{Cdouble}}(C_NULL)
	sum_integral = ccall((:evaluateInner, libSlater), Cdouble,
				(Ptr{Cdouble},
				Cdouble,
				Ptr{Cdouble},
				Ptr{Cdouble},Ptr{Cdouble},Ptr{Cdouble},
				Cint,Cint,Cint,
				Ref{Ptr{Cdouble}},
				Cint),
				input_data.c,
				input_data.r,
				input_data.w,
				input_data.xrange,input_data.yrange,input_data.zrange,
				input_data.x_axis_points,input_data.y_axis_points,input_data.z_axis_points,
				d_results_ptr,
				gpu_num )
end
function evaluate_inner(input_data::InputData, gpu_num::Int,d_results_ptr::Ref{Ptr{Cdouble}})
	sum_integral = ccall((:evaluateInner, libSlater), Cdouble,
				(Ptr{Cdouble},
				Cdouble, 
				Ptr{Cdouble}, 
				Ptr{Cdouble},Ptr{Cdouble},Ptr{Cdouble},
				Cint,Cint,Cint, 
				Ref{Ptr{Cdouble}},
				Cint), 
				input_data.c,
				input_data.r, 
				input_data.w, 
				input_data.xrange,input_data.yrange,input_data.zrange,  
				input_data.x_axis_points,input_data.y_axis_points,input_data.z_axis_points,
				d_results_ptr,
				gpu_num )
end
function  grid_to_vector(lxgrid,lygrid, lzgrid)
	n = length(lxgrid)
	result = Vector{Float64}(undef,3n)
	for i in 1:n
		result[(i-1)*3 + 1] = lxgrid[i]
		result[(i-1)*3 + 2] = lygrid[i]
		result[(i-1)*3 + 3] = lzgrid[i]
	end
	return result
end
function preProcess(x_axis_points, y_axis_points, z_axis_points, nl, gpu_num::Int)
	total_grid_points = x_axis_points * y_axis_points*z_axis_points
	max_grids = nl; 
	num_grids = max_grids
	max_grids_ptr = Ref{Cint}(max_grids)
	num_grids_ptr = Ref{Cint}(num_grids)
	#println("num_grids before preProcess: ", num_grids)
	d_results = ccall((:preProcessIntegral, libSlater), Ptr{Cdouble},
				(Cint,
				Ptr{Cint},
				Ptr{Cint}),
				total_grid_points,
				num_grids_ptr,
				max_grids_ptr)
	#println("num_grids after preProcess: ", num_grids)
	#println("If different, abort, case not handled yet") 
	return d_results, num_grids	
end
function postProcess(d_results, nl)
	ccall((:postProcessIntegral, libSlater), Cvoid,		
		(Ptr{Cdouble}, Cint),
		d_results,nl)
end


function evaluate_inner_stream(input_data::InputDataStreams, gpu_num::Int )
	sum_integral = ccall((:evaluateInnerStreams, libSlater), Cdouble,
				(Ptr{Cdouble},
				Cdouble, 
				Ptr{Cdouble},Ptr{Cdouble}, Cint, 
				Ptr{Cdouble},Ptr{Cdouble},Ptr{Cdouble},
				Cint,Cint,Cint, 
	#			Ptr{Cdouble},
				Cint), 
				input_data.c,
				input_data.r, 
				input_data.w, input_data.w_wts, input_data.nl, 
				input_data.xrange, input_data.yrange, input_data.zrange,  
				input_data.x_axis_points,input_data.y_axis_points,input_data.z_axis_points,
	#			d_results,
				gpu_num )
end
function evaluate_inner(input_data::InputData, gpu_num::Int, result::Ptr{Float64})
	sum_integral = ccall((cuslater::evaluateInner, libSlater), Cdouble, 
				(Ptr{Cdouble},Ptr{Cdouble},Ptr{Cdouble},Ptr{Cdouble}, 
				Cdouble, 
				Ptr{Cdouble}, 
				Ptr{Cdouble},Ptr{Cdouble},Ptr{Cdouble},
				Cint,Cint,Cint, 
				Ptr{Cdouble},
				Cint), 
				input_data.c1,input_data.c2,input_data.c3,input_data.c4, 
				input_data.r, 
				input_data.w, 
				input_data.xrange,input_data.yrange,input_data.zrange,  
				input_data.x_axis_points,input_data.y_axis_points,input_data.z_axis_points,
				result,
				gpu_num)
end


function test_memory_allocation(nx,nl)
	d_results_t = Vector{Ptr{Ptr{Cdouble}}}([])
	for i in 1:20
		d_results_t_i, num_grids = preProcess(nx,nx,nx,nl,0)
		push!(d_results_t, d_results_t_i)
	end

	@threads for i in 1:20
		postProcess(d_results_t[i],nl)
	end
end

function time_evaluate(nr,nl,nx,n_gpus)
	elapsed_time = @elapsed begin
		evaluate_streams(nr,nl,nx,n_gpus)
	end
	println("elapsed time = ", elapsed_time)
end
function time_function(f, nr,nl,nx,n_gpus)
	elapsed_time = @elapsed begin
		f(nr,nl,nx,n_gpus)
	end
	println("elapsed time = ", elapsed_time)
end

function test_evaluate_streams()
	input_data = InputDataStreams(
		  [0.0,0.0,0.0,     #c1
		  1.0,0.0,0.0,      #c2
		  2.0,0.0,0.0,      #c3
		  3.0,0.0,0.0],     #c4
		  1.0,			#r
		  [0.1,0.2,0.3,0.1,0.2,0.3],	#w
		  [1.0,1.0],		#w_wts
		  2,			#nl
		  [-5.0,6.0],        #xrange
		  [-5.0,6.0],        #yrange
		  [-5.0,6.0],        #zrange
		  501,              # x_axis_points
		  501,              # y_axis_points
		  501               # z_axis_points
		  )
	sum = evaluate_inner_stream(input_data, 0)
	println("sum = ",sum)
end


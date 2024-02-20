# Load the shared library
const libSlater = "/home/gkluhana/sfw/cuslater/libcuSlater.so"  # Update the path

# define the structs and function defined in C
struct InputData
    c1::Vector{Float64}
    c2::Vector{Float64}
    c3::Vector{Float64}
    c4::Vector{Float64}
    r::Float64
    w::Vector{Float64}
    xrange::Vector{Float64}
    yrange::Vector{Float64}
    zrange::Vector{Float64}
    x_axis_points::Int64
    z_axis_points::Int64
    y_axis_points::Int64
end
function evaluate_inner(input_data::InputData)
	sum_integral = ccall((:evaluateInner, libSlater), Cdouble, 
				(Ptr{Cdouble},Ptr{Cdouble},Ptr{Cdouble},Ptr{Cdouble}, 
				Cdouble, 
				Ptr{Cdouble}, 
				Ptr{Cdouble},Ptr{Cdouble},Ptr{Cdouble},
				Cint,Cint,Cint, 
				Ptr{Cdouble}), 
				input_data.c1,input_data.c2,input_data.c3,input_data.c4, 
				input_data.r, 
				input_data.w, 
				input_data.xrange,input_data.yrange,input_data.zrange,  
				input_data.x_axis_points,input_data.y_axis_points,input_data.z_axis_points,C_NULL )
end
function evaluate_inner(input_data::InputData, result::Ptr{Float64})
	sum_integral = ccall((:evaluateInner, libSlater), Cdouble, 
				(Ptr{Cdouble},Ptr{Cdouble},Ptr{Cdouble},Ptr{Cdouble}, 
				Cdouble, 
				Ptr{Cdouble}, 
				Ptr{Cdouble},Ptr{Cdouble},Ptr{Cdouble},
				Cint,Cint,Cint, 
				Ptr{Cdouble}), 
				input_data.c1,input_data.c2,input_data.c3,input_data.c4, 
				input_data.r, 
				input_data.w, 
				input_data.xrange,input_data.yrange,input_data.zrange,  
				input_data.x_axis_points,input_data.y_axis_points,input_data.z_axis_points, result)
end



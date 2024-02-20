# Set up the interface with the cuda code
include("interface_cuda.jl")

#initialize data
data = InputData(
    [0.0,0.0,0.0],	#c1
    [1.0,0.0,0.0],	#c2
    [2.0,0.0,0.0],	#c3
    [3.0,0.0,0.0],	#c4
    1.0,               # r
    [0.1,0.2,0.3],	#w
    [-5.0,10.0],	#xrange
    [-5.0,10.0],	#yrange
    [-5.0,10.0],	#zrange
    100,              # x_axis_points
    100,              # y_axis_points
    100               # z_axis_points
)

# Vector for storing integrand evaluation
f_evals = zeros(Float64,data.x_axis_points*data.y_axis_points*data.z_axis_points)
f_evals_ptr = pointer(f_evals)

# Call the julia wrapper for the C function
integral_sum = evaluate_inner(data)
#integral_sum = evaluate_inner(data,f_evals_ptr)

# Print the result of evaluate_inner
println("Result of evaluate_inner: ", integral_sum)

using LinearAlgebra
using Dates
using Plots
using SparseArrays
using Distributions

include("truncated_svd.jl")
include("resetqr.jl")
include("cresetqr.jl")

"""
Return the rate of convergence of different trials (Figure 3.6 and 3.7)

# Arguments
- `min_rows::Integer`: minimum number of rows the generated matrix can have in each trial
- `min_cols::Integer`:minimum number of columns the generated matrix can have in each trial
- `min_density::Integer`: minimum density the generated matrix can have in each trial. It must be > 0 and < 1. We suggest to not use too small densities.
- `k::Integer`: rank of the UV decomposition
- `num_tests::Integer`: number of trials to perform
- `max_iterations::Integer`: max_iteration of CRESETQR

# Returns 
- `x::Array{Float64,1}`: vector for x axis in the plot
- `y::Array{Float64,2}`: matrix of size <num_tests> x <max_iterations>. Row i contains the rate of convergence values for the i-th trial.

# Plots
plot(x, y',  lw=3, xlabel = "iteration", ylabel = "R",legend=:false)

"""
function multiple_convergence_test(min_rows, min_cols, min_density, k, num_tests=25, max_iterations=100)

    #NOTE: all the trials must end with 1000 iterations. Therefore choose the parameters so that it will happen.
    
    plotly()

    y = zeros(num_tests, max_iterations)
    j = 1

    for i = 1:num_tests
        m = rand(min_rows:750)
        n = rand(min_cols:1000)
        scaleA = rand(1:5)
        densityA = rand(Uniform(min_density,1))

        #generate matrix A
        A = Array(sprand(Float64,m,n,densityA))*scaleA

        _, y1, _, _ = test_convergence_same_starting_point(A,k,max_iterations)

        y[j,:] .= y1[:,1]
        
        j = j + 1

    end

    x = [1:1:max_iterations;]

    #To generate charts Figure 3.6 and 3.7
    #plot(x, y',  lw=3, xlabel = "iteration", ylabel = "R",legend=:false)

    return x, y
end

"""
To generate rate of convergence, relative_error_x and relative_error_f plots

# Arguments
- `A::Array{Float64,2}`: matrix of size m x n
- `k::Integer`: rank of the UV decomposition
- `max_iterations::Integer=100`: max_iteration of CRESETQR
- `p::Integer=1`: p factor in rate of convergence

# Returns
- `x::Array{Float64,1}`: vector for x axis in the plot
- `y1::Array{Float64,1}`: rate of convergence values during the execution
- `y2::Array{Float64,1}`: relative_error_x values during the execution
- `y3::Array{Float64,1}`: relative_error_f values during the execution

# Plots
- Plot rate of convergence (Figure 3.5):
    - plot(x, y1, lw=3, xlabel = "iteration", ylabel = "R", legend=:false)
- Plot relative_error_x (Figure 3.4a):
    - plot(x, y2, lw=3, yaxis=:log10, ylabel = "log(relative_error_x)", xlabel = "iteration", legend=:false)
- Plot relative_error_f (Figure 3.4b):
    - plot(x, y3, lw=3, yaxis=:log10, ylabel = "log(relative_error_f)", xlabel = "iteration", legend=:false)
"""
function test_convergence1(A, k, max_iterations=100, p=1)
    plotly()
    #generate starting point
    V = rand(k, size(A)[2])
    solution = approx_matrix_svd(A, k)
    #exec cresetqr
    _, _, errors, residuals = cresetqr(A, k; initV=V, opt=solution, max_iterations=max_iterations, min_error = 1e-16, print_error=true, save_errors=true)
    
    #Generate the chart
    y1 = zeros((length(errors), 1))
    y2 = zeros((length(residuals), 1))
    y3 = zeros((length(residuals), 1))

    #Compute the optimal value using SVD
    optimal_f = norm(A - solution)
    x = [1:1:max(length(errors),1);]

    #Fill y1 axis (rate of convergence)
    for i = 2:length(errors)
        y1[i,1] = (errors[i] - optimal_f)/(errors[i-1] - optimal_f)^p
    end

    #fill y2 axis (relative_error_x)
    for i = 1:length(errors)
        y2[i,1] = (residuals[i])/norm(solution)
    end  

    #fill y3 axis (relative_error_f)
    for i = 1:length(errors)
        y3[i,1] = (errors[i] - optimal_f)/optimal_f
    end  

    #To generate charts

    #Plot rate of convergence (Figure 3.5)
    #plot(x, y1, lw=3, xlabel = "iteration", ylabel = "R", legend=:false)

    #plot relative_error_x (Figure 3.4a)
    #plot(x, y2, lw=3, yaxis=:log10, ylabel = "log(relative_error_x)", xlabel = "iteration", legend=:false)

    #plot relative_error_f (Figure 3.4b)
    #plot(x, y3, lw=3, yaxis=:log10, ylabel = "log(relative_error_f)", xlabel = "iteration", legend=:false)

    return x, y1, y2, y3
end

"""
To generate plot 3.8 (in the report)

# Arguments
- `A::Array{Float64,2}`: matrix of size m x n
- `k::Integer`: rank of the UV decomposition
- `max_iterations::Integer=100`: max_iteration of CRESETQR
- `p::Integer=1`: p factor in rate of convergence

# Returns
- `x::Array{Float64,1}`: denominator of rate of convergence during the execution
- `y::Array{Float64,1}`: numerator of rate of convergence during the execution

# Plots
plot(x, y, lw=3, scale=:log10, xlabel = "log(f(i-1) - f(*))", ylabel = "log(f(i) - f(*))",legend=:false)

"""
function test_convergence2(A, k, max_iterations=100, p=1)
    plotly()
    #generate starting point
    V = rand(k, size(A)[2])
    solution = approx_matrix_svd(A, k)

    #exec cresetqr
    _, _, errors, _ = cresetqr(A, k; initV=V, opt=solution, max_iterations=max_iterations, min_error = 1e-12, print_error=true, save_errors=true)
        
    #Generate the chart

    x = zeros(length(errors)-1)
    y = zeros(length(errors)-1)

    #Compute the optimal value using SVD
    optimal_f = norm(A - solution)

    #Fill y1 axis (rate of convergence)
    for i = 1:length(errors)-1
        y[i] = (errors[i+1] - optimal_f)
    end

    for i = 1:length(errors)-1
        x[i] = (errors[i] - optimal_f)^p
    end

    #To generate charts
    #plot(x, y, lw=3, scale=:log10, xlabel = "log(f(i-1) - f(*))", ylabel = "log(f(i) - f(*))",legend=:false)

    return x, y
end
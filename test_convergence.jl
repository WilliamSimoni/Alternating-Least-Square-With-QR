using LinearAlgebra
using Dates
using Plots

include("truncated_svd.jl")
include("resetqr.jl")
include("cresetqr.jl")

function test_convergence_same_starting_point(A, k, max_iterations=100)
    plotly()
    #generate starting point
    V = rand(k, size(A)[2])
    #exec resetqr and cresetqr with the same initialization
    _, _, errors = resetqr(A, k; initV=V, max_iterations=max_iterations, min_error = 1e-10, print_error=true, save_errors=true)
    _, _, errors_compact = cresetqr(A, k; initV=V, max_iterations=max_iterations, min_error = 1e-10, print_error=true, save_errors=true)
    
    #Generate the chart
    y = zeros((length(errors), 2))

    #Compute the optimal value using SVD
    optimal_f = ones(1, length(errors)) * norm(A - approx_matrix_svd(A, k))
    x = [1:1:length(errors);]

    #Fill the y axis
    y[1:end,1] = ((errors' .- optimal_f) / norm(A - approx_matrix_svd(A, k)))'
    y[1:end,2] = ((errors_compact' .- optimal_f) / norm(A - approx_matrix_svd(A, k)))'

    #To generate charts
    #log log scale
    #plot(x, y, label = ["RESETQR" "CRESETQR"], lw=3, scale=:log10, xlabel = "log(iteration)", ylabel = "log(relative error (3.2))")

    #non log log scale
    #plot(x, y, label = ["RESETQR" "CRESETQR"], lw=3, xlabel = "iteration", ylabel = "relative error (3.2)")

    return x, y
end

function test_convergence_different_starting_point(A, k, max_iterations=100)
    plotly()
    #exec resetqr and cresetqr with different initialization
    _, _, errors = resetqr(A, k; max_iterations=max_iterations, min_error = 1e-10, print_error=true, save_errors=true)
    _, _, errors_compact = cresetqr(A, k; max_iterations=max_iterations, min_error = 1e-10, print_error=true, save_errors=true)
    
    #Generate the chart
    y = zeros((length(errors), 2))

    #Compute the optimal value using SVD
    optimal_f = ones(1, length(errors)) * norm(A - approx_matrix_svd(A, k))
    x = [1:1:length(errors);]

    #Fill the y axis
    y[1:end,1] = ((errors' .- optimal_f) / norm(A - approx_matrix_svd(A, k)))'
    y[1:end,2] = ((errors_compact' .- optimal_f) / norm(A - approx_matrix_svd(A, k)))'

    #To generate charts
    #log log scale
    #plot(x, y, label = ["RESETQR" "CRESETQR"], lw=3, scale=:log10, xlabel = "log(iteration)", ylabel = "log(relative error (3.2))")
    
    #non log log scale
    #plot(x, y, label = ["RESETQR" "CRESETQR"], lw=3, xlabel = "iteration", ylabel = "relative error (3.2)")

    return x, y
end
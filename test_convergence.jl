using LinearAlgebra
using Dates
using Plots

include("truncated_svd.jl")
include("myresetqr.jl")
include("myresetqrcompact.jl")

function test_convergence(A, k, max_iterations = 100)
    _, _, errors = resetqr(A,k;max_iterations = max_iterations, print_error = false, save_errors = true)
    _, _, errors_compact = cresetqr(A,k;max_iterations = max_iterations, print_error = false, save_errors = true)
    println(length(errors), " ", length(errors_compact))
    y = zeros((length(errors),3))
    optimal_f = ones(1, length(errors)) * norm(A - approx_matrix_svd(A,k))
    x = [1:1:length(errors);]
    println(size(y[1:end,2]), size(((errors_compact' .- optimal_f)/norm(A))'))
    y[1:end,1] = ((errors' .- optimal_f)/norm(A))'
    y[1:end,2] = ((errors_compact' .- optimal_f)/norm(A))'
    y[1:end,3] = (errors' - errors_compact')'
    plot(x, y, lw = 3)
end
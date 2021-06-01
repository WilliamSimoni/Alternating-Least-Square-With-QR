using LinearAlgebra
using Dates
using Plots

include("back_substitution.jl")

"""
verifies the correctness of our back-substitution algorithm num_tests times
"""
function test_back_sub(num_tests)
    errors = 0
    min_error = 1e-13

    for i = 1:num_tests
        #generate random matrix
        rows = rand(10:100)
        columns = rand(10:rows)
        A = rand(rows,columns)
        R = triu(rand(rows,rows))
        W = zeros(rows,columns)

        back_substitution!(R, A, W)

        W_j = R \ A

        if (norm(W - W_j) > min_error)
            errors += 1
        end   

    end
    return errors
end

"""
Draws a plot that compares the time (in seconds) required by our implementation and the one built-in in Julia. 

The algorithm generates Int((max_dimension - min_dimension) / step) + 1 random matrices of increasing dimension.

At step k, the matrix shape will be (min dimension+((k - 1) * step) x min dimension+((k - 1) * step)). The algorithm terminates when
the matrix shapes reach max dimension.

"""
function test_time_back_sub(min_dimension, max_dimension; step = 15)
    plotly()
    dim = Int((max_dimension - min_dimension) / step) + 1
    x = zeros(dim)
    y = zeros(dim,2)
    for i = 0:dim-1
        x[i+1] = i * step + min_dimension

        # generating random matrices for the test
        A = rand(Int(x[i+1]),Int(x[i+1]))
        R = triu(rand(Int(x[i+1]),Int(x[i+1])))
        W = zeros(Int(x[i+1]),Int(x[i+1]))

        # performing our implementation and saving the time
        y[i+1,1] = @elapsed back_substitution!(R, A, W)

        # performing JULIA's implementation and saving the time
        y[i+1,2] = @elapsed R \ A
    end
    
    plot(x, y, label = ["Our solution" "Julia's solution"], lw = 3)
end

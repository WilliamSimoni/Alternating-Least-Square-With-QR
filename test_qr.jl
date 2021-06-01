using LinearAlgebra
using Dates
using Plots

include("qr_factorization.jl")
include("truncated_svd.jl")

"""
verifies <num_tests> times that ||Q*R - A||_2 <= min_error
"""
function test_qr_A(num_tests, min_error= 1e-13)
    errors = 0

    for i = 1:num_tests

        #generating random matrix
        rows = rand(10:100)
        columns = rand(10:rows)
        r = rand(10:columns)
        #using SVD to obtain a rank r matrix
        A = approx_matrix_svd(rand(rows,columns),r)

        #calculating QR factorization
        QR, v, u = allocate_matrices(A)
        Q = rand(rows, columns)
        qr_factorization!(A,QR,v,u)
        get_Q!(QR,Q)
        R = triu(QR)[1:columns,:]

        #checking ||QR - A||
        if (norm(Q*R - A) > min_error)
            errors += 1
        end   

    end
    return errors
end

"""
verifies <num_tests> times that ||R - R_j||_2 <= min_error where R is the R 
computed by our QR factorization while R_j is the R computed by the 
JULIA's implementation
"""
function test_qr_R(num_tests, min_error = 1e-10)
    errors = 0

    for i = 1:num_tests

        #generating random matrix
        rows = rand(10:100)
        columns = rand(10:rows)
        r = rand(10:columns)
        A = approx_matrix_svd(rand(rows,columns),r)

        #calculating QR factorization
        QR, v, u = allocate_matrices(A)
        qr_factorization!(A,QR,v,u)
        Q_j,R_j = qr(A)
        R = triu(QR)[1:columns,:]

        #checking ||R - R_j||
        if (norm(R - R_j) > min_error)
            errors += 1
        end   

    end
    return errors
end

"""
verifies Q_t_times_A() function
"""
function test_qr_multiplication(num_tests, min_error = 1e-10)
    errors = 0

    for i = 1:num_tests
        #generating random matrix
        rows = rand(10:100)
        columns = rand(10:rows)
        r = rand(10:columns)
        A = approx_matrix_svd(rand(rows,columns),r)

        #calculating QR factorization
        QR, v, u = allocate_matrices(A)
        qr_factorization!(A,QR,v,u)
        Q_j,R_j = qr(A)
        W = zeros(rows,columns)
        Q_t_times_A!(QR,A,W)
        W_j = Q_j'*A   

        #checking ||W - W_j||
        if (norm(W - W_j) > min_error)
            errors += 1
        end   

    end
    return errors
end

function qr_test(A)
    QR, v, u = allocate_matrices(A)
    qr_factorization!(A,QR,v,u)
end

"""
Draws a plot that compares the time (in seconds) required by our implementation and the one built-in in Julia. 

The algorithm generates Int((max_dimension - min_dimension) / step) + 1 random matrices of increasing dimension.

At step k, the matrix shape will be (min dimension+((k - 1) * step) x min dimension+((k - 1) * step)). The algorithm terminates when
the matrix shapes reach max dimension.

"""
function test_time_qr(min_dimension, max_dimension; step = 15)
    plotly()
    dim = Int((max_dimension - min_dimension) / step) + 1
    x = zeros(dim)
    y = zeros(dim,2)
    for i = 0:dim-1
        x[i+1] = i * step + min_dimension
        A = rand(Int(x[i+1]),Int(x[i+1]))
        y[i+1,1] = @elapsed qr_test(A)
        y[i+1,2] = @elapsed qr(A)
    end
    plot(x, y, label = ["Our QR" "Julia's QR"], lw = 3)
end
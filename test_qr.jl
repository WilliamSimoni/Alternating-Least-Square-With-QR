using LinearAlgebra
using Dates
using Plots

include("qr_factorization.jl")
include("truncated.svd.jl")

function test_qr_R(num_tests)
    errors = 0
    min_error = 1e-13

    for i = 1:num_tests
        #generate random matrix
        rows = rand(10:100)
        columns = rand(10:rows)
        rank = rand(10:columns)
        A = rand(rows,columns)

        #calculate QR factorization
        QR, v, u = allocate_matrices(A)
        qr_factorization!(A,QR,v,u)
        Q_j,R_j = qr(A)
        R = triu(QR)[1:columns,:]

        if (norm(R - R_j) > min_error)
            errors += 1
        end   

    end
    return errors
end

function test_qr_multiplication(num_tests)
    errors = 0
    min_error = 1e-13

    for i = 1:num_tests
        #generate random matrix
        rows = rand(10:100)
        columns = rand(10:rows)
        rank = rand(10:columns)
        A = rand(rows,columns)

        #calculate QR factorization
        QR, v, u = allocate_matrices(A)
        qr_factorization!(A,QR,v,u)
        Q_j,R_j = qr(A)
        W = zeros(rows,columns)
        Q_t_times_A!(QR,A,W)
        W_j = Q_j'*A   
        
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
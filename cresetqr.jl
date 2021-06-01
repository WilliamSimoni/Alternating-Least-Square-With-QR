using LinearAlgebra
using Printf

include("back_substitution.jl")
include("qr_factorization.jl")

"""
Finds UV decomposition of rank k that minimizes f(U_i,V_i) = ||A - U_iV_i||_f by alternating minimization 

# Arguments
- `A::Array{Float64,2}`: matrix of size m x n.
- `k::Integer`: the rank of the UV decomposition. k must be <= rank(A).
- `initV::Array{Float64,2}=rand(k, size(A)[2])`: initial V_0 from which the algorithm starts. V_0 must be full rank.
- `min_error::Integer=1e-8`: the algorithm will stop if f(U_i,V_i) - f(U_i+1,V_i+1) <= min_error. 
- `max_iterations::Integer=100`: the algorithm will perform at most max iteration steps.
- `print_error::Bool=true`: if it is true, the algorithm will print the error at every iteration.
- `save errors::Bool=false`: if it is true, the algorithm will return an array containing the history of the errors, besides U and V
- `opt::Array{Float64,2}=A`: optimal factorization of rank k used to compute the residuals if save_errors is true. (Used only for tests)

# Return 
- `U::Array{Float64,2}`: matrix of size m x k.
- `V::Array{Float64,2}`: matrix of size k x n.
- `errors::Array{Float64,1}`: only if print_error is true.
- `residuals::Array{Float64,1}`: only if print_error is true.
"""
function cresetqr(A, k; initV = rand(k, size(A)[2]), min_error=1e-8, max_iterations=100, print_error=true, save_errors=false, opt=A)
    (m, n) = size(A)

    # allocating matrices U and V in which the program will allocate the results
    U = zeros(m, k)
    V = zeros(k,n)
    V .= initV

    num_step = 0
    
    # number of iterations the algorithm have perfomed
    num_steps = 1

    # allocating matrices in which store QR factorization and back_substitution results
    QR_V, v_V, u_V = allocate_matrices(V')
    QR_U, v_U, u_U = allocate_matrices(U)
    W_V = zeros(n, m)
    Q_U = zeros(m, k)

    # computing the initial function value
    f = norm(A - U * V, 2)

    # allocating vectors in which store the error and the residuals during the algorithm execution
    if save_errors
        errors = zeros(max_iterations)
        residuals = zeros(max_iterations)
    end

    if print_error
        println("step\t ||A - UV||_f\t ||f_old-f||")
    end

    @views @inbounds for num_step = 1:max_iterations
        # gradient = [U*V*V' - A*V', U'*U*V - U'*A]

        # Q,R = qr(V')
        qr_factorization!(V', QR_V, v_V, u_V)

        # taking the R factor
        R = triu(QR_V)[1:k,:]

        # W = Q' * A'
        Q_t_times_A!(QR_V, A', W_V)

        # R * U' = W_V by back substitution
        back_substitution!(R, U', W_V[1:k,1:m])

        # Q,R = qr(U)
        qr_factorization!(U, QR_U, v_U, u_U)

        # taking the R factor
        R = triu(QR_U)[1:k,:]

        # taking the Q factor
        get_Q!(QR_U, Q_U)

        # R * V = Q_U' * A by back substitution
        back_substitution!(R, V, Q_U' * A)

        #additional steps for CRESETQR (w.r.t. RESETQR) 
        U .= Q_U[:,1:k]
        V .= R * V

        #computing new function value
        f_old = f
        f = norm(A - U * V, 2)
        delta = norm(f_old - f, 2)

        if print_error
            @printf("%d)\t%.13f\t%.13f\n",num_step,f,delta)
        end

        #saving function value and residual
        if save_errors
            errors[num_step] = f
            residuals[num_step] = norm(U * V - opt)
            num_steps = num_step
        end

        #Stopping criteria check
        if delta <= min_error
            if print_error
                println("Reached good approximation with ", num_step, " iterations")
            end
            break
        end
    end

    if save_errors
        return U, V, errors[1:num_steps], residuals[1:num_steps]
    else
        return U, V
    end
end
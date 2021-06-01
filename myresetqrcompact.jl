using LinearAlgebra
using Statistics
using Printf

include("back_substitution.jl")
include("qr_factorization.jl")

function cresetqr(A, k; min_error = 1e-8, max_iterations = 100, print_error=true, save_errors=false)
    (m,n) = size(A)
    U = zeros(m,k)
    V = rand(k,n)
    init_v = norm(V)
    init_u = norm(U)
    error = min_error + 1
    num_step = 0

    QR_V,v_V,u_V = allocate_matrices(V')
    QR_U,v_U,u_U = allocate_matrices(U)
    W_V = zeros(n,m)
    W_U = zeros(m,n)
    Q_U = zeros(m,k)

    f = norm(A - U*V,2)

    if save_errors
        errors = zeros(max_iterations)
    end

    if print_error
        println("step\t ||A - UV||_f\t ||f_old-f||")
    end

    num_steps = 1
    @views @inbounds for num_step = 1 : max_iterations
        #gradient = [U*V*V' - A*V', U'*U*V - U'*A]

        #U_old = U
        #V_old = V

        f_old = f

        qr_factorization!(V', QR_V, v_V, u_V)
        R = triu(QR_V)[1:k,:]
        Q_t_times_A!(QR_V,A',W_V)
        back_substitution!(R, U', W_V[1:k,1:m])

        qr_factorization!(U, QR_U, v_U, u_U)
        R = triu(QR_U)[1:k,:]
        Q_t_times_A!(QR_U,A,W_U)
        back_substitution!(R, V, W_U[1:k,1:n])

        get_Q!(QR_U,Q_U)
        U .= Q_U[:,1:k]
        V .= R * V
        #println("||A - UiVi||= ", norm(A - U*V), " | ||A|| - ||UiVi||= ", norm(A) - norm(U*V))
        #println(norm(U*V), " -- < -- ", norm(A) + norm(A - U*V))
        #println("U*V: ", U*V, " | ", norm(U*V))

        f = norm(A - U*V,2)

        delta = norm(f_old - f,2)

        if print_error
            #println(num_step, ")\t", f," \t", delta)
            @printf("%d)\t%.13f\t%.13f\n",num_step,f,delta)
        end

        if save_errors
            num_steps = num_step
            errors[num_step] = f
        end

        if delta <= min_error
            if print_error
                println("Reached good approximation with ", num_step, " iterations")
            end
            break
        end
    end

    if save_errors
        return U,V,errors[1:num_steps]
    else
        return U,V
    end
end
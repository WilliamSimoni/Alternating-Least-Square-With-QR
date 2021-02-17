using LinearAlgebra
include("back_substitution.jl")
include("qr_factorization.jl")

function resetqr(A,k)
    @time begin
        
    min_error = 1e-8

    U = 0
    V = 0

    U,V = get_U_and_V(A, k, min_error)

    end
    return U,V
end

function get_U_and_V(A, k, min_error, max_steps = 100)
    (m,n) = size(A)
    U = rand(m,k)
    V = rand(k,n)
    error = min_error + 1
    num_step = 0

    QR_V,v_V,u_V = allocate_matrices(V')
    QR_U,v_U,u_U = allocate_matrices(U)
    W_V = zeros(n,m)
    W_U = zeros(m,n)

    f = norm(A - U*V,2)

    @views while(num_step < max_steps)
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
        
        println(norm(A - U*V,2))

        f = norm(A - U*V,2)

        if norm(f_old - f,2) <= min_error
            println("reached good approximation")
            break
        end

        num_step += 1
    end

    return U,V
end
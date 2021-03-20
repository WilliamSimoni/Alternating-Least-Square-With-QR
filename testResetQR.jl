using LinearAlgebra
include("back_substitution.jl")
include("qr_factorization.jl")

function resetqr(A,k)
    @time begin
        
    min_error = 1e-16

    U = 0
    V = 0

    U,V = get_U_and_V(A, k, min_error)

    end
    return U,V
end

function get_U_and_V(A, k, min_error, max_steps = 1)
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

        f_old = norm(A - U*V,2)

        println("QR nostra")
        @time qr_factorization!(V', QR_V, v_V, u_V)
        R = triu(QR_V)[1:k,:]
        println("QR julia")
        @time Q_j,R_j = qr(V')
        #println("R ", norm(R_j - R))
        
        println("molt nostra")
        @time Q_t_times_A!(QR_V,A',W_V)
        println("molt julia")
        @time W_j = Q_j'*A'

        #println("W ", norm(W_j - W_V))

        println("back nostra")
        @time back_substitution!(R, U', W_V[1:k,1:m])
        println("back loro")
        @time U' .= R \ W_V[1:k,1:m]
        #println("U ", norm(U_j - U))

        qr_factorization!(U, QR_U, v_U, u_U)
        R = triu(QR_U)[1:k,:]
        Q_j,R_j = qr(U)
        println("R ", norm(R_j - R))
        Q_t_times_A!(QR_U,A,W_U)
        W_j = Q_j'*A

        #=for i = 1 : n
            W_U[m,i] = - W_U[m,i]
        end=#

        println("W ", norm(W_j - W_U))
        back_substitution!(R, V, W_U[1:k,1:n])
        V_j = inv(R_j)*Q_j'*A
        println("U ", norm(V_j - V))
        
        println(norm(A - U*V,2))

        #if norm(f_old - f,2) <= min_error
        #    break
        #end

        num_step += 1
    end

    return U,V
end
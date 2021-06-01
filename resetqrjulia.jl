using LinearAlgebra

"""
RESETQR using JULIA functions. Used to test the algorithm at the beginning of the project.
"""
function resetqr_julia(A,k)
    @time begin
        
    min_error = 1e-16

    U = 0
    V = 0

    U,V = get_U_and_V_julia(A, k, min_error)

    end
    return U,V
end

function get_U_and_V_julia(A, k, min_error, max_steps = 20)
    (m,n) = size(A)
    U = rand(m,k)
    V = rand(k,n)
    error = min_error + 1
    num_step = 0

    f = norm(A - U*V,2)

    while(num_step < max_steps)
        #gradient = [U*V*V' - A*V', U'*U*V - U'*A]

        f_old = norm(A - U*V,2)
        Q,R = qr(V')
        U = (inv(R)*Q'*A')'
        error = norm(A - U*V,2)
        Q,R = qr(U)
        V = inv(R)*Q'*A
        f = norm(A - U*V,2)
        
        println(norm(A - U*V,2))

        if norm(f_old - f,2) <= min_error
            break
        end

        num_step += 1
    end

    return U,V
end
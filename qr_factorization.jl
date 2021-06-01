#Thin QR implementation

using LinearAlgebra


"""
Allocate matrices and vectors for the QR factorization of A

# Arguments
- `A::Array{Float64,2}`: matrix of size m x n such that m < n

# Return

- `QR::Array{Float64,2}`: matrix of size m+1 x n
- `v::Array{Float64,1}`: vector of size m
- `u::Array{Float64,1}`: vector of size m

"""
function allocate_matrices(A)
    (m,n) = size(A)

    if m < n 
        error("m must be greater or equal than n")
    end
    
    QR = zeros(m+1,n)
    v = zeros(m)
    u = zeros(m)
    return QR,v,u
end


"""
Performs QR factorization of A

# Arguments
- `A::Array{Float64,2}`: matrix of size m x n such that m < n.
- `QR::Array{Float64,2}`: matrix of size m+1 x n. The result of the factorization is stored in QR such that:
    - the upper triangular part contains R. Therefore, R = triu(QR)[1:n,:]
    - the sub-diagonal part contains the reflectors v. Hence, V = tril(QR,-1)[2:n+1,:]
- `v::Array{Float64,1}`: vector of size m
- `u::Array{Float64,1}`: vector of size m
"""
function qr_factorization!(A, QR::Array{Float64,2}, v::Array{Float64,1}, u::Array{Float64,1} )
    (m,n) = size(A)

    #copying elements of A in R
    for j = 1 : n
        for i = 1 : m
            QR[i,j] = A[i,j]
        end
    end

    #Total Complexity: O(n(3m + 2mn)) = O(3nm + 2mn^2) = O(2n^2m) + O(nm)
    @inbounds @views for j = 1 : min(m-1,n)

        #copying j-th column of R into v
        #Complexity: O(m)
        for i = 1 : m
            v[i] = i < j ? 0 : QR[i,j]
        end

        #calculating householders
        s = norm(v)
            
        if v[j] >= 0
            s = -s
        end
        
        v[j] = v[j] - s

        norm_v = norm(v)

        #Complexity: O(m)
        @. v = v/norm_v

        #calculating R
        # R = R - 2v*(v'*R)

        t = j+1
            
        QR[j,j] = s
        @. QR[t:m, j] = 0

        #Complexity: O(mn)
        u[t:n]' .= v[j:m]' * QR[j:m, t:n]
        
        #Complexity: O(mn)
        @. QR[j:m, t:n] = QR[j:m, t:n] - (2*v[j:m])*u[t:n]'

        #copying householder reflector into QR matrix
        @. QR[j+1:m+1, j] = v[j : m]
    end

    #if m== n then QR[m+1,n]=0
    if m == n
        QR[m+1,n] = 0.0
    end

end

"""
Performs Q^t * A and stores the result in W (Q^t is Q transposed)

# Arguments
- `QR::Array{Float64,2}`: matrix of size m+1 x n. The result of the factorization is stored in QR such that:
    - the upper triangular part contains R. Therefore, R = triu(QR)[1:n,:]
    - the sub-diagonal part contains the reflectors v. Hence, V = tril(QR,-1)[2:n+1,:]
- `A::Array{Float64,2}`: matrix of size m x n.
- `W::Array{Float64,2}`:matrix of size m x n in which the result is stored.
"""
function Q_t_times_A!(QR,A,W)
    #Taking householder vectors
    @views V = tril(QR, -1)[2:end,:]
    (m,n) = size(V)

    W .= 0
    W .= A .- 2 .* V[:,1] .* (V[:,1]'*A)

    #Complexity: O(mn^2)
    num_iteration = n
    if V[m,n] == 0.0
        num_iteration -= 1
    end

    for j = 2:num_iteration
        W .= W .- 2 .* V[:,j] .* (V[:,j]'*W)
    end
end

"""
Given the QR factorization returns the Q factor

# Arguments
- `QR::Array{Float64,2}`: matrix of size m+1 x n. The result of the factorization is stored in QR such that:
    - the upper triangular part contains R. Therefore, R = triu(QR)[1:n,:]
    - the sub-diagonal part contains the reflectors v. Hence, V = tril(QR,-1)[2:n+1,:]
- `Q::Array{Float64,2}`:matrix of size m x n in which the result is stored.
"""
function get_Q!(QR, Q)
    @views V = tril(QR, -1)[2:end,:]
    (m,n) = size(V)

    #initialize Q
    @views @inbounds Q .= (I - 2 .* V[:,n] .* V[:,n]')[:,1:n]
    
    @views @inbounds for j in n-1:-1:1
        Q[j:end,j:end] .= (I - 2 .* V[:,j] .* V[:,j]')[j:end,j:end] * Q[j:end,j:end] 
    end
end
#Thin QR implementation

using LinearAlgebra

function householder!(v)
    s = norm(v)
    #computing v = x - y where y = [||x||; 0 ...]

    if s >= 0
        s = -s
    end

    v[1] = v[1] - s

    norm_v = norm(v)
    for i = 1 : length(v)
        v[i] = v[i]/norm_v
    end
    
    return s
end

function qr_factorization!(A::Array{Float64,2}, Q::Array{Float64,2}, R::Array{Float64,2}, v::Array{Float64,1})
    (m,n) = size(A)

    #copying elements of A in R
    R .= A
    
    @views for j = 1 : n
        v .= R[:, j]
        s = householder!(v[j:end])
        u = v[j:end]
        R[j:end,j:end] = R[j:end,j:end] - 2u*(u'*R[j:end,j:end])
        #R[j:end,j:end] = H*R[j:end,j:end]
        Q[j:end,j] = v[j:end]
    end

    return Q,R
end
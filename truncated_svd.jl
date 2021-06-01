using LinearAlgebra

"""
Truncated SVD to find the matrix of rank k that is the closest to A

# Arguments
- `A::Array{Float64,2}`
- `k::Array{Float64,1}`

"""
function approx_matrix_svd(A,k)
    (m,n) = size(A)
    F = svd(A)
    A_SVD = F.U[1:m,1:k] * Diagonal(F.S)[1:k,1:k] * F.V'[1:k,1:n]
    return A_SVD
end
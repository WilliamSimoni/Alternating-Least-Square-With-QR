# back-substitution jl

"""
Performs Rx = b by back_substitution

# Arguments
- `R::Array{Float64,2}`: diagonal matrix of size m x m
- `x::Array{Float64,1}`: vector of size m. The result is stored in x.
- `b::Array{Float64,1}`: vector of size m.

# Complexity

O(m^2)

"""
function back_substitution_v!(R, x, b)
    (_, m) = size(R)

    x .= 0

    # O(m^2)
    @inbounds for i = m:-1:1 
        b_i = b[i]
        for k = i + 1:m
            b_i = b_i - R[i,k] * x[k]
        end
        x[i] = b_i / R[i,i]
    end    
    
end

"""
Performs RW = V by back_substitution

# Arguments
- `R::Array{Float64,2}`: diagonal matrix of size m x m
- `V::Array{Float64,2}`: matrix of size m x k
- `W::Array{Float64,2}`: matrix of size m x k. The result is stored in W.

# Complexity

O(km^2)

"""
function back_substitution!(R,V,W)

    (_,k) = size(V)
    (m,_) = size(R)

    #Complexity: O(km^2)
    @views @inbounds for i = 1 : k
        V[:,i] .= 0
        for j = m:-1:1 
            V[j,i] = W[j,i]/R[j,j];
            @. W[1:j-1,i] = W[1:j-1,i] - R[1:j-1,j]*V[j,i];
        end    
    end

end
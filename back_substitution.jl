# back-substitution jl

function back_substitution_v!(R, x, b)
    (n, m) = size(R)

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

function back_substitution!(R,V,W)
    (m,n) = size(V)
    (k, t) = size(R)
    #back_substitution_v!(R, V[:,i], W[:,i])
    @views @inbounds for i = 1 : n
        V[:,i] .= 0
        # O(m^2)
        for j = k:-1:1 
            V[j,i] = W[j,i]/R[j,j];
            @. W[1:j-1,i] = W[1:j-1,i] - R[1:j-1,j]*V[j,i];
        end    
    end
end
# back-substitution jl

function back_substitution_v(R, x, b)
    (n, m) = size(R)

    x .= 0

    # O(m^2)
    for i = m:-1:1 
        b_i = b[i]
        for k = i + 1:m
            b_i = b_i - R[i,k] * x[k]
        end
        x[i] = b_i / R[i,i]
    end    
    
end

function back_substitution(R,V,W)
    (m,n) = size(V)
    @views for i = 1 : n
        back_substitution_v(R, V[:,i], W[:,i])
    end
end
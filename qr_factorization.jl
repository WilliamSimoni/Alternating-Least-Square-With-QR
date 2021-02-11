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

function qr_factorization!(A::Array{Float64,2}, Q::Array{Float64,2}, R::Array{Float64,2}, v::Array{Float64,1}, u::Array{Float64,1} )
    (m,n) = size(A)

    #copying elements of A in R
    R .= A

    #k = min(m,n)

    #Total Complexity: O(n(3m + 2mn)) = O(3nm + 2mn^2) = O(2n^2m) + O(nm)
    @views for j = 1 : min(m-1,n)
            
            #copying j-th column of R into v
            #Complexity: O(m)
            for i = 1 : m
                v[i] = i < j ? 0 : R[i,j]
            end

            #calculation householder
            s = norm(v)
            
            if v[j] >= 0
                s = -s
            end
        
            v[j] = v[j] - s

            norm_v = norm(v)

            #Complexity: O(m)
            for i = j : m
                v[i] = v[i]/norm_v
            end

            #calculate R
            # R = R - 2v*(v'*R)

            #Complexity: O(mn)
            for i = j : n
                sum = 0
                for t = j : m
                    sum += v[t]*R[t,i]
                end
                u[i] = sum
            end

            #Complexity: O(mn)
            for i = j : n
                for t = j : m
                    R[t,i] = R[t,i] - 2*v[t]*u[i]
                end
            end

            #Complexity: O(m)
            for i = j : m
                Q[i,j] = v[i]
            end
    end

    return Q,R
end
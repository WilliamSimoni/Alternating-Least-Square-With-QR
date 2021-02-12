#Thin QR implementation

using LinearAlgebra

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

function qr_factorization!(A::Array{Float64,2}, QR::Array{Float64,2}, v::Array{Float64,1}, u::Array{Float64,1} )
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

            #calculation householder
            s = norm(v)
            
            if v[j] >= 0
                s = -s
            end
        
            v[j] = v[j] - s

            norm_v = norm(v)

            #Complexity: O(m)
            @. v = v/norm_v

            #calculate R
            # R = R - 2v*(v'*R)

            t = j+1
            
            QR[j,j] = s
            @. QR[t:m, j] = 0

            #Complexity: O(mn)
            u[t:n]' .= v[j:m]' * QR[j:m, t:n]

            #Complexity: O(mn)
            @. QR[j:m, t:n] = QR[j:m, t:n] - (2*v[j:m])*u[t:n]'

            #Complexity: O(m)
            @. QR[j+1:m+1, j] = v[j : m]
    end

    #if m== n then we have still to compute the last reflector
    if m == n
        #copying j-th column of R into v
        #Complexity: O(m)
        for i = 1 : m
            @inbounds v[i] = i < n ? 0 : QR[i,n]
        end

        #calculation householder
        s = norm(v)
            
        if v[n] >= 0
            s = -s
        end
        
        v[n] = v[n] - s

        #Complexity: O(m)
        v[n] = v[n]/norm(v)

        QR[m+1,n] = v[n]
    end

end

function Q_t_times_A(QR,A,W)
    (m,n) = size(A)
    @views V = tril(QR, -1)[2:end,:]
    W .= A

    for j = 1:n
        println(j)
        W .= W .- 2 .* V[:,j] .* (V[:,j]'*W)
    end
end
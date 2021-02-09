#Thin QR implementation

using LinearAlgebra

function householder(x)
    norm_x = norm(x)
    #computing v = x - y where y = [||x||; 0 ...]
    #Note: I used the deep copy
    v = deepcopy(x)
    v[1] = v[1] >= 0 ? v[1] + norm_x : v[1] - norm_x
    #normalizing v
    u = v/norm(v)
    #calculating the Householder matrix
    #H = I - 2*u*u'
    return u, norm_x
end

function qr_factorization(A)
    (m,n) = size(A)
    R = deepcopy(A)
    for j = 1:n
        u_j,s_j = householder(R[j:end,j])
        R[j,j] = s_j
        R[j+1:n,j] = 0
        R[j:n, j+1:n] = R[j:m, j+1:n] - 2*u*(u_j'*A[j:end,j+1:end])
        global Q[:, j:end] = Q[:, j:end] - Q[:, j:end]*u_j*2*u_j'
    end
    return Q,R
end

function test(A)
    @time begin
        qr(A);
    end
    @time begin
        ;
    end
end
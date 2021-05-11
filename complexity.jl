function complexityRQR(m,n,k,num_iter)
    return  (2 * n * k^2 + 2/3 * k^3 + n * k + k*m*n + m * k^2 + 2 * m * k^2 + 2/3 * k^3 + m*k + k * m * n + n * k^2) * num_iter
end

#function complexityRQR(m,n,k,num_iter)
#    return  (k*m*n) * num_iter
#end

function complexityRQRTHETA(m,n,k,num_iter)
    return  (k^2 * max(m,n)) * num_iter
end
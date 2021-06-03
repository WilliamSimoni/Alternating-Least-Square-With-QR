# Alternating-Least-Square-With-QR

# The problem

The problem consists of finding a low-rank approximation of a matrix A of size m x n
defined by the following equation:
```
||A - UV||_F 
```
Where U is a matrix of size m x k and V is a matrix of size k x n.
The  approximation  is  computed  using  an  alternating  minimization  procedure: choose V = V0, compute U1 = argmin‖A−UV0‖_F, then use it to compute V1 = argmin‖A−U1V‖_F,  and  so  on  until  we  find  a  sufficiently  good  approximation.

# How to execute

You need Julia installed in your PC. Then:
  1. Open the Julia console.
  2. Exec ```include("resetqr.jl")``` for RESETQR and ```include("cresetqr.jl")``` for CRESETQR.
  3. Create  a  matrix A for  which  you  want  to  find  the  UV  decomposition.   For instance, ```A = rand(10,20);```
  4.  Call theresetqr/cresetqrfunction with the following parameters:
      - A: matrix of size m x n
      - k: the rank of the UV decomposition. Note that k <= rank(A).
      - init_V: initial V_0 from which the algorithm starts. V_0 must be full rank. ([optional] default value is rand(k, n)).
      - min_error: the algorithm will stop if ‖fold−f‖F ≤ min_error ([optional] default value is 1e−8).
      - max_iterations: the algorithm will perform at most maxiteration steps ([optional] default value is 100).
      - print_error:   if  it  is  true,  the  algorithm  will  print  the  error  at  every iteration ([optional] default value is true).

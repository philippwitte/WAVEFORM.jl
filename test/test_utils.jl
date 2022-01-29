using WAVEFORM, LinearAlgebra


# Partitioning
n = 96
partition_size = 48
overlap = 0

y = partition(n, partition_size, overlap)

@test size(y) == (partition_size, Int(n / partition_size))


# Grid interpolator
n_fine = (100, 100)
n_coarse = (50, 50)
T = Float64
(f2c,c2f) = fine2coarse(n_fine,n_coarse,T)

x = randn(100, 100)
y = randn(50, 50)

@test dot(vec(y), f2c*vec(x)) - dot(vec(x), f2c'*vec(y)) < 1e-9
@test dot(vec(y), c2f'*vec(x)) - dot(vec(x), c2f*vec(y)) < 1e-9
using WAVEFORM, LinearAlgebra

g1 = range(0, stop=1, length=100)
g2 = range(0, stop=1, length=200)
I = joLagrangeInterp1D(g1, g2, Float64)

x = randn(100)
y = randn(200)

# Dot test
@test dot(y, I*x) - dot(x, I'*y) < 1e-9
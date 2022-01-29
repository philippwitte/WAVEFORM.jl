using WAVEFORM, LinearAlgebra, Test

# Test gmres
n = 100
A = randn(n,n)
A = A'*A
x = randn(100)
b = A*x

# Solve
x0 = similar(x) * 0.0
x0, rec = FGMRES(A, b, x0; tol=1e-9)

@test norm(x - x0) / norm(x0) < 1e-4

# Solve w/ preconditioner
Ai = A^-1   # perfect preconditioner
precond(x) = Ai*x

x0 = similar(x) * 0.0
x0, rec = FGMRES(A, b, x0; tol=1e-9, precond=precond)

@test norm(x - x0) / norm(x0) < 1e-4


# Test linear solver wrapper: gmres
opts = LinSolveOpts(; tol=1e-9, maxit=1000, solver=:fgmres, precond=:identity)
solver = solvesystem(A, opts)
x0 = randn(n)
x0 = solver(b, x0, true)

@test norm(x - x0) / norm(x0) < 1e-4

# Test linear solver wrapper: lufact
opts = LinSolveOpts(; tol=1e-9, maxit=1000, solver=:lufact, precond=:identity)
solver = solvesystem(A, opts)
x0 = randn(n)
x0 = solver(b, x0, true)

@test norm(x - x0) / norm(x0) < 1e-4
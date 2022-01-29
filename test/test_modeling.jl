using WAVEFORM, LinearAlgebra, Test

###################################################################################################
# wave_base.jl

# 2D model
n = 101 .* [1;1]
d = 10.0 .*[1;1]
o = 0.0 .* [1;1]

freqs = [5.0;10.0;15.0]
t0 = 0.0
f0 = 10.0
unit = "m/s"

vel_background = 2000
L = n.*d

xsrc = 0.0:100.0:L[2]
ysrc = [0.0]
zsrc = [10.0]

# Receiver grid definition
xrec = 0:10.0:L[2]
yrec = [0.0]
zrec = [950.0]

# Model type containing the domain geometry
model = Model{Int64,Float64}(n,d,o,t0,f0,unit,freqs,xsrc,ysrc,zsrc,xrec,yrec,zrec)

@test typeof(model) == Model{Int64, Float64}
@test model.n == n
@test model.d == d
@test model.o == o

# 2D: helm2d_chen9p or helm2d_std7; 3D: helm3d_operto27 or helm3d_std9
pde_scheme = WAVEFORM.helm2d_chen9p
cut_pml = true
implicit_matrix = true
npml = [1 1; 1 1]
misfit = least_squares;
srcfreqmask = trues(length(xsrc), length(freqs))
lsopts = LinSolveOpts(; tol=1e-9, maxit=1000, solver=:lufact, precond=:identity)

# Misfit function for the objective
pdeopts = PDEopts(pde_scheme, n, d, o, cut_pml, implicit_matrix, npml, minimum, srcfreqmask, lsopts)

@test typeof(pdeopts) == PDEopts{Int64, Float64}
@test pdeopts.comp_n == n
@test pdeopts.comp_d == d
@test pdeopts.comp_o == o
@test pdeopts.srcfreqmask == srcfreqmask

###################################################################################################
# wave_util.jl

idx = index_block([1,2,3,4], 2)
@test size(idx) == (2,)
@test idx[1] == [1,2]
@test idx[2] == [3,4]

w = fwi_wavelet(freqs, t0, f0)
@test typeof(w) == Vector{ComplexF64}
@test length(w) == length(freqs)

g1, g2 = odn_to_grid(o, d, n)
@test length(g1) == n[1]
@test length(g2) == n[2]
@test g1[1] == o[1]
@test g2[1] == o[2]
@test g1[end] == o[1] + (n[1]-1)*d[1]
@test g2[end] == o[2] + (n[2]-1)*d[2]

o_, d_, n_ = grid_to_odn([g1, g2])
@test o_ == o
@test d_ == d
@test n_ == n

###################################################################################################
# helm.jl

ndims=2
n = 101 .* [1;1]
npml = npml = [1 1; 1 1]
Pext, Ppad = WAVEFORM.get_pad_ext_ops(n,npml,ndims)

x = randn((n...))
y = randn(((n + sum(npml, dims=2))...))

@test dot(vec(y), Pext*vec(x)) - dot(vec(x), Pext'*vec(y)) < 1e-9
@test dot(vec(y), Ppad*vec(x)) - dot(vec(x), Ppad'*vec(y)) < 1e-9

phys_to_comp = Pext
comp_to_phys = Ppad'


# Param to wavenum
f, df, ddf = param_to_wavenum(vel_background,freqs[1],"m/s")
f, df, ddf = param_to_wavenum(vel_background,freqs[1],"s/m")
f, df, ddf = param_to_wavenum(vel_background,freqs[1],"s2/km2")



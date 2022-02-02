using JOLI
using WAVEFORM
using PyPlot
using LinearAlgebra
using Flux, CUDA
using NeuralOperators

n = 101*[1;1]
d = 10.0*[1;1]
o = 0.0*[1;1]
v0 = 2000
freq = 20.0
λ = v0/freq

t0 = 0.0
f0 = 20.0
unit = "m/s"
xsrc = [(n.*d)[1]/2]
ysrc = [0.0]
zsrc = [100.0]
xrec = range(0.0,stop=(n.*d)[1]::Float64,length=n[1])
yrec = [0.0]
zrec = [100.0]
freqs = [freq]
nsrc = length(xsrc)*length(ysrc)*length(zsrc)
nfreq = length(freqs)
model = Model{Int64,Float64}(n,d,o,t0,f0,unit,freqs,xsrc,ysrc,zsrc,xrec,yrec,zrec)

comp_n = n
comp_d = d
comp_o = o
npml = round(Int,λ/minimum(comp_d))*[1 1; 1 1];
scheme = WAVEFORM.helm2d_chen9p
cut_pml = true
implicit_matrix = true
srcfreqmask = trues(nsrc,nfreq)
misfit = WAVEFORM.least_squares
lsopts = LinSolveOpts(solver=:lufact);
opts = PDEopts{Int64,Float64}(scheme,comp_n,comp_d,comp_o,cut_pml,implicit_matrix,npml,misfit,srcfreqmask,lsopts);

v = v0*ones(Float64,n...)
v[51:end, :] .= 3000
v = vec(v)


(H,comp_grid,T,DT_adj,P) = helmholtz_system(v,model,freq,opts)
nt = comp_grid.comp_n
q = zeros(eltype(H),tuple(nt...))
q[div.(nt,2)...] = 1.0
q = vec(q)
u = H\q
u = reshape(u,nt...)

clim = 10
imshow(real(u),vmin=-clim,vmax=clim)
savefig("waveform")

################################################################################################################

model = Chain(
    Dense(2, 64),
    FourierOperator(64=>64, (16, 16), relu),
    FourierOperator(64=>64, (16, 16), relu),
    FourierOperator(64=>64, (16, 16), relu),
    FourierOperator(64=>64, (16, 16)),
    Dense(64, 128, relu),
    Dense(128, 2)
)

# Grid
n = [121, 121]
#x = range(0.0, stop=(n[1] - 1).*d[1], length=n[1]) .* ones(1, n[2])
#y = range(0.0, stop=(n[2] - 1).*d[2], length=n[2])' .* ones(n[1], 1)

#x = reshape(x, 1, n[1], n[2], 1)
#y = reshape(y, 1, n[1], n[2], 1)
q = reshape(q, 121, 121)
x = reshape(real(q), 1, n[1], n[2], 1)
y = reshape(real(q), 1, n[1], n[2], 1)

X = cat(x, y, dims=1)
ur = reshape(real(u), 1, n[1], n[2], 1)
ui = reshape(imag(u), 1, n[1], n[2], 1)
Y = cat(ur, ui, dims=1)
Y_ = model(X)

loss(x, y) = sum(abs2, y .- model(x)) / size(x)[end]
opt = Flux.Optimiser(WeightDecay(1f-4), Flux.ADAM(1f-3))

X = X |> gpu
Y = Y |> gpu
model = model |> gpu

epochs = 500
ps = params(model)
loss_history = []
for i=1:epochs
    gs = Flux.gradient(() -> loss(X, Y), ps)
    Flux.update!(opt, ps, gs)
    global loss_history = push!(loss_history, loss(X, Y))
    print("Iter ", i, "; ", loss_history[end], "\n")
end

Y_ = model(X)
imshow(Y_[1,:,:,1], vmin=-10, vmax=10); savefig("_Y_REAL")
imshow(Y[1,:,:,1], vmin=-10, vmax=10); savefig("Y_REAL")
imshow(Y_[1,:,:,1] - Y[1,:,:,1], vmin=-1, vmax=1); savefig("Diff")

u0 = zeros(ComplexF64, length(q))
u1, res1 = FGMRES(H, q, u0; maxiter=20, precond=identity, tol=1e-2)

u0 = pre(q)#zeros(ComplexF64, length(q))
u2, res2 = FGMRES(H, q, u0; maxiter=20, precond=identity, tol=1e-2)

u0 = zeros(ComplexF64, length(q))
u3, res3 = FGMRES(H, q, u0; maxiter=20, precond=pre, tol=1e-2)

u4 = pre(u3)
subplot(2,2,1)
imshow(reshape(real(uT), 121, 121), vmin=-10, vmax=10)
subplot(2,2,2)
imshow(reshape(real(u1), 121, 121), vmin=-10, vmax=10)
subplot(2,2,3)
imshow(reshape(real(u2), 121, 121), vmin=-10, vmax=10)
subplot(2,2,4)
imshow(reshape(real(u4), 121, 121), vmin=-10, vmax=10)
savefig("comparison")

plot(res1)
plot(res2)
plot(res3)
legend(["Orig", "Apriori", "Precon"])
savefig("convergence")

u = reshape(u, 121, 121)
imshow(real.(u)); savefig("u")

function pre(x)
    xr = reshape(real(x), 1, 121, 121, 1)
    xi = reshape(imag(x), 1, 121, 121, 1)
    x = cat(xr, xi, dims=1)
    y = model(x)
    return vec(y[1,:,:,1] + im*y[2,:,:,1])
end


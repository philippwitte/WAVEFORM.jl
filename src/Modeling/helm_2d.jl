# 2D Helmoltz discretizations
# Curt Da Silva, 2016
#

export helm2d_7pt, helm2d_chen2013, param_to_wavenum

function helm2d_7pt(n,d,npml,freq,v,f0,unit::String)
# Standard 7pt discretization of
# d/dz( ex/ez du/dz ) + d/dx( ez/ex du/dx ) + ezexk^2u
#
# Usage:
#  [H,dH,ddH] = helm2d_7pt(n,d,npml,freq,v,f0)
#
# Input:
#   n     - [nz,nx] number of points in each direction (including PML)
#   d     - [dz,dx] grid spacing in each direciton (in meters)
#   npml  - [npz_bot,npz_top] - number of pml points on each edge of the domain
#           [npx_bot,npx_top]
#   freq  - frequency in Hz
#   v     - velocity (with pml included), of size [nz,nx]
#   f0    - peak frequency of source wavelet in Hz
#
# Output:
#   H     - Helmholtz matrix
#   dH    - derivative of Helmholtz matrix
#   ddH   - second derivative of Helmholtz matrix

    nz,nx = n
    Δz,Δx = d
    npz = npml[1,:]
    npx = npml[2,:]
    a0 = 1.79

    (k,dk,ddk) = param_to_wavenum(v,freq,unit)
    ex = pml_func1d(nx,npx,a0,f0,freq)
    ez = pml_func1d(nz,npz,a0,f0,freq)

    xg = 1:nx
    zg = 1:nz

    ez_up = ez(zg.+0.5)
    ez_down = ez(zg.-0.5)
    dz = 1/Δz^2*spdiagm(-1 => 1 ./ ez_down[2:end], 0 => -(1 ./ ez_down+1 ./ ez_up), 1 => 1 ./ ez_up[1:end-1])
    Dz = kron(spdiagm(ex(xg)),dz)

    ex_up = ex(xg.+0.5)
    ex_down = ex(xg.-0.5)
    dx = 1/Δx^2*spdiagm(-1 => 1 ./ ex_down[2:end], 0 => -(1 ./ ex_down+1 ./ ex_up), 1 => 1 ./ ex_up[1:end-1])
    Dx = kron(dx,spdiagm(ez(zg)))
    A = kron(spdiagm(ex(xg)),spdiagm(ez(zg)))

    H = Dz + Dx + A*spdiagm(vec(k))

    dH = A*spdiagm(vec(dk))
    ddH = A*spdiagm(vec(ddk))

  return H,dH,ddH
end


function helm2d_chen2013(n,d,npml,freq,v,f0,unit::String)
  # 9pt stencil from  Chen, et. al. "AN OPTIMAL 9-POINT FINITE DIFFERENCE
  # SCHEME FOR THE HELMHOLTZ EQUATION WITH PML", 2013
  #

    nz,nx = n
    N = nz*nx
    Δz,Δx = d
    Δz==Δx || error("z and x spacing must be identical")
    f0 > 0 || error("f0 must be positive")
    h = Δz
    npz = npml[1,:]
    npx = npml[2,:]
    a0 = 1.79

    (k,dk,ddk) = param_to_wavenum(v,freq,unit)

    vmin = minimum(vec(v))
    vmax = maximum(vec(v))
    Gmin = vmin/(h*freq)
    Gmax = vmax/(h*freq)
    IG = Gmin
    if IG >= 2.5 && IG <= 3
        b,d,e = (0.6803,0.444,0.0008)
    elseif IG >= 3 && IG <= 4
        b,d,e = (0.73427,0.4088,-0.0036)
    elseif IG >= 4 && IG <= 5
        b,d,e = (0.7840,0.3832,-0.0060)
    elseif IG >= 5 && IG <= 6
        b,d,e = (0.802,0.3712,-0.0072)
    elseif IG >= 6 && IG <= 8
        b,d,e = (0.8133,0.3637,-0.0075)
    elseif IG >= 8 && IG <= 10
        b,d,e = (0.8219,0.3578,-0.0078)
    elseif IG >= 10 && IG <= 400
        b,d,e = (0.8271,0.3540,-0.0080)
    end
    c = 1-d-e

    ex = pml_func1d(nx,npx,a0,f0,freq)
    ez = pml_func1d(nz,npz,a0,f0,freq)
    xg = 1:nx
    zg = 1:nz
    ez_up = ez(zg.+0.5)
    ez_down = ez(zg.-0.5)
    ex_up = ex(xg.+0.5)
    ex_down = ex(xg.-0.5)
    lx = 1/Δx^2*spdiagm(-1 => 1 ./ ex_down[2:end], 0 => -(1 ./ ex_down+1 ./ ex_up), 1 => 1 ./ ex_up[1:end-1])
    Lx = j->kron(lx,spdiagm(j => ez(zg.+j)[offset_range(j,nz)]))
    lz = 1/Δz^2*spdiagm(-1 => 1 ./ ez_down[2:end], 0 => -(1 ./ ez_down.+1 ./ ez_up), 1 => 1 ./ ez_up[1:end-1])
    Lz = j->kron(spdiagm(j => ex(xg.+j)[offset_range(j,nx)]),lz)

    L = b*Lx(0) + (1-b)/2*(Lx(-1) + Lx(1)) + b*Lz(0) + (1-b)/2*(Lz(-1) + Lz(1))

    # Neighbourhood notation around the point uNN
    #
    #    uMM  -- uMN -- uMP
    #             |
    #    uNM  -- uNN -- uNP
    #             |
    #    uPM  -- uPN -- uPP
    #
    #

    # Average along the points uMN, uPN,uNM,uNP
    # Eliminate points that don't have these offsets as neighbours
    eNM = ones(nz,nx)
    eNM[:,1] .= 0
    eNP = ones(nz,nx)
    eNP[:,end] .= 0
    eMN = ones(nz,nx)
    eMN[1,:] .= 0
    ePN = ones(nz,nx)
    ePN[end,:] .= 0
    I0 = 1/4 .* spdiagm(-nz => vec(eNM)[nz+1:N], nz => vec(eNP)[1:N-nz], -1 => vec(eMN)[2:N], 1 => vec(ePN)[1:N-1])

    # Average along the points uMM,uMP,uPM,uPP (corners of square above)
    # Eliminate points that don't have these offsets as neighbours
    eMM = ones(nz,nx)
    eMM[1,:] .= 0
    eMM[:,1] .= 0
    ePP = ones(nz,nx)
    ePP[end,:] .= 0
    ePP[:,end] .= 0
    ePM = ones(nz,nx)
    ePM[end,:] .= 0
    ePM[:,1] .= 0
    eMP = ones(nz,nx)
    eMP[1,:] .= 0
    eMP[:,end] .= 0

    I45 = 1/4 .* spdiagm(-nz-1 => vec(eMM)[(nz+2):N], -nz+1 => vec(ePM)[nz:N], nz-1 => vec(eMP)[1:N-nz+1], nz+1 => vec(ePP)[1:N-(nz+1)])

    A = (c*spdiagm(ones(N))+d*I0+e*I45)*kron(spdiagm(ex(xg)), spdiagm(ez(zg)))
    H = L + A*spdiagm(vec(k))

    dH = A*spdiagm(vec(dk))
    ddH = A*spdiagm(vec(ddk))

    return H,dH,ddH
end

function offset_range(j,nx)
# offset_range - Generates the range of indices to compensate for the differences
# between Matlab's spdiags and Julia's spdiagm
# j - offset in (-nx+1,nx-1)
# nx - number of points
  if j >= 0
    I = 1:nx-j
  else
    I = 1+abs(j):nx
  end
end

function pml_func1d(nx,nb,a0,f0,freq)
# Function that increases quadratically from the interior of the pml domain
# σ on p. 391 of Chen, et. al. "AN OPTIMAL 9-POINT FINITE DIFFERENCE
# SCHEME FOR THE HELMHOLTZ EQUATION WITH PML", 2013
#
  dist_from_int = x -> (x.<= nb[1]) .* (nb[1].-x) + (x .> nx.-nb[2]) .* (x.-(nx-nb[2]))
  σ = x-> 2*pi*a0*f0*(dist_from_int(x)/maximum(nb)).^2
  func = x -> 1 .- im*σ(x)/freq
end

function param_to_wavenum(v,freq,unit::String)
# Convert input parameter to rad^2*s2/m2
#
    ω2 = (2*pi*freq)^2
    if unit == "m/s"
        f = ω2*(v.^(-2))
        df = -2*ω2*(v.^(-3))
        ddf = 6*ω2*(v.^(-4))
    elseif unit == "s/m"
        f = ω2*(v.^2)
        df = 2*ω2*v
        ddf = 2*ω2*ones(size(v))
    elseif unit == "s2/m2"
        f = ω2*v
        df = zeros(size(v))
        ddf = zeros(size(v))
    elseif unit == "s2/km2"
        f = ω2*1e-6*v
        df = zeros(size(v))
        ddf = zeros(size(v))
    else
        error("Unknown unit type $unit")
    end
    return f,df,ddf
end

using WAVEFORM, SparseArrays

n = 101 .* [1;1]
d = 10.0 .*[1;1]
npml = [1 1; 1 1]
freq = 5.0
v = ones((n...)) .* 2000
f0 = 10.0
unit = "m/s"

H, dH, ddH = helm2d_7pt(n,d,npml,freq,v,f0,unit)    # 10201
H, dH, ddH = WAVEFORM.helm2d_chen2013(n,d,npml,freq,v,f0,unit)
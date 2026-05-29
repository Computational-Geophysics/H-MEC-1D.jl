#= ===========================================================================
 1D steady-state hydromechanical simple-shear solver (Julia)
 Faithful port of steady1Dg_dilatation_ratewidthloop_GOOD.m  (Gerya / Dal Zilio)

   momentum :  dSxy/dy = 0 ,  Sxy = 2*ETA*Exy            (uniform shear stress)
   pressure :  (k/ETAf) d^2 peff/dy^2 = peff/((1-phi)ETAphi) - DILP
   yield    :  Syield = (C + mu*peff)(|Exy|/eps0)^a ,  ETA = Syield/(2|Exy|)
   compaction viscosity :  ETAphi = ETA/phi
 Picard iteration between the velocity and pressure sub-problems, to
 convergence (max|dPeff| < 1e-3 Pa) for each imposed boundary velocity.

 Per imposed velocity it records:
   - SXY (uniform shear stress),
   - ysize (imposed domain height),
   - width (ACTUAL shear-zone width: where |vxs| drops below vysfract*|V|),
 and stores full profiles for selected velocities (Fig C).

 Parameters set to Table 1 of the manuscript.  PEFFTOP = Delta P_tf (Table 1).
=========================================================================== =#

using SparseArrays, LinearAlgebra, Printf, DelimitedFiles
# Read/write all data in the shared repository-root results directory.
const RESULTS_DIR = get(ENV, "HMEC_RESULTS",
                        abspath(joinpath(@__DIR__, "..", "results")))
mkpath(RESULTS_DIR); cd(RESULTS_DIR)

# ---- Table 1 parameters --------------------------------------------------
const cohesion     = 5.0e6     # C  [Pa]
const friction     = 0.3       # mu0
const dilatation   = 3.0e-3    # dilatancy coefficient (MATLAB GOOD value; adjustable)
const permeability = 1.0e-18   # k  [m^2]
const epsilonyi    = 5.0e-13   # reference strain rate eps0 [1/s]
const gammayi      = 0.03      # rate-strengthening exponent a
const ETAf         = 1.0e-3    # fluid viscosity [Pa s]
const ETAphi0      = 0.0       # >0 -> constant compaction viscosity
const porosity     = 0.01      # phi
const PEFFTOP      = 60.0e6    # peff at the drained boundary = Delta P_tf [Pa]
const vysfract     = 0.999     # velocity fraction used to measure the shear-zone width
const ysize1ms     = 0.01      # ysize [m] at |V| = 1 m/s
const Ny           = 1001
const Ny1          = Ny + 1
const tolP         = 1.0e-3    # convergence tolerance on peff [Pa] (as in MATLAB)
const niterglobal  = 500000

# velocities to sweep (log10); profiles are saved for ALL of them
const VXPOW   = collect(-1.0:-0.2:-10.0)
const VXSAVE  = Set(VXPOW)            # save every swept velocity (set a subset here to thin out)

# ---- one converged solve at imposed log-velocity vxpow -------------------
function solve_one(vxpow::Float64)
    bcupper = -10.0^vxpow; bclower = 0.0
    ysize   = ysize1ms/10.0^(vxpow/2)
    dypatch = 400.0

    # irregular grid, refined toward y = ysize
    dy = ysize/(Ny-1); b = dy/10; Dlen = ysize - b; Nn = Ny-2; F = 1.1
    for _ in 1:100; F = (1 + Dlen/b*(1 - 1/F))^(1/Nn); end
    y = zeros(Ny1); y[1] = 0.0
    for i in 2:Ny; y[i] = y[i-1] + b*F^(Ny - i); end
    y[Ny] = ysize; y[Ny1] = y[Ny] + (y[Ny] - y[Ny-1])
    yp = zeros(Ny1); yp[1] = y[1] - (y[2]-y[1])/2
    for i in 2:Ny; yp[i] = (y[i-1] + y[i])/2; end
    yp[Ny1] = y[Ny] + (y[Ny]-y[Ny-1])/2
    yvx = copy(yp)

    vxs = [bcupper + (bclower - bcupper)*(yvx[i])/ysize for i in 1:Ny1]
    ETA = zeros(Ny1); ETAphi = zeros(Ny1); SXY = zeros(Ny1)
    EXY = zeros(Ny1); SIGMAY = zeros(Ny1); DILP = zeros(Ny1)
    PEFF0 = [PEFFTOP - 0.95*PEFFTOP*exp(-0.5*((yp[i]-ysize)/dypatch)^2) for i in 1:Ny1]
    peff  = fill(PEFFTOP, Ny1)

    itused = 0
    for outer itused in 1:niterglobal
        # strength & shear viscosity (basic nodes)
        for i in Ny:-1:1
            EXY[i] = 0.5*(vxs[i+1]-vxs[i])/(yvx[i+1]-yvx[i]); exyi = EXY[i]
            sy = (cohesion + friction*(PEFF0[i]+PEFF0[i+1])/2)*(abs(EXY[i])/epsilonyi)^gammayi
            ETA[i] = sy/2/abs(exyi)
            if ETA[i] > 1e9*ETA[Ny]; ETA[i] = 1e9*ETA[Ny]; end
            SXY[i] = 2*ETA[i]*EXY[i]
        end
        # momentum: dSxy/dy = 0
        L = spzeros(Ny1,Ny1); R = zeros(Ny1)
        for i in 1:Ny1
            if i == 1;        L[i,i]=1; L[i,i+1]=1; R[i]=2*bcupper
            elseif i == Ny1;  L[i,i]=1; L[i,i-1]=1; R[i]=2*bclower
            else
                dy1=yvx[i]-yvx[i-1]; dy2=yvx[i+1]-yvx[i]; dy12=(dy1+dy2)/2
                L[i,i-1]=ETA[i-1]/dy1/dy12
                L[i,i]  =-ETA[i-1]/dy1/dy12 - ETA[i]/dy2/dy12
                L[i,i+1]=ETA[i]/dy2/dy12
            end
        end
        vxs = L \ R
        # recompute strength / viscosity / yield stress
        for i in Ny:-1:1
            EXY[i] = 0.5*(vxs[i+1]-vxs[i])/(yvx[i+1]-yvx[i]); exyi = EXY[i]
            sy = (cohesion + friction*(PEFF0[i]+PEFF0[i+1])/2)*(abs(EXY[i])/epsilonyi)^gammayi
            ETA[i] = sy/2/abs(exyi)
            if ETA[i] > 1e9*ETA[Ny]; ETA[i] = 1e9*ETA[Ny]; end
            SXY[i] = 2*ETA[i]*EXY[i]; SIGMAY[i] = sy
        end
        # compaction viscosity & dilatancy (pressure nodes)
        for i in Ny:-1:2
            exyi = (EXY[i-1]+EXY[i])/2
            if exyi < EXY[Ny]*1e-9; exyi = EXY[Ny]*1e-9; end
            sy = (cohesion + friction*PEFF0[i])*(abs(exyi)/epsilonyi)^gammayi
            ETAphi[i] = sy/2/abs(exyi)/porosity
            if ETAphi[i] > 1e9*ETAphi[Ny]; ETAphi[i] = 1e9*ETAphi[Ny]; end
            if ETAphi0 > 0; ETAphi[i] = ETAphi0/porosity; end
            DILP[i] = 2*dilatation*abs(exyi)
        end
        # pressure: (k/ETAf) peff'' - peff/((1-phi)ETAphi) = -DILP
        LP = spzeros(Ny1,Ny1); RP = zeros(Ny1)
        for i in 1:Ny1
            if i == 1;        LP[i,i]=1;  LP[i,i+1]=1; RP[i]=2*PEFFTOP
            elseif i == Ny1;  LP[i,i]=-1; LP[i,i-1]=1; RP[i]=0
            else
                dy1=yp[i]-yp[i-1]; dy2=yp[i+1]-yp[i]; dy12=(dy1+dy2)/2
                LP[i,i-1]= permeability/ETAf/dy1/dy12
                LP[i,i]  =-permeability/ETAf*(1/dy1+1/dy2)/dy12 - 1/(1-porosity)/ETAphi[i]
                LP[i,i+1]= permeability/ETAf/dy2/dy12
                RP[i]    =-DILP[i]
            end
        end
        SP = LP \ RP
        DP = SP .- PEFF0
        peff = SP
        PEFF0 .= peff
        if maximum(abs.(DP)) < tolP; break; end
    end

    # ---- shear-zone width: |vxs| transition through vysfract*|bcupper| ----
    thr = abs(bcupper*vysfract); width = NaN
    for i in Ny:-1:1                                   # no break: matches MATLAB (last crossing wins)
        if abs(vxs[i]) > thr && abs(vxs[i+1]) <= thr
            ycr = yvx[i] + (yvx[i+1]-yvx[i])*(abs(vxs[i])-thr)/(abs(vxs[i])-abs(vxs[i+1]))
            width = ysize - ycr
        end
    end
    yc = yvx[argmax(abs.(EXY[1:Ny]))]            # band centre (max strain rate)
    return (; y, yp, yvx, vxs, peff, SIGMAY, EXY, ETAphi,
              SXY=SXY[Ny], ysize, width, yc, bcupper, itused)
end

# ---- sweep & save --------------------------------------------------------
open("couette_summary.txt","w") do io
    @printf io "# vxpow   V[m/s]       ysize[m]     SXY[Pa]      width[m]     width/ysize  iters\n"
end
open("couette_profiles.txt","w") do io; write(io,"# 1D Couette profiles (selected velocities)\n"); end

@printf "%-7s %-11s %-11s %-11s %-11s %-11s %-7s\n" "vxpow" "V[m/s]" "ysize[m]" "SXY[MPa]" "width[m]" "w/ysize" "iters"
for vxpow in VXPOW
    s = solve_one(vxpow)
    V = -s.bcupper
    @printf "%-7.2f %-11.3e %-11.3e %-11.3f %-11.3e %-11.3f %-7d\n" vxpow V s.ysize s.SXY/1e6 s.width s.width/s.ysize s.itused
    open("couette_summary.txt","a") do io
        @printf io "%7.2f %12.4e %12.4e %12.4e %12.4e %12.5f %7d\n" vxpow V s.ysize s.SXY s.width s.width/s.ysize s.itused
    end
    if vxpow in VXSAVE
        rng = 1:Ny
        yphys = s.yvx[rng] .- s.yc
        yorel = s.yvx[rng] ./ s.ysize
        Vcell = [2*s.EXY[i]*(s.y[i+1]-s.y[i]) for i in rng]
        open("couette_profiles.txt","a") do io
            @printf io "# vxpow=%.1f ysize=%.6e width=%.6e yc=%.6e\n" vxpow s.ysize s.width s.yc
            @printf io "# y_phys[m]  y_over_ysize  vxs[m/s]  peff[Pa]  SIGMAY[Pa]  EXY[1/s]  eta_phi[Pas]  Vcell[m/s]\n"
            for (j,i) in enumerate(rng)
                @printf io "%.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e\n" yphys[j] yorel[j] s.vxs[i] s.peff[i] s.SIGMAY[i] s.EXY[i] s.ETAphi[i] Vcell[j]
            end
        end
    end
end
println("\nWrote couette_summary.txt and couette_profiles.txt")

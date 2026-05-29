# =============================================================================
# H-MEC 1-D  —  Hydro-Mechanical Earthquake Cycles
# =============================================================================
# Authors : Luca Dal Zilio  (Nanyang Technological University)
#           Taras Gerya     (ETH Zürich)
#
# Reference: Dal Zilio & Gerya (2025)
#
# Governing equations (staggered-grid, P-v formulation)
# -------------------------------------------------------
#   Total X-Stokes  : ∂(σ'xx)/∂x + ∂(σ'xy)/∂y − ∂Pt/∂x = −ρt gx
#   Total Y-Stokes  : ∂(σ'yx)/∂x + ∂(σ'yy)/∂y − ∂Pt/∂y = −ρt gy
#   Solid continuity: ∂Vxs/∂x + ∂Vys/∂y + (Pt−Pf)/ηbulk = 0
#   Fluid X-Darcy   : −ηf/K·VxD − ∂Pf/∂x = −ρf gx
#   Fluid Y-Darcy   : −ηf/K·VyD − ∂Pf/∂y = −ρf gy
#   Fluid continuity: ∂VxD/∂x + ∂VyD/∂y − (Pt−Pf)/ηbulk = 0
#
# =============================================================================
# UNIFIED RHEOLOGY VERSION
# =============================================================================
# Select the fault-zone constitutive law via the `rheology` flag below.
#
#   :rate_strengthening
#       Non-associated rate-dependent power-law plasticity (Yi et al., 2018)
#       as used in the original H-MEC framework (Dal Zilio et al., 2022).
#       With this flag the simulation is bit-for-bit equivalent to the
#       reference code 1_h_mec_1D.jl.
#
#   :rate_and_state
#       Dieterich–Ruina rate-and-state friction embedded in the H-MEC
#       continuum.  State evolution follows the Dieterich (1979) aging law.
#       The dynamic timestep uses the Lapusta et al. (2000) / Lapusta & Liu
#       (2009) quasi-static criterion.
#
# =============================================================================

# All input/output (EVO_*.txt, fault.txt, *.jld2 checkpoints) is read from and
# written to a single shared results directory at the repository root, so that
# the solver in src/ and the post-processing scripts in analysis/ and plotting/
# all operate on the same files. Override the location with the HMEC_RESULTS
# environment variable.
const RESULTS_DIR = get(ENV, "HMEC_RESULTS",
                        abspath(joinpath(@__DIR__, "..", "results")))
mkpath(RESULTS_DIR)
cd(RESULTS_DIR)

using SparseArrays
using Plots
using Statistics
using Printf
using JLD2
using LinearAlgebra
using SuiteSparse
using Dates


# =============================================================================
#  RHEOLOGY SELECTOR
# =============================================================================
# Choose ONE of:  :rate_strengthening  |  :rate_and_state
const rheology = :rate_and_state
# =============================================================================
const is_rsf = (rheology == :rate_and_state)
const is_rs  = (rheology == :rate_strengthening)
@assert is_rsf || is_rs "rheology must be :rate_strengthening or :rate_and_state"


# -----------------------------------------------------------------------------
#  RESTART / CLEAN CONTROL
# -----------------------------------------------------------------------------
# Set `restart_file` to `nothing` for a fresh run, or to a checkpoint path.
# NOTE: not declared `const` so you can flip between nothing and a path in the
# same Julia REPL session without triggering "invalid redefinition of constant".
# -----------------------------------------------------------------------------
restart_file = nothing            # nothing = fresh run from t = 0
#restart_file = "h_mec__0023600.jld2"

const do_restart = !(restart_file === nothing || restart_file == "")
if do_restart && !isfile(restart_file)
    error("Restart file not found: \"$restart_file\". " *
          "Set `restart_file = nothing` for a fresh run, or check the path.")
end

# Output files removed at the start of a fresh run.
# fault.txt MUST be in this list: it stores y-grid coordinates and is opened
# in append mode, so a stale copy would corrupt downstream plotting scripts.
const outfiles = ["EVO_Vslip.txt", "EVO_EIIp.txt", "EVO_viscosity.txt",
    "EVO_press_flu.txt", "EVO_press_eff.txt", "EVO_SigmaY.txt",
    "EVO_DILP.txt", "EVO_COMP.txt", "EVO_data.txt", "fault.txt"]

if !do_restart
    for file in outfiles
        if isfile(file)
            rm(file)
            println("Removed: $file")
        end
    end
    println("Fresh run from t = 0.")
else
    println("Restart mode: loading checkpoint \"", restart_file, "\".")
end

print("\033c")

# Numerical model definition
xsize, ysize = 62.5, 40000.0   # Grid size
Nx, Ny = 3, 481                # Grid steps (481)

# Banner (printed after the ANSI screen-clear so it stays visible)
println("=========================================================")
println("  Hydro-Mechanical Earthquake Cycles (H-MEC) — 1D")
println("  Fluid-saturated fault-zone simulator")
println("  Dal Zilio & Gerya - 2026")
println("=========================================================")
println("  Rheology : ", rheology)
println("  Grid     : Nx=", Nx, "  Ny=", Ny)
println("=========================================================")
println()


# =============================================================================
#  MODEL PARAMETERS
# =============================================================================
save_data = true
nname = "h_mec_"

# Staggered grid
Nx1, Ny1 = Nx + 1, Ny + 1

dx, dy = xsize / (Nx - 1), ysize / (Ny - 1)
xbeg, xend = 0.0, xsize
ybeg, yend  = 0.0, ysize

# Coordinate arrays (uniform, will be refined below)
x = collect(range(xbeg, stop=xend + dx, step=dx))
y = collect(range(ybeg, stop=yend + dy, step=dy))

# -----------------------------------------------------------------------------
#  Non-uniform y-grid (finer at fault centre)
# -----------------------------------------------------------------------------
# RS uses a 50-μm minimum step to resolve the physical shear band.
# RSF uses a coarser 1-cm step because the band collapses to one cell anyway.
b = is_rsf ? 0.0010 : 0.00005  # Minimal grid step (m)
D = ysize / 2 - b              # non-uniform region half-width (m)
N = (Ny - 1) / 2 - 1           # steps in the non-uniform half

global F = 1.1
for _ in 1:100
    global F = (1 + D / b * (1 - 1 / F))^(1 / N)
end

y[1] = ybeg
for i in 2:Int(N + 2)
    y[i] = y[i-1] + b * F^(Int(N + 2) - i)
end
for i in Int(N + 3):Ny
    y[i] = y[i-1] + b * F^(i - Int(N + 3))
end
y[Ny]  = yend
y[Ny1] = yend + (y[Ny] - y[Ny-1])

# Pressure-node grids (midpoints between velocity nodes)
xp = zeros(Nx1)
xp[1] = x[1] - (x[2] - x[1]) / 2
for j in 2:Nx;  xp[j] = (x[j-1] + x[j]) / 2;  end
xp[Nx1] = x[Nx] + (x[Nx] - x[Nx-1]) / 2

yp = zeros(Ny1)
yp[1] = y[1] - (y[2] - y[1]) / 2
for i in 2:Ny;  yp[i] = (y[i-1] + y[i]) / 2;  end
yp[Ny1] = y[Ny] + (y[Ny] - y[Ny-1]) / 2

# Staggered velocity grids
xvx, yvx = x, yp      # Vx lives at (x[j], yp[i])
xvy, yvy = xp, y      # Vy lives at (xp[j], y[i])

# Precomputed grid-step arrays (avoid repeated diff() inside loops)
dxp = diff(xp)
dyp = diff(yp)
dxn = diff(x)
dyn = diff(y)


# -----------------------------------------------------------------------------
#  Material parameters
# -----------------------------------------------------------------------------
cohes, friction   = 5.0e6, 0.3
dilatation        = 0.00015     # (RS: matches reference code)
gammadil          = 0.03
cohest, tensile   = 5.0e6, 0.3
shearmod          = 30e9        # shear modulus (Pa)
viscosity         = 1e30        # background viscosity (Pa·s)
density           = 3000.0      # solid density (kg/m³)
permeability      = 1e-18       # (m²)
BETTAFLUID        = 4e-10       # fluid compressibility (Pa⁻¹)
BETTASOLID        = 2.5e-11     # solid compressibility (Pa⁻¹)
VSLIP0            = 0.5e-9      # reference slip velocity (m/s)
epsilonyi         = VSLIP0 / 1000
gamma_min         = 0.03        # power-law exponent
viscosityf        = 1e-3        # fluid viscosity (Pa·s)
porosity          = 0.01
densityf          = 1000.0      # fluid density (kg/m³)

# Zhao & Cai (2010) dilation-angle constants (hoisted out of the inner loop)
const ZC_aa1 = 20.93;  const ZC_aa2 = 35.28;  const ZC_aa3 = 2.34
const ZC_bb1 =  0.99;  const ZC_bb2 = 44.39;  const ZC_bb3 = 0.73
const ZC_cc1 =  0.37;  const ZC_cc2 =  3.54;  const ZC_cc3 = 0.47

gx, gy  = 0.0, 0.0
PCONF   = 10e6
PTFDIFF = 60e6       # total–fluid pressure offset (Pa)

etamin, etamax = 1e-3, 1e22


# =============================================================================
#  GRID ARRAYS — INITIALISATION
# =============================================================================
RHO   = fill(density,    Ny1, Nx1)
KKK   = fill(permeability, Ny1, Nx1)
TTT   = ones(Ny1, Nx1)
SXX   = zeros(Ny1, Nx1);  SXX0 = zeros(Ny1, Nx1)
SYY   = zeros(Ny1, Nx1);  SYY0 = zeros(Ny1, Nx1)
SXY   = zeros(Ny1, Nx1)
GGG   = fill(shearmod, Ny1, Nx1)
GGGB  = fill(shearmod, Ny1, Nx1)
GGGP  = fill(shearmod, Ny1, Nx1)
COHT  = fill(cohest,   Ny1, Nx1)
FRIT  = fill(tensile,  Ny1, Nx1)
COHC  = fill(cohes,    Ny1, Nx1)
FRIC  = fill(friction, Ny1, Nx1)
DILC  = fill(dilatation, Ny1, Nx1)
AMURSF = fill(gamma_min, Ny1, Nx1)   # power-law exponent field

POR   = fill(porosity, Ny1, Nx1)

# Vx-node arrays
RHOX  = fill(density,   Ny1, Nx1)
RHOFX = fill(densityf,  Ny1, Nx1)
PORX  = fill(porosity,  Ny1, Nx1)
ETADX = fill(viscosityf / permeability, Ny1, Nx1)

# Vy-node arrays
RHOY  = fill(density,   Ny1, Nx1)
RHOFY = fill(densityf,  Ny1, Nx1)
PORY  = fill(porosity,  Ny1, Nx1)
ETADY = fill(viscosityf / permeability, Ny1, Nx1)

# Pressure initialisation
PT0  = fill(PCONF + PTFDIFF, Ny1, Nx1)
PF0  = fill(PCONF,           Ny1, Nx1)
PTF0 = similar(PT0)

# -----------------------------------------------------------------------------
#  Rate-and-state friction state variable (allocated unconditionally so the
#  checkpoint format is identical for both rheologies).
# -----------------------------------------------------------------------------
V0   = 1e-9    # reference slip rate (m/s)
ARSF = 0.012   # a-parameter
BRSF = 0.016   # b-parameter
LRSF = 0.200   # state-evolution distance L (m)
OM0  = fill(30.0, Ny, Nx)   # state variable at previous timestep
OM5  = fill(30.0, Ny, Nx)   # state variable, current plastic iteration
OM   = fill(30.0, Ny, Nx)   # working copy

# -----------------------------------------------------------------------------
#  Fault initialisation — initial stress and fluid-pressure profiles
# -----------------------------------------------------------------------------
# FIX-BUG-1: RS initial shear stress is 10 MPa (was 15 MPa in v2).
# FIX-BUG-2: RS Gaussian half-width for PF0 is ysize/10 (was ysize/20 in v2).
# The RSF path keeps its own distinct nucleation parameters unchanged.
SXY0 = zeros(Ny1, Nx1)
YNY0 = zeros(Ny, Nx)
DILP = zeros(Ny1, Nx1)

for j in 1:Nx1
    for i in 1:Ny1
        gauss_sigma = is_rsf ? (ysize / 20) : (ysize / 10)
        PF0[i, j] = PCONF + 0.3 * PTFDIFF *
            exp(-0.5 * ((yp[i] - (ybeg + yend) / 2) / gauss_sigma)^2)
        if is_rsf
            SXY0[i, j] = (2 * PCONF + 0.2 * PTFDIFF *
                exp(-0.5 * ((yp[i] - (ybeg + yend) / 2) / (ysize / 10))^2))
        else
            SXY0[i, j] = 10e6    # uniform 10 MPa background (matches v1)
        end
    end
end

# Velocity and Darcy-flux arrays
VX0   = zeros(Ny1, Nx1);  VXF0  = zeros(Ny1, Nx1)
VY0   = zeros(Ny1, Nx1);  VYF0  = zeros(Ny1, Nx1)
VSLIPB   = zeros(Ny1, Nx1)
pt       = zeros(Ny1, Nx1)
vxs      = zeros(Ny1, Nx1)
vys      = zeros(Ny1, Nx1)
pf       = zeros(Ny1, Nx1)
vxD      = zeros(Ny1, Nx1)
vyD      = zeros(Ny1, Nx1)
VIS_COMP = zeros(Ny1, Nx1)
EIIB     = zeros(Ny, Nx)
SIIB     = zeros(Ny, Nx)

# Preallocated work arrays (avoid per-iteration allocation)
ESP   = zeros(Ny, Nx);  EXY  = zeros(Ny, Nx);  DSXY = zeros(Ny, Nx)
EXX   = zeros(Ny1, Nx1); DSXX = zeros(Ny1, Nx1)
EYY   = zeros(Ny1, Nx1); DSYY = zeros(Ny1, Nx1)
EII   = zeros(Ny1, Nx1); EIIVP = zeros(Ny1, Nx1)
SII   = zeros(Ny1, Nx1); DIS   = zeros(Ny1, Nx1)

# Plasticity work arrays (preallocated once, reused every inner iteration)
AXY    = zeros(Ny, Nx)
DSY    = zeros(Ny, Nx)
YNY    = zeros(Ny, Nx)
SigmaY = zeros(Ny, Nx)
Vfault = zeros(Ny, Nx)
EII_p  = zeros(Ny, Nx)

# Sparse-system buffers
Nsys    = Nx1 * Ny1 * 6
max_nnz = Nsys * 44
LL = Vector{Int}(undef, max_nnz)
CL = Vector{Int}(undef, max_nnz)
VL = Vector{Float64}(undef, max_nnz)
R  = Vector{Float64}(undef, Nsys)
S  = zeros(Float64, Nsys)
fill!(R, 0.0)

# Boundary velocities (opposing plates)
bcupper, bclower = -1e-9, +1e-9


# =============================================================================
#  VISCOSITY ARRAYS
# =============================================================================
ETAP0   = fill(viscosity, Ny1, Nx1)
ETAB0   = viscosity ./ POR
ETA0    = fill(viscosity, Ny1, Nx1)
ETA00   = fill(viscosity, Ny1, Nx1)
ETA1    = similar(ETA0)
ETA50   = similar(ETA0)   # last plastic-correction viscosity (pre-convergence)
ETA5_arr = similar(ETA0)  # per-inner-iteration working viscosity
IETAPLB = zeros(Ny, Nx)
ETA     = fill(viscosity, Ny1, Nx1)
ETAP    = fill(viscosity, Ny1, Nx1)
ETAB    = zeros(Ny1, Nx1)


# =============================================================================
#  TIME-STEPPING PARAMETERS
# =============================================================================
global dt      = 1e8
global timesum = 0.0
dtelastic0     = 1e8
if !do_restart
    global timesum = 0.0
    global dt      = 1e8
end
dtmin, dtminbeg, dtminend, dtminiter = 1e-9, 1e-9, 1e-4, 490
dtslip  = dt
ascale  = 1e0
dt_rsf  = 1e8   # Lapusta-Liu running minimum (RSF only; initialized large)

savematstep = 200
niterglobal = 10000
ynlastmax   = 500
dtstep      = 100
vratiomax   = 0.001
dtkoef      = 1.05
dtkoefup    = 1.02
dtkoefv     = is_rsf ? 1.000 : 1.001
errmin      = is_rsf ? 1e3 : 1.0
etawt       = 0.0
syieldmin   = 1e-3
stpmax      = 1e-5
tyield      = 1
timestep    = 1
yndtdecrease = 1
maxvxy0     = 0
maxvxy      = 0


# =============================================================================
#  LOAD CHECKPOINT (restart mode only)
# =============================================================================
if do_restart
    data = load(restart_file)

    global timestep = data["timestep"] + 1
    global timesum  = data["timesum"]
    global dt       = data["dt"]

    SXX  .= data["SXX"];   SYY  .= data["SYY"];   SXY  .= data["SXY"]
    SXX0 .= data["SXX0"];  SYY0 .= data["SYY0"];  SXY0 .= data["SXY0"]
    pt   .= data["pt"];    pf   .= data["pf"]
    PT0  .= data["PT0"];   PF0  .= data["PF0"]
    ETA  .= data["ETA"];   ETA0 .= data["ETA0"];   ETA00 .= data["ETA00"]
    VSLIPB .= data["VSLIPB"]
    SIIB   .= data["SIIB"]
    SigmaY .= data["SigmaY"]
    YNY    .= data["YNY"]
    OM   .= data["OM"];    OM0  .= data["OM0"];    OM5  .= data["OM5"]
    dt_rsf  = data["dt_rsf"]

    println("Restarted from timestep ", timestep, "  (time = ", timesum, " s)")
end


# =============================================================================
#  MAIN TIME-STEPPING LOOP
# =============================================================================
start_step       = timestep
global_start_time = time()

for step in start_step:1_000_000
    global timestep = step
    loop_start_time = time()

    local dx1, dx2, dy1, dy2
    local ynpl, ddd, dtpl, ptscale, pfscale, c_err, c_err_old
    local V

    global dtmin, ETA0, ETA00, ETA5_arr, YNY0, VX0, VY0
    global SYY, SXX, SXY, SXX0, SYY0, SXY0, PT0, PF0, EII_p, SIIB
    global dtslip, ETA, dt, dtslip00, yndtdecrease, VSLIPB
    global timesum, maxvxy, maxvxy0, data_save
    global L, S
    global OM, OM0, OM5, dt_rsf

    dtslip = 1e30

    # Preserve loaded dt on the very first step of a restart
    if !do_restart || step > start_step
        dt = 1e30
    end

    # -------------------------------------------------------------------------
    #  Initialise viscosity for this timestep
    # -------------------------------------------------------------------------
    if timestep == 1
        @inbounds ETA1   .= ETA0;   @inbounds ETA  .= ETA0
        @inbounds ETA50  .= ETA0;   @inbounds ETA5_arr .= ETA0
    else
        @inbounds ETA1   .= ETA00;  @inbounds ETA  .= ETA00
        @inbounds ETA50  .= ETA00;  @inbounds ETA5_arr .= ETA00
    end

    dtslip00 = dtslip

    # Dynamic timestep selection
    if is_rsf
        # RSF: use the Lapusta-Liu running minimum from the previous step
        dt = max(dt_rsf, dtmin)
    else
        # RS: elastic ceiling, allow slow growth
        dt = max(min(dtslip, dt * dtkoefup, dtelastic0), dtmin)
    end

    yndtdecrease = 0
    dt00         = dt
    DSYLSQ       = Float64[]
    ynlast       = 0
    last_iterstep = 0

    # =========================================================================
    #  INNER PLASTIC-ITERATION LOOP
    # =========================================================================
    for iterstep in 1:niterglobal

        fill!(R, 0.0)

        etamincur = shearmod * dt * 1e-4   # effective viscosity floor

        # -- Ghost / symmetry copies of pressure --
        pt[:, 1] = pt[:, 2];  pt[:, Nx1] = pt[:, Nx]
        pt[1, :] = pt[2, :];  pt[Ny1, :] = pt[Ny, :]
        pf[:, 1] = pf[:, 2];  pf[:, Nx1] = pf[:, Nx]
        pf[1, :] = pf[2, :];  pf[Ny1, :] = pf[Ny, :]

        # -----------------------------------------------------------------
        #  Viscosity floor and plastic strain rate on BASIC nodes
        # -----------------------------------------------------------------
        for i in 1:Ny
            for j in 1:Nx
                if ETA[i, j] < etamincur
                    ETA[i, j] = etamincur
                end
                if ETA[i, j] < ETA0[i, j]
                    dxm  = (x[j]  - xp[j])  / (xp[j+1] - xp[j])
                    dym  = (y[i]  - yp[i])  / (yp[i+1] - yp[i])
                    wij   = (1-dxm)*(1-dym);  wi1j  = (1-dxm)*dym
                    wij1  = dxm*(1-dym);      wi1j1 = dxm*dym
                    SIIB[i, j] = sqrt(
                        SXY[i, j]^2 +
                        0.5*(SXX[i,j]*wij + SXX[i+1,j]*wi1j + SXX[i,j+1]*wij1 + SXX[i+1,j+1]*wi1j1)^2 +
                        0.5*(SYY[i,j]*wij + SYY[i+1,j]*wi1j + SYY[i,j+1]*wij1 + SYY[i+1,j+1]*wi1j1)^2 +
                        0.5*((-SXX[i,j]-SYY[i,j])*wij + (-SXX[i+1,j]-SYY[i+1,j])*wi1j +
                             (-SXX[i,j+1]-SYY[i,j+1])*wij1 + (-SXX[i+1,j+1]-SYY[i+1,j+1])*wi1j1)^2)
                    EIIB[i, j]   = SIIB[i,j]/2/ETA[i,j] - SIIB[i,j]/2/ETA0[i,j]
                    IETAPLB[i,j] = 1/ETA[i,j] - 1/ETA0[i,j]
                else
                    EIIB[i, j]   = 0.0
                    IETAPLB[i,j] = 0.0
                end
            end
        end

        # -----------------------------------------------------------------
        #  Viscoplastic viscosity and dilatation on PRESSURE nodes
        # -----------------------------------------------------------------
        for i in 2:Ny
            for j in 2:Nx
                IETAPL = (IETAPLB[i-1,j-1] + IETAPLB[i,j-1] +
                          IETAPLB[i-1,j]   + IETAPLB[i,j])   / 4
                if YNY0[i-1,j-1]>0 || YNY0[i,j-1]>0 || YNY0[i-1,j]>0 || YNY0[i,j]>0
                    ETAP[i,j] = 1/(1/ETAP0[i,j] + IETAPL)
                    ETAB[i,j] = 1/(1/ETAB0[i,j] + IETAPL*POR[i,j])
                else
                    ETAP[i,j] = ETAP0[i,j]
                    ETAB[i,j] = ETAB0[i,j]
                end
                ETAP[i,j] = max(ETAP[i,j], etamincur)
                if ETAB[i,j]*POR[i,j] < etamincur
                    ETAB[i,j] = etamincur / POR[i,j]
                end

                GGGB[i,j] = GGGP[i,j] / POR[i,j]

                # Zhao & Cai (2010) dilation angle
                ss3 = min(max((pt[i,j]-pf[i,j])*1e-6, 0.0), 100.0)
                aa  = ZC_aa1 + ZC_aa2*exp(-ss3/ZC_aa3)
                bb  = ZC_bb1 + ZC_bb2*exp(-ss3/ZC_bb3)
                cc  = ZC_cc1 + ZC_cc2*0.01*ss3^ZC_cc3
                dil = sin(aa*bb*(exp(-bb*dilatation)-exp(-cc*dilatation))/(cc-bb)*(pi/180))
                DILP[i,j] = 0.5*dil*(EIIB[i-1,j-1]+EIIB[i,j-1]+EIIB[i-1,j]+EIIB[i,j])
            end
        end

        # -----------------------------------------------------------------
        #  Scaling factors (improve matrix conditioning)
        # -----------------------------------------------------------------
        gggbkoef = 1.0
        ptscale  = shearmod * dt * 1e-6 / dx
        pfscale  = ptscale

        # -----------------------------------------------------------------
        #  Assemble global sparse system
        # -----------------------------------------------------------------
        GL = 1

        for j in 1:Nx1
            for i in 1:Ny1
                kp  = ((j-1)*Ny1 + (i-1))*6 + 1
                kx  = kp + 1;  ky  = kp + 2
                kpf = kp + 3;  kxf = kp + 4;  kyf = kp + 5

                # -- 5a) Vxs equation --
                if i==1 || i==Ny1 || j==1 || j==Nx || j==Nx1
                    if j==Nx1
                        LL[GL]=kx; CL[GL]=kx; VL[GL]=1; GL+=1; R[kx]=0
                    elseif i==1
                        LL[GL]=kx; CL[GL]=kx;   VL[GL]= 1; GL+=1
                        LL[GL]=kx; CL[GL]=kx+6; VL[GL]= 1; GL+=1
                        R[kx] = 2*bcupper
                    elseif i==Ny1
                        LL[GL]=kx; CL[GL]=kx;   VL[GL]= 1; GL+=1
                        LL[GL]=kx; CL[GL]=kx-6; VL[GL]= 1; GL+=1
                        R[kx] = 2*bclower
                    elseif j==1 && i>1 && i<Ny1
                        LL[GL]=kx; CL[GL]=kx;          VL[GL]= 1; GL+=1
                        LL[GL]=kx; CL[GL]=kx+Ny1*6;   VL[GL]=-1; GL+=1
                        R[kx] = 0
                    elseif j==Nx && i>1 && i<Ny1
                        LL[GL]=kx; CL[GL]=kx;          VL[GL]= 1; GL+=1
                        LL[GL]=kx; CL[GL]=kx-Ny1*6;   VL[GL]=-1; GL+=1
                        R[kx] = 0
                    end
                else
                    dx1  = xvx[j]-xvx[j-1]; dx2  = xvx[j+1]-xvx[j]; dx12 = (dx1+dx2)/2
                    dy1  = yvx[i]-yvx[i-1]; dy2  = yvx[i+1]-yvx[i]; dy12 = (dy1+dy2)/2
                    ETAXY1 = ETA[i-1,j]*dt*GGG[i-1,j]/(dt*GGG[i-1,j]+ETA[i-1,j])
                    ETAXY2 = ETA[i,j]  *dt*GGG[i,j]  /(dt*GGG[i,j]  +ETA[i,j])
                    ETAXX1 = ETAP[i,j]  *dt*GGGP[i,j]  /(dt*GGGP[i,j]  +ETAP[i,j])
                    ETAXX2 = ETAP[i,j+1]*dt*GGGP[i,j+1]/(dt*GGGP[i,j+1]+ETAP[i,j+1])
                    KXY1 = dt*GGG[i-1,j]/(dt*GGG[i-1,j]+ETA[i-1,j])
                    KXY2 = dt*GGG[i,j]  /(dt*GGG[i,j]  +ETA[i,j])
                    KXX1 = dt*GGGP[i,j]  /(dt*GGGP[i,j]  +ETAP[i,j])
                    KXX2 = dt*GGGP[i,j+1]/(dt*GGGP[i,j+1]+ETAP[i,j+1])
                    SXY1 = SXY0[i-1,j]*(1-KXY1); SXY2 = SXY0[i,j]*(1-KXY2)
                    SXX1 = SXX0[i,j]  *(1-KXX1); SXX2 = SXX0[i,j+1]*(1-KXX2)
                    dRHOdx = ((RHOX[i,j+1]*(1-PORX[i,j+1])+RHOFX[i,j+1]*PORX[i,j+1]) -
                               (RHOX[i,j-1]*(1-PORX[i,j-1])+RHOFX[i,j-1]*PORX[i,j-1]))/(2*dx12)
                    dRHOdy = ((RHOX[i+1,j]*(1-PORX[i+1,j])+RHOFX[i+1,j]*PORX[i+1,j]) -
                               (RHOX[i-1,j]*(1-PORX[i-1,j])+RHOFX[i-1,j]*PORX[i-1,j]))/(2*dy12)
                    LL[GL]=kx; CL[GL]=kxf; VL[GL]=-RHOFX[i,j]*ascale/dt; GL+=1
                    LL[GL]=kx; CL[GL]=kx;
                    VL[GL]=(-4/3*(ETAXX1/dx1+ETAXX2/dx2)/dx12
                            -(ETAXY1/dy1+ETAXY2/dy2)/dy12
                            -gx*dt*dRHOdx
                            -ascale*(RHOX[i,j]*(1-PORX[i,j])+RHOFX[i,j]*PORX[i,j])/dt); GL+=1
                    LL[GL]=kx; CL[GL]=kx-Ny1*6; VL[GL]= 4/3*ETAXX1/dx1/dx12; GL+=1
                    LL[GL]=kx; CL[GL]=kx+Ny1*6; VL[GL]= 4/3*ETAXX2/dx2/dx12; GL+=1
                    LL[GL]=kx; CL[GL]=kx-6;     VL[GL]=    ETAXY1/dy1/dy12;  GL+=1
                    LL[GL]=kx; CL[GL]=kx+6;     VL[GL]=    ETAXY2/dy2/dy12;  GL+=1
                    LL[GL]=kx; CL[GL]=ky-6;     VL[GL]= ETAXY1/dx12/dy12-2/3*ETAXX1/dx12/dy12-gx*dt*dRHOdy/4; GL+=1
                    LL[GL]=kx; CL[GL]=ky;        VL[GL]=-ETAXY2/dx12/dy12+2/3*ETAXX1/dx12/dy12-gx*dt*dRHOdy/4; GL+=1
                    LL[GL]=kx; CL[GL]=ky-6+Ny1*6;VL[GL]=-ETAXY1/dx12/dy12+2/3*ETAXX2/dx12/dy12-gx*dt*dRHOdy/4; GL+=1
                    LL[GL]=kx; CL[GL]=ky+Ny1*6;  VL[GL]= ETAXY2/dx12/dy12-2/3*ETAXX2/dx12/dy12-gx*dt*dRHOdy/4; GL+=1
                    LL[GL]=kx; CL[GL]=kp;        VL[GL]= ptscale/dx12; GL+=1
                    LL[GL]=kx; CL[GL]=kp+Ny1*6;  VL[GL]=-ptscale/dx12; GL+=1
                    R[kx] = (-ascale*(RHOX[i,j]*(1-PORX[i,j])*VX0[i,j]+RHOFX[i,j]*PORX[i,j]*VXF0[i,j])/dt
                              -(RHOX[i,j]*(1-PORX[i,j])+RHOFX[i,j]*PORX[i,j])*gx
                              -(SXX2-SXX1)/dx12-(SXY2-SXY1)/dy12)
                end

                # -- 5b) Vys equation --
                if j==1 || j==Nx1 || i==1 || i==Ny || i==Ny1
                    if i==Ny1
                        LL[GL]=ky; CL[GL]=ky; VL[GL]=1; GL+=1; R[ky]=0
                    elseif j==1
                        LL[GL]=ky; CL[GL]=ky;        VL[GL]= 1; GL+=1
                        LL[GL]=ky; CL[GL]=ky+Ny1*6; VL[GL]=-1; GL+=1; R[ky]=0
                    elseif j==Nx1
                        LL[GL]=ky; CL[GL]=ky;        VL[GL]= 1; GL+=1
                        LL[GL]=ky; CL[GL]=ky-Ny1*6; VL[GL]=-1; GL+=1; R[ky]=0
                    elseif i==1
                        LL[GL]=ky; CL[GL]=ky; VL[GL]=1; GL+=1; R[ky]=0
                    elseif i==Ny
                        LL[GL]=ky; CL[GL]=ky; VL[GL]=1; GL+=1; R[ky]=0
                    end
                else
                    dx1  = xvy[j]-xvy[j-1]; dx2  = xvy[j+1]-xvy[j]; dx12=(dx1+dx2)/2
                    dy1  = yvy[i]-yvy[i-1]; dy2  = yvy[i+1]-yvy[i]; dy12=(dy1+dy2)/2
                    ETAXY1 = ETA[i,j-1]*dt*GGG[i,j-1]/(dt*GGG[i,j-1]+ETA[i,j-1])
                    ETAXY2 = ETA[i,j]  *dt*GGG[i,j]  /(dt*GGG[i,j]  +ETA[i,j])
                    ETAYY1 = ETAP[i,j]  *dt*GGGP[i,j]  /(dt*GGGP[i,j]  +ETAP[i,j])
                    ETAYY2 = ETAP[i+1,j]*dt*GGGP[i+1,j]/(dt*GGGP[i+1,j]+ETAP[i+1,j])
                    KXY1 = dt*GGG[i,j-1]/(dt*GGG[i,j-1]+ETA[i,j-1])
                    KXY2 = dt*GGG[i,j]  /(dt*GGG[i,j]  +ETA[i,j])
                    KYY1 = dt*GGGP[i,j]  /(dt*GGGP[i,j]  +ETAP[i,j])
                    KYY2 = dt*GGGP[i+1,j]/(dt*GGGP[i+1,j]+ETAP[i+1,j])
                    SXY1 = SXY0[i,j-1]*(1-KXY1); SXY2 = SXY0[i,j]  *(1-KXY2)
                    SYY1 = SYY0[i,j]  *(1-KYY1); SYY2 = SYY0[i+1,j]*(1-KYY2)
                    dRHOdx = ((RHOY[i,j+1]*(1-PORY[i,j+1])+RHOFY[i,j+1]*PORY[i,j+1]) -
                               (RHOY[i,j-1]*(1-PORY[i,j-1])+RHOFY[i,j-1]*PORY[i,j-1]))/(2*dx12)
                    dRHOdy = ((RHOY[i+1,j]*(1-PORY[i+1,j])+RHOFY[i+1,j]*PORY[i+1,j]) -
                               (RHOY[i-1,j]*(1-PORY[i-1,j])+RHOFY[i-1,j]*PORY[i-1,j]))/(2*dy12)
                    LL[GL]=ky; CL[GL]=kyf; VL[GL]=-RHOFY[i,j]*ascale/dt; GL+=1
                    LL[GL]=ky; CL[GL]=ky;
                    VL[GL]=(-4/3*(ETAYY1/dy1+ETAYY2/dy2)/dy12
                            -(ETAXY1/dx1+ETAXY2/dx2)/dx12
                            -gy*dt*dRHOdy
                            -ascale*(RHOY[i,j]*(1-PORY[i,j])+RHOFY[i,j]*PORY[i,j])/dt); GL+=1
                    LL[GL]=ky; CL[GL]=ky-Ny1*6; VL[GL]=ETAXY1/dx1/dx12; GL+=1
                    LL[GL]=ky; CL[GL]=ky+Ny1*6; VL[GL]=ETAXY2/dx2/dx12; GL+=1
                    LL[GL]=ky; CL[GL]=ky-6;     VL[GL]=4/3*ETAYY1/dy1/dy12; GL+=1
                    LL[GL]=ky; CL[GL]=ky+6;     VL[GL]=4/3*ETAYY2/dy2/dy12; GL+=1
                    LL[GL]=ky; CL[GL]=kx-Ny1*6;    VL[GL]= ETAXY1/dx12/dy12-2/3*ETAYY1/dx12/dy12-gy*dt*dRHOdx/4; GL+=1
                    LL[GL]=ky; CL[GL]=kx+6-Ny1*6;  VL[GL]=-ETAXY1/dx12/dy12+2/3*ETAYY2/dx12/dy12-gy*dt*dRHOdx/4; GL+=1
                    LL[GL]=ky; CL[GL]=kx;           VL[GL]=-ETAXY2/dx12/dy12+2/3*ETAYY1/dx12/dy12-gy*dt*dRHOdx/4; GL+=1
                    LL[GL]=ky; CL[GL]=kx+6;         VL[GL]= ETAXY2/dx12/dy12-2/3*ETAYY2/dx12/dy12-gy*dt*dRHOdx/4; GL+=1
                    LL[GL]=ky; CL[GL]=kp;   VL[GL]= ptscale/dy12; GL+=1
                    LL[GL]=ky; CL[GL]=kp+6; VL[GL]=-ptscale/dy12; GL+=1
                    R[ky] = (-ascale*(RHOY[i,j]*(1-PORY[i,j])*VY0[i,j]+RHOFY[i,j]*PORY[i,j]*VYF0[i,j])/dt
                              -(RHOY[i,j]*(1-PORY[i,j])+RHOFY[i,j]*PORY[i,j])*gy
                              -(SYY2-SYY1)/dy12-(SXY2-SXY1)/dx12)
                end

                # -- 5c) Pt equation --
                if i==1 || j==1 || i==Ny1 || j==Nx1
                    LL[GL]=kp; CL[GL]=kp; VL[GL]=1; GL+=1; R[kp]=0
                else
                    dx1 = x[j]-x[j-1];  dy1 = y[i]-y[i-1]
                    BETTADRAINED = (1/GGGB[i,j]+BETTASOLID)/(1-POR[i,j])
                    KBW = 1 - BETTASOLID/BETTADRAINED
                    LL[GL]=kp; CL[GL]=kx-Ny1*6; VL[GL]=-1/dx1;  GL+=1
                    LL[GL]=kp; CL[GL]=kx;        VL[GL]= 1/dx1;  GL+=1
                    LL[GL]=kp; CL[GL]=ky-6;      VL[GL]=-1/dy1;  GL+=1
                    LL[GL]=kp; CL[GL]=ky;         VL[GL]= 1/dy1;  GL+=1
                    LL[GL]=kp; CL[GL]=kp;
                    VL[GL]=ptscale*(1/ETAB[i,j]/(1-POR[i,j])+gggbkoef*BETTADRAINED/dt); GL+=1
                    LL[GL]=kp; CL[GL]=kpf;
                    VL[GL]=-pfscale*(1/ETAB[i,j]/(1-POR[i,j])+gggbkoef*BETTADRAINED*KBW/dt); GL+=1
                    R[kp] = gggbkoef*BETTADRAINED*(PT0[i,j]-KBW*PF0[i,j])/dt + DILP[i,j]
                end

                # -- 5d) VxD equation --
                if i==1 || i==Ny1 || j==1 || j==Nx || j==Nx1
                    if j==Nx1
                        LL[GL]=kxf; CL[GL]=kxf; VL[GL]=1; GL+=1; R[kxf]=0
                    elseif i==1
                        LL[GL]=kxf; CL[GL]=kxf;   VL[GL]= 1; GL+=1
                        LL[GL]=kxf; CL[GL]=kxf+6; VL[GL]=-1; GL+=1; R[kxf]=0
                    elseif i==Ny1
                        LL[GL]=kxf; CL[GL]=kxf;   VL[GL]= 1; GL+=1
                        LL[GL]=kxf; CL[GL]=kxf-6; VL[GL]=-1; GL+=1; R[kxf]=0
                    elseif j==1
                        LL[GL]=kxf; CL[GL]=kxf; VL[GL]=1; GL+=1; R[kxf]=0
                    elseif j==Nx
                        LL[GL]=kxf; CL[GL]=kxf; VL[GL]=1; GL+=1; R[kxf]=0
                    end
                else
                    dx1 = xp[j+1]-xp[j]
                    LL[GL]=kxf; CL[GL]=kxf;
                    VL[GL]=-ETADX[i,j]-RHOFX[i,j]/PORX[i,j]*ascale/dt; GL+=1
                    LL[GL]=kxf; CL[GL]=kx;
                    VL[GL]=-RHOFX[i,j]*ascale/dt; GL+=1
                    LL[GL]=kxf; CL[GL]=kpf;        VL[GL]= pfscale/dx1; GL+=1
                    LL[GL]=kxf; CL[GL]=kpf+Ny1*6;  VL[GL]=-pfscale/dx1; GL+=1
                    R[kxf] = -RHOFX[i,j]*(ascale*VXF0[i,j]/dt+gx)
                end

                # -- 5e) VyD equation --
                if j==1 || j==Nx1 || i==1 || i==Ny || i==Ny1
                    if i==Ny1
                        LL[GL]=kyf; CL[GL]=kyf; VL[GL]=1; GL+=1; R[kyf]=0
                    elseif j==1 && i>1 && i<Ny
                        LL[GL]=kyf; CL[GL]=kyf;        VL[GL]= 1; GL+=1
                        LL[GL]=kyf; CL[GL]=kyf+Ny1*6; VL[GL]=-1; GL+=1; R[kyf]=0
                    elseif j==Nx1 && i>1 && i<Ny
                        LL[GL]=kyf; CL[GL]=kyf;        VL[GL]= 1; GL+=1
                        LL[GL]=kyf; CL[GL]=kyf-Ny1*6; VL[GL]=-1; GL+=1; R[kyf]=0
                    elseif i==1
                        LL[GL]=kyf; CL[GL]=kyf; VL[GL]=1; GL+=1; R[kyf]=0
                    elseif i==Ny
                        LL[GL]=kyf; CL[GL]=kyf; VL[GL]=1; GL+=1; R[kyf]=0
                    end
                else
                    dy1 = yp[i+1]-yp[i]
                    LL[GL]=kyf; CL[GL]=kyf;
                    VL[GL]=-ETADY[i,j]-RHOFY[i,j]/PORY[i,j]*ascale/dt; GL+=1
                    LL[GL]=kyf; CL[GL]=ky;
                    VL[GL]=-RHOFY[i,j]*ascale/dt; GL+=1
                    LL[GL]=kyf; CL[GL]=kpf;   VL[GL]= pfscale/dy1; GL+=1
                    LL[GL]=kyf; CL[GL]=kpf+6; VL[GL]=-pfscale/dy1; GL+=1
                    R[kyf] = -RHOFY[i,j]*(ascale*VYF0[i,j]/dt+gy)
                end

                # -- 5f) Pf equation --
                if i==1 || j==1 || i==Ny1 || j==Nx1
                    LL[GL]=kpf; CL[GL]=kpf; VL[GL]=1; GL+=1; R[kpf]=0
                elseif i==2 && (j>1 && j<Nx1)
                    LL[GL]=kpf; CL[GL]=kpf; VL[GL]=-1*pfscale; GL+=1
                    LL[GL]=kpf; CL[GL]=kp;  VL[GL]= 1*ptscale; GL+=1
                    R[kpf] = PTFDIFF
                elseif i==Ny && (j>1 && j<Nx1)
                    LL[GL]=kpf; CL[GL]=kpf; VL[GL]=-1*pfscale; GL+=1
                    LL[GL]=kpf; CL[GL]=kp;  VL[GL]= 1*ptscale; GL+=1
                    R[kpf] = PTFDIFF
                else
                    dx1 = x[j]-x[j-1];  dy1 = y[i]-y[i-1]
                    BETTADRAINED = (1/GGGB[i,j]+BETTASOLID)/(1-POR[i,j])
                    KBW = 1 - BETTASOLID/BETTADRAINED
                    KSK = (BETTADRAINED-BETTASOLID)/(BETTADRAINED-BETTASOLID+POR[i,j]*(BETTAFLUID-BETTASOLID))
                    LL[GL]=kpf; CL[GL]=kxf-Ny1*6; VL[GL]=-1/dx1; GL+=1
                    LL[GL]=kpf; CL[GL]=kxf;        VL[GL]= 1/dx1; GL+=1
                    LL[GL]=kpf; CL[GL]=kyf-6;      VL[GL]=-1/dy1; GL+=1
                    LL[GL]=kpf; CL[GL]=kyf;         VL[GL]= 1/dy1; GL+=1
                    LL[GL]=kpf; CL[GL]=kp;
                    VL[GL]=-ptscale*(1/ETAB[i,j]/(1-POR[i,j])+gggbkoef*BETTADRAINED*KBW/dt); GL+=1
                    LL[GL]=kpf; CL[GL]=kpf;
                    VL[GL]= pfscale*(1/ETAB[i,j]/(1-POR[i,j])+gggbkoef*BETTADRAINED*KBW/KSK/dt); GL+=1
                    R[kpf] = -gggbkoef*BETTADRAINED*KBW*(PT0[i,j]-1/KSK*PF0[i,j])/dt - DILP[i,j]
                end
            end  # i
        end  # j

        # -----------------------------------------------------------------
        #  Solve sparse system  (SuiteSparse LU)
        # -----------------------------------------------------------------
        nnz = GL - 1
        L   = sparse(view(LL,1:nnz), view(CL,1:nnz), view(VL,1:nnz), Nsys, Nsys)
        S  .= L \ R

        # Reload solution
        for j in 1:Nx1, i in 1:Ny1
            kp  = ((j-1)*Ny1 + (i-1))*6 + 1
            pt[i,j]  = S[kp]*ptscale
            vxs[i,j] = S[kp+1]
            vys[i,j] = S[kp+2]
            pf[i,j]  = S[kp+3]*pfscale
            vxD[i,j] = S[kp+4]
            vyD[i,j] = S[kp+5]
        end

        global dt0 = dt
        global yn  = 0

        # -----------------------------------------------------------------
        #  Strain rates and stresses on BASIC nodes
        # -----------------------------------------------------------------
        @inbounds for i in 1:Ny, j in 1:Nx
            ESP[i,j] = 0.5*((vys[i,j+1]-vys[i,j])/dxp[j] - (vxs[i+1,j]-vxs[i,j])/dyp[i])
            EXY[i,j] = 0.5*((vxs[i+1,j]-vxs[i,j])/dyp[i] + (vys[i,j+1]-vys[i,j])/dxp[j])
            KXY = dt*GGG[i,j]/(dt*GGG[i,j]+ETA[i,j])
            SXY[i,j] = 2*ETA[i,j]*EXY[i,j]*KXY + SXY0[i,j]*(1-KXY)
            DSXY[i,j] = SXY[i,j]-SXY0[i,j]
        end

        # Strain rates and stresses on PRESSURE nodes
        @inbounds for i in 2:Ny, j in 2:Nx
            EXX[i,j] = (2*(vxs[i,j]-vxs[i,j-1])/dxn[j-1] - (vys[i,j]-vys[i-1,j])/dyn[i-1])/3
            EYY[i,j] = (2*(vys[i,j]-vys[i-1,j])/dyn[i-1] - (vxs[i,j]-vxs[i,j-1])/dxn[j-1])/3
            KXX = dt*GGGP[i,j]/(dt*GGGP[i,j]+ETAP[i,j])
            SXX[i,j] = 2*ETAP[i,j]*EXX[i,j]*KXX + SXX0[i,j]*(1-KXX)
            SYY[i,j] = 2*ETAP[i,j]*EYY[i,j]*KXX + SYY0[i,j]*(1-KXX)
            DSXX[i,j] = SXX[i,j]-SXX0[i,j]
            DSYY[i,j] = SYY[i,j]-SYY0[i,j]
        end

        # Symmetry / ghost copies
        pt[:,  [1,Nx1]] .= pt[:,  [2,Nx]];  pt[[1,Ny1],:] .= pt[[2,Ny],:]
        pf[:,  [1,Nx1]] .= pf[:,  [2,Nx]];  pf[[1,Ny1],:] .= pf[[2,Ny],:]
        EXX[:, [1,Nx1]] .= EXX[:, [2,Nx]];  EXX[[1,Ny1],:] .= EXX[[2,Ny],:]
        SXX[:, [1,Nx1]] .= SXX[:, [2,Nx]];  SXX[[1,Ny1],:] .= SXX[[2,Ny],:]
        SXX0[:,[1,Nx1]] .= SXX0[:,[2,Nx]];  SXX0[[1,Ny1],:] .= SXX0[[2,Ny],:]
        EYY[:, [1,Nx1]] .= EYY[:, [2,Nx]];  EYY[[1,Ny1],:] .= EYY[[2,Ny],:]
        SYY[:, [1,Nx1]] .= SYY[:, [2,Nx]];  SYY[[1,Ny1],:] .= SYY[[2,Ny],:]
        SYY0[:,[1,Nx1]] .= SYY0[:,[2,Nx]];  SYY0[[1,Ny1],:] .= SYY0[[2,Ny],:]
        ETAP[:,[1,Nx1]] .= ETAP[:,[2,Nx]];  ETAP[[1,Ny1],:] .= ETAP[[2,Ny],:]
        ETAB[:,[1,Nx1]] .= ETAB[:,[2,Nx]];  ETAB[[1,Ny1],:] .= ETAB[[2,Ny],:]
        GGGP[:,[1,Nx1]] .= GGGP[:,[2,Nx]];  GGGP[[1,Ny1],:] .= GGGP[[2,Ny],:]
        GGGB[:,[1,Nx1]] .= GGGB[:,[2,Nx]];  GGGB[[1,Ny1],:] .= GGGB[[2,Ny],:]

        # Stress / strain invariants and dissipation on pressure nodes
        @inbounds for i in 2:Ny, j in 2:Nx
            EXY2 = (EXY[i,j]^2+EXY[i-1,j]^2+EXY[i,j-1]^2+EXY[i-1,j-1]^2)*0.25
            EII[i,j] = sqrt(0.5*(EXX[i,j]^2+EYY[i,j]^2)+EXY2)
            sxy_avg2 = ((SXY[i,j]/(2*ETA[i,j]))^2+(SXY[i-1,j]/(2*ETA[i-1,j]))^2+
                        (SXY[i,j-1]/(2*ETA[i,j-1]))^2+(SXY[i-1,j-1]/(2*ETA[i-1,j-1]))^2)*0.25
            EIIVP[i,j] = sqrt(0.5*((SXX[i,j]/2/ETAP[i,j])^2+(SYY[i,j]/2/ETAP[i,j])^2)+sxy_avg2)
            SXY2 = (SXY[i,j]^2+SXY[i-1,j]^2+SXY[i,j-1]^2+SXY[i-1,j-1]^2)*0.25
            SII[i,j] = sqrt(0.5*(SXX[i,j]^2+SYY[i,j]^2)+SXY2)
            disxy = (SXY[i,j]^2/2/ETA[i,j]+SXY[i-1,j]^2/2/ETA[i-1,j]+
                     SXY[i,j-1]^2/2/ETA[i,j-1]+SXY[i-1,j-1]^2/2/ETA[i-1,j-1])*0.25
            DIS[i,j] = SXX[i,j]^2/2/ETAP[i,j]+SYY[i,j]^2/2/ETAP[i,j]+2*disxy
        end

        # -----------------------------------------------------------------
        #  Plasticity solver — reset work arrays
        # -----------------------------------------------------------------
        fill!(AXY,0.0); fill!(DSY,0.0); fill!(YNY,0.0)
        fill!(SigmaY,0.0); fill!(Vfault,0.0); fill!(EII_p,0.0)
        ynpl = 0;  ddd = 0.0

        # FIX-BUG-3 (RS path): ETA5_arr mirrors v1's `ETA5 = copy(ETA0)` reset.
        # It is reset to ETA0 at the start of every inner iteration so that
        # non-yielding nodes always recover their background viscosity.
        if is_rs
            @inbounds ETA5_arr .= ETA0
        end

        if timestep > tyield
            for i in 1:Ny
                for j in 1:Nx
                    dxm  = (x[j]-xp[j])/(xp[j+1]-xp[j])
                    dym  = (y[i]-yp[i])/(yp[i+1]-yp[i])
                    wij   = (1-dxm)*(1-dym);  wi1j  = (1-dxm)*dym
                    wij1  = dxm*(1-dym);      wi1j1 = dxm*dym
                    SXX_avg = wij*SXX[i,j]+wi1j*SXX[i+1,j]+wij1*SXX[i,j+1]+wi1j1*SXX[i+1,j+1]
                    SYY_avg = wij*SYY[i,j]+wi1j*SYY[i+1,j]+wij1*SYY[i,j+1]+wi1j1*SYY[i+1,j+1]
                    SIIB[i,j] = sqrt(SXY[i,j]^2 + 0.5*(SXX_avg^2+SYY_avg^2) +
                        0.5*((-SXX[i,j]-SYY[i,j])^2+(-SXX[i+1,j]-SYY[i+1,j])^2+
                             (-SXX[i,j+1]-SYY[i,j+1])^2+(-SXX[i+1,j+1]-SYY[i+1,j+1])^2))
                    ptB = wij*pt[i,j]+wi1j*pt[i+1,j]+wij1*pt[i,j+1]+wi1j1*pt[i+1,j+1]
                    pfB = wij*pf[i,j]+wi1j*pf[i+1,j]+wij1*pf[i,j+1]+wi1j1*pf[i+1,j+1]
                    prB = ptB - pfB

                    kfxy  = ETA[i,j]/(GGG[i,j]*dt+ETA[i,j])
                    siiel = SIIB[i,j]/kfxy
                    dyW_i = yp[i+1]-yp[i]

                    ETAVP = 0.0;  syield = 0.0;  V = 0.0;  EIISLIP = 0.0

                    if is_rsf
                        # ====================================================
                        #  Rate-and-state friction (Dieterich, 1979; Ruina, 1983)
                        #  Aging-law state evolution
                        # ====================================================
                        SIIB1 = SIIB[i,j]
                        V = 2*V0*sinh(max(SIIB1,0)/ARSF/prB)*exp(-(BRSF*OM[i,j]+FRIC[i,j])/ARSF)
                        OM5[i,j] = V*dt/LRSF > 1e-6 ?
                            log(V0/V+(exp(OM0[i,j])-V0/V)*exp(-V*dt/LRSF)) :
                            log(exp(OM0[i,j])*(1-V*dt/LRSF)+V0*dt/LRSF)
                        syield = max(syieldmin, prB*ARSF*asinh(V/2/V0*exp((BRSF*OM5[i,j]+FRIC[i,j])/ARSF)))
                        ETAVP  = ETA0[i,j]*syield/(ETA0[i,j]*(V/dyW_i)+syield)
                        SIIB2  = siiel*ETAVP/(GGG[i,j]*dt+ETAVP)
                        DSIIB1 = SIIB2-SIIB1

                        V = 2*V0*sinh(max(SIIB2,0)/ARSF/prB)*exp(-(BRSF*OM[i,j]+FRIC[i,j])/ARSF)
                        OM5[i,j] = V*dt/LRSF > 1e-6 ?
                            log(V0/V+(exp(OM0[i,j])-V0/V)*exp(-V*dt/LRSF)) :
                            log(exp(OM0[i,j])*(1-V*dt/LRSF)+V0*dt/LRSF)
                        syield = max(syieldmin, prB*ARSF*asinh(V/2/V0*exp((BRSF*OM5[i,j]+FRIC[i,j])/ARSF)))
                        ETAVP  = ETA0[i,j]*syield/(ETA0[i,j]*(V/dyW_i)+syield)
                        SIIB3  = siiel*ETAVP/(GGG[i,j]*dt+ETAVP)
                        DSIIB2 = SIIB3-SIIB2

                        # Bisection convergence loop (bounded to 10 iterations)
                        if (DSIIB1>=0 && DSIIB2<=0)||(DSIIB1<=0 && DSIIB2>=0)
                            DSIIB = 1e9;  ijk = 0
                            while abs(DSIIB)>1e-3 && ijk<10
                                SIIB4 = (SIIB1+SIIB2)/2
                                V = 2*V0*sinh(max(SIIB4,0)/ARSF/prB)*exp(-(BRSF*OM[i,j]+FRIC[i,j])/ARSF)
                                OM5[i,j] = V*dt/LRSF > 1e-6 ?
                                    log(V0/V+(exp(OM0[i,j])-V0/V)*exp(-V*dt/LRSF)) :
                                    log(exp(OM0[i,j])*(1-V*dt/LRSF)+V0*dt/LRSF)
                                syield = max(syieldmin, prB*ARSF*asinh(V/2/V0*exp((BRSF*OM5[i,j]+FRIC[i,j])/ARSF)))
                                ETAVP  = ETA0[i,j]*syield/(ETA0[i,j]*(V/dyW_i)+syield)
                                SIIB5  = siiel*ETAVP/(GGG[i,j]*dt+ETAVP)
                                DSIIB  = SIIB5-SIIB4
                                (DSIIB>=0 && DSIIB1>=0)||(DSIIB<=0 && DSIIB1<=0) ? (SIIB1=SIIB4) : (SIIB2=SIIB4)
                                ijk += 1
                            end
                        end

                        SigmaY[i,j] = syield
                        VSLIPB[i,j] = V
                        Vfault[i,j] = V
                        EII_p[i,j]  = V/(2*dyW_i)

                        # Lapusta-Liu (2009) quasi-static stiffness timestep estimate
                        Bmod   = 1/BETTASOLID
                        vi_nu  = (3*Bmod-2*GGG[i,j])/(6*Bmod+2*GGG[i,j])
                        kstiff = 2/pi*GGG[i,j]/(1-vi_nu)/dyW_i
                        xiarg  = 0.25*(kstiff*LRSF/ARSF/prB-(BRSF-ARSF)/ARSF)^2 - kstiff*LRSF/ARSF/prB
                        dTETAmax = xiarg < 0 ?
                            min(1-(BRSF-ARSF)*prB/(kstiff*LRSF), 0.2) :
                            min(ARSF*prB/(kstiff*LRSF-(BRSF-ARSF)*prB), 0.2)
                        if V > 0
                            dt_rsf = min(dt_rsf, abs(dTETAmax*LRSF/V))
                        end

                        AXY[i,j]  = syield/siiel
                        ETA[i,j]  = max(min(ETAVP, ETA0[i,j]), etamin)
                        YNY[i,j]  = 1
                        ynn = (YNY0[i,j] > 0) ? 1 : 0
                        if ynn == 0
                            DSY[i,j] = SIIB[i,j]-syield
                            ddd += DSY[i,j]^2
                            ynpl += 1
                        else
                            DSY[i,j] = SIIB[i,j]-syield
                            ddd += DSY[i,j]^2
                            ynpl += 1
                        end

                    else
                        # ====================================================
                        #  Rate-strengthening power-law plasticity (Yi et al., 2018)
                        # ====================================================
                        gammayi = AMURSF[i,j]
                        kfxy0   = ETA0[i,j]/(GGG[i,j]*dt+ETA0[i,j])
                        SIIB0   = siiel*kfxy0
                        syield0 = max(min(COHC[i,j]+FRIC[i,j]*prB, COHT[i,j]+FRIT[i,j]*prB), syieldmin)

                        SIIB1    = SIIB[i,j]
                        EIISLIP  = max(SIIB1/2/etamax, epsilonyi*max(0,SIIB1/syield0)^(1/gammayi))
                        ETAPL    = SIIB1/2/EIISLIP
                        ETAVP    = 1/(1/ETA0[i,j]+1/ETAPL)
                        SIIB2    = siiel*ETAVP/(GGG[i,j]*dt+ETAVP)
                        DSIIB1   = SIIB2-SIIB1

                        EIISLIP  = max(SIIB2/2/etamax, epsilonyi*max(0,SIIB2/syield0)^(1/gammayi))
                        ETAPL    = SIIB2/2/EIISLIP
                        ETAVP    = 1/(1/ETA0[i,j]+1/ETAPL)
                        SIIB3    = siiel*ETAVP/(GGG[i,j]*dt+ETAVP)
                        DSIIB2   = SIIB3-SIIB2

                        if (DSIIB1>=0 && DSIIB2<=0)||(DSIIB1<=0 && DSIIB2>=0)
                            DSIIB = 1e9;  ijk = 0
                            while abs(DSIIB)>1e-3
                                SIIB4   = (SIIB1+SIIB2)/2
                                EIISLIP = max(SIIB4/2/etamax, epsilonyi*max(0,SIIB4/syield0)^(1/gammayi))
                                ETAPL   = SIIB4/2/EIISLIP
                                ETAVP   = 1/(1/ETA0[i,j]+1/ETAPL)
                                SIIB5   = siiel*ETAVP/(GGG[i,j]*dt+ETAVP)
                                DSIIB   = SIIB5-SIIB4
                                (DSIIB>=0 && DSIIB1>=0)||(DSIIB<=0 && DSIIB1<=0) ? (SIIB1=SIIB4) : (SIIB2=SIIB4)
                                ijk += 1
                            end
                        end

                        VSLIPB[i,j] = EIISLIP*2*dyW_i
                        Vfault[i,j] = EIISLIP*2*b
                        EII_p[i,j]  = EIISLIP

                        syield      = syield0*(EIISLIP/epsilonyi)^gammayi
                        SigmaY[i,j] = syield
                        A           = syield/siiel
                        AXY[i,j]    = A

                        ynn = (YNY0[i,j] > 0) ? 1 : 0
                        if ynn == 1
                            DSY[i,j] = SIIB[i,j]-syield
                            ddd += DSY[i,j]^2
                            ynpl += 1
                        end

                        # FIX-BUG-3: ETA5_arr carries the per-node plastic
                        # viscosity (or ETA0 for non-yielding nodes, which was
                        # already set above by the reset `ETA5_arr .= ETA0`).
                        # This mirrors v1's ETA5 array exactly.
                        if A < 1
                            etapl = dt*GGG[i,j]*A/(1-A)
                            if etapl < ETA0[i,j]
                                ETA5_arr[i,j] = etapl^(1-etawt)*ETA[i,j]^etawt
                                YNY[i,j] = 1
                                if ynn == 0
                                    DSY[i,j] = SIIB[i,j]-syield
                                    ddd += DSY[i,j]^2
                                    ynpl += 1
                                end
                            else
                                ETA5_arr[i,j] = ETA0[i,j]   # not plastic: revert
                            end
                        else
                            ETA5_arr[i,j] = ETA0[i,j]       # not plastic: revert
                        end
                    end  # rheology branch
                end  # j
            end  # i
        end  # timestep > tyield

        # -----------------------------------------------------------------
        #  Convergence error
        # -----------------------------------------------------------------
        if ynpl > 0
            push!(DSYLSQ, sqrt(ddd/ynpl))
            c_err = DSYLSQ[iterstep]
            c_err_old = iterstep > 1 ? DSYLSQ[iterstep-1] : nothing
        else
            c_err = nothing;  c_err_old = nothing
        end

        # Reset ETA if no yielding
        if ynpl == 0
            @inbounds ETA .= ETA0
        end

        # -----------------------------------------------------------------
        #  Timestep adjustment
        # -----------------------------------------------------------------
        dtpl = dt
        if ynpl>0 && iterstep<niterglobal && ynlast>=dtstep &&
           (ynlast>ynlastmax || log10(c_err/c_err_old)>=0 ||
            log10(c_err/c_err_old)>log10(errmin/c_err)/(ynlastmax-ynlast))
            dtpl = dt/dtkoef;  yn = 1
        end

        if iterstep > 1;  maxvxy0 = maxvxy;  end
        maxvxy = sqrt((maximum(vxs)-minimum(vxs))^2 + (maximum(vys)-minimum(vys))^2)

        dtslip = 1e30
        for j in 1:Nx
            VSLIPB[Ny1,j] = 0.0
            for i in 1:Ny
                VSLIPB[Ny1,j] = max(VSLIPB[Ny1,j], VSLIPB[i,j]/(2*(yp[i+1]-yp[i])))
            end
            VSLIPB[Ny1,j] > 0 && (dtslip = min(dtslip, stpmax/VSLIPB[Ny1,j]))
        end

        if !is_rsf && ynpl>0 && dtslip<dt
            yn = 1;  dtslip = dtslip/dtkoefv
        end

        if yn>0 && dt>dtmin
            dtold = dt
            dt = is_rsf ? max(min(dtpl, 1.0*dt_rsf), dtmin) : max(min(dtpl, dtslip), dtmin)
            dt < dtold && (ynlast = 0)
        else
            yn = 0
        end

        # -----------------------------------------------------------------
        #  Check convergence and update ETA / ETA50
        # -----------------------------------------------------------------
        ynstop = 0
        if iterstep > 1
            vratio = log10(maxvxy/maxvxy0)
        end

        if yn==0 && (ynpl==0 || (c_err<errmin && iterstep>1 && abs(vratio)<vratiomax))
            ynstop = 1
        elseif c_err >= errmin
            if is_rs
                # Apply the plastic-viscosity correction (matches v1's ETA←ETA5 update)
                for i in 1:Ny, j in 1:Nx
                    ETA[i,j] = max(min(ETA5_arr[i,j], ETA0[i,j]), etamin)
                end
            end
            @inbounds ETA50 .= ETA   # save last pre-convergence plastic state
            YNY0 = copy(YNY)
            if is_rsf
                @inbounds OM .= OM5
            end
        end

        ynlast += 1
        last_iterstep = iterstep
        ynstop == 1 && break

    end  # inner iteration loop

    # =========================================================================
    #  POST-ITERATION UPDATES
    # =========================================================================

    dt00 > dt && (yndtdecrease = 1)

    # Adaptive dtmin
    if dt > dtminend && dtmin > dtminbeg
        dtmin = dtminbeg
    end
    if dt < dtminend && ynpl>0 && last_iterstep<dtminiter && dtslip>=dtslip00
        dtmin = min(dtmin*dtkoefup, dtminend)
    end
    if dt > dtminbeg && ynpl>0 && last_iterstep>dtminiter
        dtmin = max(dtmin/dtkoefup, dtminbeg)
    end

    # RS:  propagate the last pre-convergence plastic viscosity (ETA50) so
    #      the next timestep restarts from the correct bisection state
    #      (matches v1's `ETA00 = copy(ETA50)`).
    # RSF: propagate the FINAL CONVERGED viscosity (ETA) because ETA is
    #      updated unconditionally inside the plasticity loop; using ETA50
    #      would lag by one iteration and prevent localisation.
    if is_rsf
        @inbounds ETA00 .= ETA
    else
        @inbounds ETA00 .= ETA50
    end
    if is_rsf
        @inbounds OM0 .= OM
    end

    # Final stress and strain-rate computation (used as initial conditions
    # for the next timestep via SXX0/SYY0/SXY0).
    @inbounds for i in 1:Ny, j in 1:Nx
        if i < Ny && j < Nx
            ESP[i,j] = 0.5*((vys[i,j+1]-vys[i,j])/dxp[j]-(vxs[i+1,j]-vxs[i,j])/dyp[i])
            EXY[i,j] = 0.5*((vxs[i+1,j]-vxs[i,j])/dyp[i]+(vys[i,j+1]-vys[i,j])/dxp[j])
            KXY = dt*GGG[i,j]/(dt*GGG[i,j]+ETA[i,j])
            SXY[i,j] = 2*ETA[i,j]*EXY[i,j]*KXY + SXY0[i,j]*(1-KXY)
            DSXY[i,j] = SXY[i,j]-SXY0[i,j]
        end
        if i >= 2 && j >= 2
            EXX[i,j] = (2*(vxs[i,j]-vxs[i,j-1])/dxn[j-1]-(vys[i,j]-vys[i-1,j])/dyn[i-1])/3
            EYY[i,j] = (2*(vys[i,j]-vys[i-1,j])/dyn[i-1]-(vxs[i,j]-vxs[i,j-1])/dxn[j-1])/3
            KXX = dt*GGGP[i,j]/(dt*GGGP[i,j]+ETAP[i,j])
            SXX[i,j] = 2*ETAP[i,j]*EXX[i,j]*KXX + SXX0[i,j]*(1-KXX)
            SYY[i,j] = 2*ETAP[i,j]*EYY[i,j]*KXX + SYY0[i,j]*(1-KXX)
            DSXX[i,j] = SXX[i,j]-SXX0[i,j]
            DSYY[i,j] = SYY[i,j]-SYY0[i,j]
            EXY1 = (EXY[i,j]+EXY[i-1,j]+EXY[i,j-1]+EXY[i-1,j-1])*0.25
            EII[i,j]  = sqrt(0.5*(EXX[i,j]^2+EYY[i,j]^2)+EXY1^2)
            SXY1 = (SXY[i,j]+SXY[i-1,j]+SXY[i,j-1]+SXY[i-1,j-1])*0.25
            SII[i,j]  = sqrt(0.5*(SXX[i,j]^2+SYY[i,j]^2)+SXY1^2)
            disxy = (SXY[i,j]^2/2/ETA[i,j]+SXY[i-1,j]^2/2/ETA[i-1,j]+
                     SXY[i,j-1]^2/2/ETA[i,j-1]+SXY[i-1,j-1]^2/2/ETA[i-1,j-1])*0.25
            DIS[i,j] = SXX[i,j]^2/2/ETAP[i,j]+SYY[i,j]^2/2/ETAP[i,j]+2*disxy
            pt_ave = (i<Ny) ? (pt[i,j]+pt[i+1,j])/2 : pt[i,j]
            pf_ave = (i<Ny) ? (pf[i,j]+pf[i+1,j])/2 : pf[i,j]
            VIS_COMP[i,j] = (pt_ave-pf_ave)/(ETAB[i,j]*(1-POR[i,j]))
        end
    end

    timesum += dt

    # Update fluid velocities
    for i in 1:Ny1, j in 1:Nx1
        PORX[i,j] > 0 && (VXF0[i,j] = vxs[i,j]+vxD[i,j]/PORX[i,j])
        PORY[i,j] > 0 && (VYF0[i,j] = vys[i,j]+vyD[i,j]/PORY[i,j])
    end

    @inbounds VX0  .= vxs;   @inbounds VY0  .= vys
    @inbounds SXX0 .= SXX;   @inbounds SYY0 .= SYY;  @inbounds SXY0 .= SXY
    @inbounds @. PTF0 = pt-pf
    @inbounds PT0 .= pt;     @inbounds PF0  .= pf

    Vmax = maximum(VSLIPB[1:Ny,:])

    if Vmax > 5e-1
        println("Stopping: Vmax exceeded threshold (", Vmax, ")")
        break
    end

    # -----------------------------------------------------------------
    #  Console output
    # -----------------------------------------------------------------
    runtime = time()-loop_start_time
    @printf "#: %5d | dt: %.5E | time: %.7E | run-time: %.3f s | iter: %4d | Vmax: %.6E\n" timestep dt timesum runtime last_iterstep Vmax


    # =============================================================================
    #  FILE OUTPUT
    # =============================================================================
    if save_data
        if timesum == dt && timestep == start_step
            # Grid coordinates — written once at the start of each fresh/restarted run
            open("fault.txt", "a") do file
                for val in yp
                    write(file, @sprintf("%.20E\n", val))
                end
            end
        else
            # Slip rate profile (fault velocity at each y-node)
            open("EVO_Vslip.txt", "a") do file
                write(file, @sprintf("%.6E ", timesum))
                write(file, @sprintf("%.6E ", dt))
                for val in Vfault[1:end-1, 2];  write(file, @sprintf("%.6E ", val));  end
                write(file, "\n")
            end

            # Plastic strain-rate invariant
            open("EVO_EIIp.txt", "a") do file
                write(file, @sprintf("%.6E ", timesum))
                write(file, @sprintf("%.6E ", dt))
                for val in EII_p[1:end-1, 2];  write(file, @sprintf("%.6E ", val));  end
                write(file, "\n")
            end

            # Viscosity
            open("EVO_viscosity.txt", "a") do file
                write(file, @sprintf("%.6E ", timesum))
                write(file, @sprintf("%.6E ", dt))
                for val in ETA[1:end, 2];  write(file, @sprintf("%.6E ", val));  end
                write(file, "\n")
            end

            # Fluid pressure
            open("EVO_press_flu.txt", "a") do file
                write(file, @sprintf("%.6E ", timesum))
                write(file, @sprintf("%.6E ", dt))
                for val in pf[1:end, 2];  write(file, @sprintf("%.6E ", val));  end
                write(file, "\n")
            end

            # Effective pressure  (Pt − Pf)
            P_diff = pt .- pf
            open("EVO_press_eff.txt", "a") do file
                write(file, @sprintf("%.6E ", timesum))
                write(file, @sprintf("%.6E ", dt))
                for val in P_diff[1:end, 2];  write(file, @sprintf("%.6E ", val));  end
                write(file, "\n")
            end

            # Yield stress
            open("EVO_SigmaY.txt", "a") do file
                write(file, @sprintf("%.9E ", timesum))
                write(file, @sprintf("%.9E ", dt))
                for val in SigmaY[1:end, 2];  write(file, @sprintf("%.9E ", val));  end
                write(file, "\n")
            end

            # Dilation rate
            open("EVO_DILP.txt", "a") do file
                write(file, @sprintf("%.9E ", timesum))
                write(file, @sprintf("%.9E ", dt))
                for val in DILP[1:end, 2];  write(file, @sprintf("%.9E ", val));  end
                write(file, "\n")
            end

            # Volumetric compaction/dilation rate
            open("EVO_COMP.txt", "a") do file
                write(file, @sprintf("%.9E ", timesum))
                write(file, @sprintf("%.9E ", dt))
                for val in VIS_COMP[1:end, 2];  write(file, @sprintf("%.9E ", val));  end
                write(file, "\n")
            end

            # Summary row: time, dt, Vmax, iterations, plastic iterations
            Vmax_inner = maximum(VSLIPB[2:end-1,:])
            open("EVO_data.txt", "a") do file
                write(file, @sprintf("%.12E  %.12E  %.12E  %d   %d\n",
                    timesum, dt, Vmax_inner, ynlast, last_iterstep))
            end
        end
    end


    # =============================================================================
    #  CHECKPOINT  (every savematstep timesteps)
    # =============================================================================
    if timestep % savematstep == 0
        snapfile = @sprintf("%s_%07d.jld2", nname, timestep)
        jldsave(snapfile;
            timestep, timesum, dt,
            rheology = String(rheology),
            SXX, SYY, SXY, SXX0, SYY0, SXY0,
            pt, pf, PT0, PF0,
            ETA, ETA0, ETA00,
            VSLIPB, Vfault, EII_p, SIIB, SigmaY, YNY,
            OM, OM0, OM5, dt_rsf,
            x, y, xp, yp,
        )
        @printf "         | --> [checkpoint] step %d -> %s  (t = %.3e s)\n" timestep snapfile timesum
    end

end  # main time-stepping loop

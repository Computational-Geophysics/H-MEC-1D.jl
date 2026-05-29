# =============================================================================
#  plot_EIIp_pressure_snapshots.jl
#
#  Barras-style shifted profiles across the fault zone:
#
#    Panel (a): plastic strain-rate invariant EII_p across the gouge
#    Panel (b): fluid pressure pf across the gouge
#
#  Input files:
#    fault.txt
#    EVO_EIIp.txt
#    EVO_press_flu.txt
# =============================================================================

using Plots
using DelimitedFiles
using Printf
using Statistics
using LaTeXStrings

# Shared repository-root results directory (data input + figure output).
const RESULTS_DIR = get(ENV, "HMEC_RESULTS",
                        abspath(joinpath(@__DIR__, "..", "results")))
mkpath(RESULTS_DIR); cd(RESULTS_DIR)

# =============================================================================
# Configuration
# =============================================================================

const DATA_DIR   = RESULTS_DIR
const FAULT_FILE = joinpath(DATA_DIR, "fault.txt")
const EII_FILE   = joinpath(DATA_DIR, "EVO_EIIp.txt")
const PF_FILE    = joinpath(DATA_DIR, "EVO_press_flu.txt")

const OUT_PDF = "EIIp_pressure_profiles.pdf"
const OUT_PNG = "EIIp_pressure_profiles.png"

# Manual snapshots. Set to nothing for automatic selection.
manual_idx_local = (400, 650, 750, 950, 1580)
# manual_idx_local = nothing

# Number of automatic snapshots if manual_idx_local = nothing
const N_AUTO_SNAP = 10

# Floors / scaling
const EII_FLOOR = 1e-30
const PF_SCALE  = 1.0e6      # pressure plotted in MPa
const PROFILE_WIDTH = 0.65   # horizontal amplitude of shifted profiles

# =============================================================================
# Load data
# =============================================================================

println("Loading data ...")

yp = vec(readdlm(FAULT_FILE))

data_eii = readdlm(EII_FILE)
data_pf  = readdlm(PF_FILE)

times_eii = data_eii[:, 1]
dts_eii   = data_eii[:, 2]
EII       = data_eii[:, 3:end]

times_pf = data_pf[:, 1]
dts_pf   = data_pf[:, 2]
PF       = data_pf[:, 3:end]

n_iters_eii, neii = size(EII)
n_iters_pf,  npf  = size(PF)

@printf("  fault.txt          : %d y-values\n", length(yp))
@printf("  EVO_EIIp.txt       : %d saved iterations × %d columns\n", n_iters_eii, neii)
@printf("  EVO_press_flu.txt  : %d saved iterations × %d columns\n", n_iters_pf, npf)

if n_iters_eii != n_iters_pf
    @warn "EII and pressure files have different number of saved iterations."
end

n_iters = min(n_iters_eii, n_iters_pf)

# =============================================================================
# Build y coordinates
# =============================================================================

function build_y_profile(yp, nprofile; quantity_name="profile")
    if nprofile == length(yp)
        # nodal quantity, e.g. pf[1:end, 2]
        y = copy(yp)

    elseif nprofile == length(yp) - 1
        # interval-centred quantity
        y = 0.5 .* (yp[1:end-1] .+ yp[2:end])

    elseif nprofile == length(yp) - 2
        # interior nodal/cell quantity with both boundaries removed,
        # e.g. EII_p[1:end-1, 2] when fault.txt contains two extra boundary points
        y = copy(yp[2:end-1])

    else
        error("Cannot match $quantity_name columns ($nprofile) with fault.txt length ($(length(yp))).")
    end

    # Centre around the geometric middle of the fault-zone coordinate.
    # Do not use argmin(abs(y)) here because fault.txt may not be centred on zero.
    ycenter = 0.5 * (minimum(y) + maximum(y))
    return y .- ycenter
end

y_eii = build_y_profile(yp, neii; quantity_name="EII_p")
y_pf  = build_y_profile(yp, npf;  quantity_name="fluid pressure")

# =============================================================================
# Robust central window around the shear zone
# =============================================================================

const N_HALF_CELLS = 80

function central_mask(y, nhalf)
    ic = argmin(abs.(y))
    i1 = max(1, ic - nhalf)
    i2 = min(length(y), ic + nhalf)

    mask = falses(length(y))
    mask[i1:i2] .= true

    h = maximum(abs.(y[mask]))
    yn = y ./ h

    return mask, yn, h
end

zoom_eii, yn_eii, h_eii = central_mask(y_eii, N_HALF_CELLS)
zoom_pf,  yn_pf,  h_pf  = central_mask(y_pf,  N_HALF_CELLS)

@printf("  EII central window: %d points, h = %.6e m\n", count(zoom_eii), h_eii)
@printf("  PF  central window: %d points, h = %.6e m\n", count(zoom_pf),  h_pf)

@printf("  EII y/h range      : %.3f to %.3f\n", minimum(yn_eii[zoom_eii]), maximum(yn_eii[zoom_eii]))
@printf("  PF  y/h range      : %.3f to %.3f\n", minimum(yn_pf[zoom_pf]),  maximum(yn_pf[zoom_pf]))

# =============================================================================
# Snapshot selection
# =============================================================================

if manual_idx_local !== nothing
    snap_idx = collect(manual_idx_local)
else
    EIImax_t = vec(maximum(EII, dims=2))
    idx_peak = argmax(EIImax_t)

    idx1 = max(1, idx_peak - 400)
    idx2 = idx_peak

    snap_idx = unique(round.(Int, range(idx1, idx2, length=N_AUTO_SNAP)))
end

for idx in snap_idx
    if idx < 1 || idx > n_iters
        error("Invalid snapshot index $idx. Valid range is 1:$n_iters")
    end
end

nsnap = length(snap_idx)
xshift = collect(0:nsnap-1)

println("Selected snapshots:")
for (j, idx) in enumerate(snap_idx)
    @printf("  %2d -> row %5d   time = %.6e s\n", j, idx, times_eii[idx])
end

# =============================================================================
# Normalize profiles for shifted plotting
# =============================================================================

function normalize_profile(v, vmin, vmax)
    den = vmax - vmin

    if abs(den) < 1.0e-99
        return fill(0.0, length(v))
    end

    return (v .- vmin) ./ den
end

# =============================================================================
# Plot style
# =============================================================================

default(
    fontfamily        = "Computer Modern",
    framestyle        = :box,
    grid              = false,
    legendfontsize    = 9,
    guidefontsize     = 11,
    tickfontsize      = 10,
    titlefontsize     = 11,
    linewidth         = 1.3,
    dpi               = 300,
)

cols_eii = range(
    RGB(1.00, 0.85, 0.65),   # pale orange
    RGB(0.90, 0.35, 0.00),   # strong orange
    length=nsnap
)

cols_pf = range(
    RGB(0.70, 0.85, 1.00),   # pale blue
    RGB(0.00, 0.35, 0.85),   # strong blue
    length=nsnap
)

# =============================================================================
# Panel (a): EII_p
# =============================================================================

p1 = plot(
    xlabel = "Relative snapshot position",
    ylabel = L"y/h",
    title  = L"\mathrm{strain\ rate\ across\ the\ shear\ zone}",
    xlims  = (-0.5, nsnap + 1),
    ylims  = (-1.0, 1.0),
    legend = false,
)

for (j, idx) in enumerate(snap_idx)
    prof = log10.(max.(EII[idx, zoom_eii], EII_FLOOR))
    prof = prof .- minimum(prof)

    pmax = maximum(prof)
    if pmax > 0.0
        profn = prof ./ pmax*1.2
    else
        profn = fill(0.0, length(prof))
    end

    xprof = xshift[j] .+ PROFILE_WIDTH .* profn
    yprof = yn_eii[zoom_eii]

    plot!(p1, xprof, yprof;
        color = cols_eii[j],
        lw = 1.4)
end

# =============================================================================
# Panel (b): fluid pressure
# =============================================================================

p2 = plot(
    xlabel = "Relative snapshot position",
    ylabel = L"y/h",
    title  = L"\mathrm{fluid\ pressure\ across\ the\ shear\ zone}",
    xlims  = (-0.5, nsnap + 1),
    ylims  = (-1.0, 1.0),
    legend = false,
)

for (j, idx) in enumerate(snap_idx)
    dpf = PF[idx, zoom_pf]
    dpf = dpf .- minimum(dpf)

    pmax = maximum(dpf)
    if pmax > 0.0
        dpfn = dpf ./ pmax.*1.2
    else
        dpfn = fill(0.0, length(dpf))
    end

    xprof = xshift[j] .+ PROFILE_WIDTH .* dpfn
    yprof = yn_pf[zoom_pf]

    plot!(p2, xprof, yprof;
        color = cols_pf[j],
        lw = 1.4)
end

# =============================================================================
# Add simple scale bars
# =============================================================================

annotate!(p1, nsnap - 0.2, -0.85, text(L"\dot{\gamma}/\dot{\gamma}_{max}", 10, :black))
annotate!(p2, nsnap - 0.2, -0.85, text(L"\Delta p_f", 10, :black))

# =============================================================================
# Combine and save
# =============================================================================

plt = plot(
    p1, p2;
    layout = (1,2),
    size = (800, 320),
    left_margin   = 6Plots.mm,
    right_margin  = 5Plots.mm,
    bottom_margin = 6Plots.mm,
    top_margin    = 4Plots.mm,
)

savefig(plt, OUT_PDF)
savefig(plt, OUT_PNG)

display(plt)

println("\nSaved figures:")
println("  ", OUT_PDF)
println("  ", OUT_PNG)
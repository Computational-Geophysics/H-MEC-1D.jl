# =============================================================================
#  plot_vslip_snapshots.jl
#
#  Two-panel figure of the slip-rate profile across the fault zone.
#
#    Left panel  : full domain (the entire simulated y-extent)
#    Right panel : zoomed view (default ±10 mm around the fault centre)
#
#  Three time snapshots are overlaid in each panel:
#    t1 -- loading       : early, before localization
#    t2 -- localization  : intermediate, slip rate focusing toward the fault
#    t3 -- diffusion     : peak coseismic slip, diffusion-controlled bandwidth
#
#  Dal Zilio & Gerya (2026)
# =============================================================================

using Plots
using DelimitedFiles
using Printf
using Statistics
using Dates

# Shared repository-root results directory (data input + figure output).
const RESULTS_DIR = get(ENV, "HMEC_RESULTS",
                        abspath(joinpath(@__DIR__, "..", "results")))
mkpath(RESULTS_DIR); cd(RESULTS_DIR)

#manual_idx_local = nothing
manual_idx_local = (500,820,1580) # <-- edit here directly
println("Manual indices selected: ", manual_idx_local)

# =============================================================================
#  Path / input-file diagnostics
# =============================================================================
const DATA_DIR = RESULTS_DIR
const FAULT_FILE = joinpath(DATA_DIR, "fault.txt")
const VSLIP_FILE = joinpath(DATA_DIR, "EVO_Vslip.txt")

println("=========================================================")
println("  fault.txt path        : ", abspath(FAULT_FILE))
println("  EVO_Vslip.txt path    : ", abspath(VSLIP_FILE))
println("=========================================================")

# =============================================================================
#  Configuration
# =============================================================================

# Half-width of the zoom panel (m). The screenshot uses 0.01 (i.e. ±10 mm).
const ZOOM_HALFWIDTH = 0.01 #0.002

# Floor used when taking log10 of slip rate, so empty/zero entries don't blow up.
const VMIN = 1e-30

# Output filenames
# const OUT_PNG = "vslip_snapshots.png"
const OUT_PDF = "vslip_snapshots.pdf"

# =============================================================================
#  Load data
# =============================================================================

println("Loading data ...")
yp = vec(readdlm(FAULT_FILE))                                           # length Ny1
data = readdlm(VSLIP_FILE)                                              # rows x cols
times = data[:, 1]                                                      # cumulative time (s)
dts   = data[:, 2]                                                      # dt at each saved iter (s)
V     = data[:, 3:end-1]                                                # rows x nslip slip rates (m/s)

n_iters, nslip = size(V)
@printf("  fault.txt        : %d y-values\n", length(yp))
@printf("  EVO_Vslip.txt    : %d saved iterations × %d slip-rate columns\n", n_iters, nslip)

@printf("  Last saved time  : %.6e s\n", times[end])
@printf("  Last dt          : %.6e s\n", dts[end])
@printf("  V min / max      : %.6e / %.6e m/s\n", minimum(V), maximum(V))
println()

# =============================================================================
# Vmax(t): peak slip rate at each saved iteration
Vmax_t = vec(maximum(V, dims = 2))

# Global peak slip-rate timestep
idx_peak = argmax(Vmax_t)

@printf("  Row index        : %d\n", idx_peak)
@printf("  Time             : %.15e s\n", times[idx_peak])
@printf("  Maximum V        : %.12e m/s\n", Vmax_t[idx_peak])
println()
# =============================================================================

#
# Match each slip-rate column to the physical fault-cell centres.
# After removing the final stale/extra column from EVO_Vslip.txt
# we use the first nslip intervals only.
#
y_plot = 0.5 .* (yp[1:nslip] .+ yp[2:nslip+1])


# =============================================================================
#  Snapshot selection
# =============================================================================

# Vmax(t): peak slip rate at each saved iteration
Vmax_t = vec(maximum(V, dims = 2))
log10_Vmax = log10.(max.(Vmax_t, VMIN))

if manual_idx_local !== nothing
    # User-defined indices take full priority
    idx_loading, idx_localization, idx_diffusion = manual_idx_local
else
    # Auto-pick: t1 = first iter, t3 = global Vmax peak, t2 = halfway in log-Vmax
    idx_loading      = 1
    idx_diffusion    = argmax(Vmax_t)
    if idx_diffusion <= idx_loading + 1
        # No clear peak yet — fall back to evenly-spaced
        idx_loading      = 1
        idx_localization = max(1, n_iters ÷ 2)
        idx_diffusion    = n_iters
        @warn "No clear Vmax peak detected; using evenly-spaced snapshots."
    else
        target = (log10_Vmax[idx_loading] + log10_Vmax[idx_diffusion]) / 2
        idx_localization = argmin(abs.(log10_Vmax[idx_loading:idx_diffusion] .- target)) + idx_loading - 1
    end
end

# Validate indices
for idx in (idx_loading, idx_localization, idx_diffusion)
    if idx < 1 || idx > n_iters
        error("manual_idx contains invalid index: $idx (valid range: 1:$n_iters)")
    end
end

snap_idx = (idx_loading, idx_localization, idx_diffusion)
#snap_idx = idx_localization
snap_lbl = ("\$t_1\$: loading", "\$t_2\$: localization", "\$t_3\$: dynamic")
snap_col = (RGB(0.78, 0.78, 0.78),  # light grey
            RGB(0.45, 0.45, 0.45),  # mid grey
            RGB(0.05, 0.05, 0.05))  # near-black

#
# -----------------------------------------------------------------------------
# Centre profiles around the geometric centre of the fault domain.
#
# The slip-rate maximum migrates during rupture propagation and therefore must
# not be used to define the geometric centre for plotting.
#
# The fault centre is fixed and corresponds to y = 0 in the original domain.
# We therefore centre using the grid point closest to y = 0.
# -----------------------------------------------------------------------------

icenter = length(yp) ÷ 2            # Find the index of the grid point closest to y = 0

y_center_value = y_plot[icenter]
y_centred = y_plot .- y_center_value

ysize = maximum(y_plot) - minimum(y_plot)

for (k, idx) in enumerate(snap_idx)
    @printf("  %s  -> row %4d   t = %10.3e s   Vmax = %10.3e m/s\n",
            snap_lbl[k], idx, times[idx], Vmax_t[idx])
end

# =============================================================================
#  Build the plot
# =============================================================================

# Take log10 of the slip-rate profiles for the chosen rows
log10V_snap = [log10.(max.(V[idx, :], VMIN)) for idx in snap_idx]

# Common plot styling

# Common plot styling
default(
    fontfamily        = "Computer Modern",
    framestyle        = :box,
    grid              = false,
    legendfontsize    = 10,
    guidefontsize     = 12,
    tickfontsize      = 10,
    titlefontsize     = 10,
    linewidth         = 1.0,
    markershape       = :circle,
    markersize        = 2.4,
    markerstrokewidth = 0.3,
    dpi               = 300,
)

# Common axis ranges
# Use the same logarithmic slip-rate range in both panels. With the corrected
# V = data[:, 3:end-1], early snapshots can fall below 1e-10 m/s; using
# (-10, 0) in the zoom panel can therefore make the curves disappear.
xlims_common_a = (-20, 0)
xlims_common_b = (-10, 0)

# --- Panel 1: full domain ----------------------------------------------------
p1 = plot(
    xlabel  = "Log slip rate (m/s)",
    ylabel  = "Distance from the shear zone (km)",
    #title   = "(a) Full domain",
    xlims   = xlims_common_a,
    ylims   = (-15,+15), #(minimum(y_centred./1e3), maximum(y_centred./1e3)),
    legend  = :topright,
)

# Remove the single lowest point (y = minimum)
y_km = y_centred ./ 1e3
ymin = minimum(y_km)
mask = y_km .> ymin

for (k, idx) in enumerate(snap_idx)
    plot!(p1, log10V_snap[k][mask], y_km[mask],
          color             = snap_col[k],
          markercolor       = snap_col[k],
          markerstrokecolor = snap_col[k],
          label             = snap_lbl[k])
end

# --- Panel 2: zoomed view ----------------------------------------------------
p2 = plot(
    xlabel  = "Log slip rate (m/s)",
    ylabel  = "Fault zone thickness (m)",
    #title   = "(b) Zoom: ±$(round(ZOOM_HALFWIDTH*1000, digits=1)) mm",
    xlims   = xlims_common_b,
    ylims   = (-ZOOM_HALFWIDTH, ZOOM_HALFWIDTH),
    legend  = false,
)
zoom_mask = abs.(y_centred) .<= ZOOM_HALFWIDTH

for (k, idx) in enumerate(snap_idx)
    plot!(p2, log10V_snap[k][zoom_mask], y_centred[zoom_mask],
          color             = snap_col[k],
          markercolor       = snap_col[k],
          markerstrokecolor = snap_col[k])
end

# --- Combine -----------------------------------------------------------------
plt = plot(p1, p2,
           layout = grid(1, 2, widths = [0.60, 0.40]),
           size   = (900, 500),
           dpi    = 200,
           left_margin   = 5Plots.mm,
           right_margin  = 5Plots.mm,
           bottom_margin = 5Plots.mm,
           top_margin    = 3Plots.mm)

#savefig(plt, OUT_PNG)
savefig(plt, OUT_PDF)

println("\nSaved figures:")

display(plt)

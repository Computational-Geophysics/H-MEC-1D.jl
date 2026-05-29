using DelimitedFiles
using Printf
using Plots
using LaTeXStrings
using Measures

gr()
# Read fault.txt and write figures in the shared results directory.
const RESULTS_DIR = get(ENV, "HMEC_RESULTS",
                        abspath(joinpath(@__DIR__, "..", "results")))
mkpath(RESULTS_DIR); cd(RESULTS_DIR)

# -----------------------------
# Input / output
# -----------------------------
const FAULT_FILE = "fault.txt"
const OUT_PDF = "fault_resolution.pdf"
const OUT_PNG = "fault_resolution.png"

# -----------------------------
# Load fault grid
# -----------------------------
yp = vec(readdlm(FAULT_FILE))

# Cell sizes Δy and cell-center coordinates
dy = abs.(diff(yp))
yc = 0.5 .* (yp[1:end-1] .+ yp[2:end])

# Center domain at geometric midpoint
y0 = 0.5 * (minimum(yp) + maximum(yp))
yc_km = (yc .- y0) ./ 1e3

dy_min = minimum(dy)
dy_max = maximum(dy)

# -----------------------------
# Style
# -----------------------------
default(
    fontfamily        = "Computer Modern",
    framestyle        = :box,
    grid              = false,
    legendfontsize    = 9,
    guidefontsize     = 12,
    tickfontsize      = 12,
    titlefontsize     = 12,
    linewidth         = 1.2,
    dpi               = 300,
)

# -----------------------------
# Plot
# -----------------------------
p = plot(
    dy, yc_km;
    xscale = :log10,
    color  = :black,
    lw     = 1.2,
    label  = "",
    xlabel = L"\mathrm{Resolution},\ \Delta y~[\mathrm{m}]",
    ylabel = L"\mathrm{Distance\ from\ the\ fault}~[\mathrm{km}]",
    xlims  = (1e-5, 1e5),
    ylims  = (-20, 20),
    size   = (420, 600),
    left_margin   = 8mm,
    right_margin  = 4mm,
    bottom_margin = 7mm,
    top_margin    = 4mm,
)

annotate!(p, 3e-5, 17.0,
    text(L"\mathrm{max}\;(\Delta y) = %$(round(dy_max/1e3, digits=1))~\mathrm{km}", 14, :left))

annotate!(p, 3e-5, 13.5,
    text(L"\mathrm{min}\;(\Delta y) = %$(round(dy_min*1e6, digits=0))~\mu\mathrm{m}", 14, :left))

savefig(p, OUT_PDF)
savefig(p, OUT_PNG)

display(p)

println("Wrote ", OUT_PDF, " and ", OUT_PNG)
@printf("Min Δy = %.6e m\n", dy_min)
@printf("Max Δy = %.6e m\n", dy_max)
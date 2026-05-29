# =============================================================================
#  LSA_dispersion_analysis.jl   (paper-production version)
#  Self-contained test of the LSA derived in LSA_derivation_manuscript.tex
#  (Barras & Brantut 2025 framework, H-MEC).
#
#  Dispersion:  sigma(k) = -D k^2 + W
#       D = kp/(etaf*S),   W = Lw/(S*etaphi*(1-phi))
#       Lw = f*p_e/(n*(C+f*p_e)) - 1
#       k_c = sqrt(W/D),   ell_c = 1/k_c       (S cancels in ell_c)
#
#  Panels: (a) sigma(k);  (b) storage S cancels in ell_c;
#          (c) ell_c ~ sqrt(eta_phi);  (d) instability boundary Lw>0.
# =============================================================================

using Plots, Printf, LaTeXStrings, Measures
# Write output figures to the shared repository-root results directory.
const RESULTS_DIR = get(ENV, "HMEC_RESULTS",
                        abspath(joinpath(@__DIR__, "..", "results")))
mkpath(RESULTS_DIR); cd(RESULTS_DIR)
gr(); print("\033c")

# ---- code parameters (h_mec_1D.jl) ----
const phi  = 0.01
const C    = 5.0e6
const f    = 0.3
const n    = 0.03
const kp   = 1e-18
const etaf = 1e-3
const G    = 30e9
const bS   = 2.5e-11
const bF   = 4e-10

storage(phi,bF,bS) = begin
    GB=G/phi; bdr=(1/GB+bS)/(1-phi); KBW=1-bS/bdr
    KSK=(bdr-bS)/(bdr-bS+phi*(bF-bS)); bdr*KBW/KSK
end
Lw_of(pe) = f*pe/(n*(C+f*pe)) - 1

# representative localised state
S0      = storage(phi,bF,bS)
etaphi0 = 1e17
pe0     = 1.5e7
Lw0     = Lw_of(pe0)
D0      = kp/(etaf*S0)
W0      = Lw0/(S0*etaphi0*(1-phi))
kc0     = sqrt(W0/D0)
ellc0   = 1/kc0

println("="^70)
println("  LSA dispersion analysis  (H-MEC)")
println("="^70)
@printf "  S=%.3e 1/Pa   D=%.3e m^2/s   Lw=%.2f\n" S0 D0 Lw0
@printf "  W=%.3e 1/s    k_c=%.3e 1/m   ell_c=%.3e m\n" W0 kc0 ellc0
pe_crit = n*C/(f*(1-n))
@printf "  instability onset: p_e > n*C/(f(1-n)) = %.3e Pa (%.3f MPa)\n" pe_crit pe_crit/1e6
println("="^70)

# =============================================================================
#  PLOTTING  (paper production)
# =============================================================================
const C_LINE  = RGB(0.16, 0.30, 0.66)   # primary curve (blue)
const C_PRED = RGB(0.80, 0.10, 0.15)    # secondary / marker line (red)
const C_REF  = RGB(0.50, 0.50, 0.50)    # reference lines (gray)

# Common plot styling
default(
    fontfamily        = "Computer Modern",
    framestyle        = :box,
    grid              = false,
    legendfontsize    = 9,
    guidefontsize     = 10,
    tickfontsize      = 10,
    titlefontsize     = 10,
    linewidth         = 0.8,
    markershape       = :circle,
    markersize        = 4.5,
    markerstrokewidth = 1.1,
    dpi               = 300,
)
ln = (markershape=:none,)   # pure-line series (suppress global circle default)

# Consistent reference-line style, matching panel (d)
refdash = (; ls=:dash, lw=1)

# (a) dispersion relation sigma(k)
k = 10 .^ range(log10(kc0/30), log10(kc0*30), length=400)
sigma = -D0 .* k.^2 .+ W0

pa = plot(k, sigma;
    lw=2,
    color=C_LINE,
    xscale=:log10,
    xlabel=L"k~[\mathrm{1/m}]",
    ylabel=L"\sigma(k)~[\mathrm{1/s}]",
    title=L"(a) \mathrm{dispersion}",
    legend=false,
    ln...)

hline!(pa, [0.0];
    color=C_REF,
    refdash...,
    ln...)

vline!(pa, [kc0];
    color=C_PRED,
    refdash...,
    ln...)

annotate!(pa, kc0+0.1, -0.005,
    text(L"k_c", 12, C_PRED, :left))

# (b) storage S cancels in ell_c
Svals = 10 .^ range(log10(S0/30), log10(S0*30), length=40)

ellc_vs_S = [
    1 / sqrt((Lw0 / (S * etaphi0 * (1 - phi))) / (kp / (etaf * S)))
    for S in Svals
]

Wvs = [
    Lw0 / (S * etaphi0 * (1 - phi))
    for S in Svals
]

pb = plot(Svals, ellc_vs_S ./ ellc0;
    lw=2,
    color=C_LINE,
    xscale=:log10,
    xlabel=L"S~[\mathrm{1/Pa}]",
    ylabel="normalised",
    title=L"(b)\ S\ \mathrm{cancels\ in}\ \ell_c",
    label=L"\ell_c/\ell_c^0",
    legend=:topright,
    ylims=(0,2),
    ln...)

plot!(pb, Svals, Wvs ./ W0;
    lw=2,
    color=C_PRED,
    refdash...,
    label=L"W/W^0\ (\propto 1/S)",
    ln...)

hline!(pb, [1.0];
    color=C_REF,
    refdash...,
    label="",
    ln...)

# (c) ell_c vs compaction viscosity
eps = 10 .^ range(13, 21, length=200)

ellc_vs = [
    1 / sqrt((Lw0 / (S0 * e * (1 - phi))) / (kp / (etaf * S0)))
    for e in eps
]

pc = plot(eps, ellc_vs;
    lw=2,
    color=C_LINE,
    xscale=:log10,
    xlabel=L"\eta_\phi~[\mathrm{Pa\,s}]",
    ylabel=L"\ell_c~[\mathrm{m}]",
    xticks=([1e14, 1e16, 1e18, 1e20],[L"10^{14}", L"10^{16}", L"10^{18}", L"10^{20}"]),
    title=L"(c)\ \ell_c\propto\sqrt{\eta_\phi}",
    legend=false,
    ln...)

# (d) instability boundary Lw(p_e)
pevals = 10 .^ range(5, 8, length=200)
Lwvals = Lw_of.(pevals)

pd = plot(pevals, Lwvals;
    lw=2,
    color=C_LINE,
    xscale=:log10,
    xlabel=L"p_{\mathrm{eff}}~[\mathrm{Pa}]",
    ylabel=L"\Lambda_w",
    xticks=([1e5, 1e6, 1e7, 1e8],[L"10^{5}", L"10^{6}", L"10^{7}", L"10^{8}"]),
    title=L"(d)\ \mathrm{instability:}\ \Lambda_w>0",
    legend=false,
    ln...)

hline!(pd, [0.0];
    color=C_PRED,
    refdash...,
    ln...)

vline!(pd, [pe_crit];
    color=C_REF,
    refdash...,
    ln...)

annotate!(pa, 1e-2, 0.003, text(L"\mathrm{a}", 15))
annotate!(pb, 3.0e-13, 2.1, text(L"\mathrm{b}", 15))
annotate!(pc, 1.0e12, 280, text(L"\mathrm{c}", 15))
annotate!(pd, 5.0e4, 30, text(L"\mathrm{d}", 15))

fig = plot(pa, pb, pc, pd;
    layout=(2,2),
    size=(1000,760),
    left_margin=8mm,
    right_margin=4mm,
    bottom_margin=7mm,
    top_margin=4mm)

savefig(fig, "LSA_dispersion_analysis.pdf")
savefig(fig, "LSA_dispersion_analysis.svg")

display(fig)

println("\nWrote LSA_dispersion_analysis.png/.pdf")
println("Panel (b): ell_c flat vs S (storage cancels); W ~ 1/S (rate depends on S).")

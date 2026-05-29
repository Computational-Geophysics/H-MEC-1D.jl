#= ===========================================================================
 Plotting for the 1D steady-state Couette solver (paper-production style).
 Reads couette_summary.txt and couette_profiles.txt written by 1D_steady_state.jl.

 Fig A  couette_tau_vs_V.pdf          : shear stress SXY vs imposed V
 Fig B  couette_width_vs_V.pdf        : MEASURED shear-zone width vs V, with the
                                        imposed ysize overlaid for comparison
 Fig C1 couette_profiles_normalized.pdf : profiles vs y/ysize   (collapse view)
 Fig C2 couette_profiles_physical.pdf   : profiles vs physical y [m] (absolute)
=========================================================================== =#

using DelimitedFiles, Printf, Plots, LaTeXStrings, Measures
# Read the Couette output and write figures in the shared results directory.
const RESULTS_DIR = get(ENV, "HMEC_RESULTS",
                        abspath(joinpath(@__DIR__, "..", "results")))
mkpath(RESULTS_DIR); gr(); cd(RESULTS_DIR)

const C_NUM = RGB(0.16,0.30,0.66)   # measured / primary (blue)
const C_EDGE= RGB(0.95,0.55,0.10)   # marker edge (orange)
const C_REF = RGB(0.50,0.50,0.50)   # imposed / reference (gray)
default(fontfamily="Arial", framestyle=:box, grid=false,
        legendfontsize=11, guidefontsize=12, tickfontsize=11, titlefontsize=12,
        linewidth=1.2, markershape=:circle, markersize=5, markerstrokewidth=0.6, dpi=300)
mknum = (seriestype=:scatter, markercolor=C_NUM, markerstrokecolor=C_EDGE)
ln    = (markershape=:none,)
safelog10(x) = x > 0 ? log10(x) : NaN     # mask non-positive (undefined) entries instead of faking -300

# ---- summary -------------------------------------------------------------
S = readdlm("couette_summary.txt", comments=true, comment_char='#')
vxpow=S[:,1]; V=S[:,2]; ysize=S[:,3]; SXY=S[:,4]; width=S[:,5]

# Common plot styling
default(
    fontfamily        = "Computer Modern",
    framestyle        = :box,
    grid              = false,
    legendfontsize    = 9,
    guidefontsize     = 12,
    tickfontsize      = 12,
    titlefontsize     = 10,
    linewidth         = 0.3,
    markershape       = :circle,
    markersize        = 4.0,
    markerstrokewidth = 0.5,
    dpi               = 300,
)

# Fig A : tau vs V
pA = plot(
    log10.(V), SXY ./ 1e6;
    xlabel = L"\log_{10}(V)~[\mathrm{m/s}]",
    ylabel = L"\tau~[\mathrm{MPa}]",
    label  = "",
    mknum...,
)
plot!(pA, log10.(V), SXY ./ 1e6;
    label = "",
    color = C_NUM,
    ln...,
)

# Fig B : measured width vs imposed ysize
pB = plot(;
    xlabel = L"\log_{10}(V)~[\mathrm{m/s}]",
    ylabel = L"\mathrm{thickness}~[\mathrm{m}]",
    yscale = :log10,
    ylims  = (1e-3, 1e3),
    yticks = ([1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3],
              [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}", L"10^{1}", L"10^{2}", L"10^{3}"]),
    legend = :topright,
)
plot!(pB, log10.(V), ysize;
    color = C_REF,
    ls    = :dash,
    lw    = 1.0,
    label = L"\mathrm{imposed}\ y_{\mathrm{size}}",
    ln...,
)
plot!(pB, log10.(V), width;
    color = C_NUM,
    lw    = 0.8,
    label = "",
    ln...,
)
plot!(pB, log10.(V), width;
    label = L"\mathrm{measured\ shear\ zone\ width}",
    mknum...,
)

# Combined two-panel summary figure
figAB = plot(
    pA, pB;
    layout        = grid(1, 2, widths = [0.50, 0.50]),
    size          = (1000, 390),
    left_margin   = 6mm,
    right_margin  = 4mm,
    bottom_margin = 7mm,
    top_margin    = 4mm,
)

savefig(figAB, "couette_tau_width_vs_V.pdf")
savefig(figAB, "couette_tau_width_vs_V.png")

# Also keep individual outputs
savefig(pA, "couette_tau_vs_V.pdf")
savefig(pA, "couette_tau_vs_V.png")
savefig(pB, "couette_width_vs_V.pdf")
savefig(pB, "couette_width_vs_V.png")

# ---- profile blocks ------------------------------------------------------
struct Blk; vxpow::Float64; ysize::Float64; width::Float64; yc::Float64
    yphys::Vector{Float64}; yorel::Vector{Float64}; vxs::Vector{Float64}
    peff::Vector{Float64}; sigy::Vector{Float64}; exy::Vector{Float64}
    etaphi::Vector{Float64}; Vcell::Vector{Float64}; end

function read_blocks(fname)
    B=Blk[]; vp=0.0; ys=1.0; wd=0.0; yc=0.0; c=[Float64[] for _ in 1:8]; have=false
    push_blk() = have && push!(B, Blk(vp,ys,wd,yc,c[1],c[2],c[3],c[4],c[5],c[6],c[7],c[8]))
    for line in eachline(fname)
        if startswith(line,"# vxpow=")
            push_blk(); c=[Float64[] for _ in 1:8]; have=true
            for tok in split(line)
                occursin("vxpow=",tok) && (vp=parse(Float64,split(tok,"=")[2]))
                occursin("ysize=",tok) && (ys=parse(Float64,split(tok,"=")[2]))
                occursin("width=",tok) && (wd=parse(Float64,split(tok,"=")[2]))
                occursin("yc=",tok)    && (yc=parse(Float64,split(tok,"=")[2]))
            end
        elseif startswith(line,"#"); continue
        else
            v=parse.(Float64,split(strip(line))); length(v)==8 || continue
            for j in 1:8; push!(c[j],v[j]); end
        end
    end
    push_blk(); return B
end
B = read_blocks("couette_profiles.txt")

# ---- profile figure (xkey :norm -> y/ysize ; :phys -> physical y) --------
# ---- profile figure with FULL manual control of panel position + size ----
# Every panel is placed by an absolute bounding box  bbox(x, y, w, h)  whose
# coordinates are fractions of the whole figure, measured from the TOP-LEFT
# corner (x grows rightward, y grows DOWNWARD). Edit the five boxes below to
# move/resize any panel independently, including the colorbar.

# Common plot styling
default(
    fontfamily        = "Computer Modern",
    framestyle        = :box,
    grid              = false,
    legendfontsize    = 9,
    guidefontsize     = 10,
    tickfontsize      = 10,
    titlefontsize     = 10,
    linewidth         = 0.3,
    markershape       = :circle,
    markersize        = 3.5,
    markerstrokewidth = 1.1,
    dpi               = 300,
)

function profiles(xkey, ylabel, ylims, fname;
        figsize = (1000, 500),
        box1 = bbox(0.040, 0.10, 0.200, 0.82),   # p_eff panel
        box2 = bbox(0.270, 0.10, 0.200, 0.82),   # gamma-dot panel
        box3 = bbox(0.500, 0.10, 0.200, 0.82),   # sigma_Y panel
        box4 = bbox(0.730, 0.10, 0.200, 0.82),   # eta_phi panel
        boxcb= bbox(0.960, 0.10, 0.012, 0.82))   # colorbar
    vps  = [bk.vxpow for bk in B]; vmin, vmax = minimum(vps), maximum(vps)
    grad = cgrad(:lipari10); tnorm(v) = vmax==vmin ? 0.5 : (v - vmin)/(vmax - vmin)
    boxes = (box1, box2, box3, box4)
    xlabs = (L"p_{\mathrm{f}}~[\mathrm{MPa}]", L"\log_{10}(\dot{\gamma})~[1/s]",
             L"\sigma_Y~[\mathrm{MPa}]", L"\log_{10}(\eta_\phi)~[\mathrm{Pa\,s}]")
    getx(k, bk) = k==1 ? (60e6.-bk.peff)./1e6 : #bk.peff./60e6 :
                  k==2 ? safelog10.(abs.(bk.exy)) :
                  k==3 ? bk.sigy./1e6 : safelog10.(bk.etaphi)

    # subplot 1 = blank parent canvas; data panels become subplots 2..5
    fig = plot(; size=figsize, legend=false, grid=false, framestyle=:none, ticks=nothing)
    for k in 1:4
        sp = k + 1
        plot!(fig, [NaN], [NaN]; inset=(1, boxes[k]), subplot=sp,
              framestyle=:box, legend=false, xlabel=xlabs[k],
              ylabel = (k==1 ? ylabel : ""),
              yformatter = (k==1 ? :auto : (_ -> "")))   # y-numbers on the left panel only
        for bk in B
            yy = xkey==:norm ? bk.yorel : bk.yphys
            plot!(fig, getx(k, bk), yy; subplot=sp,
                  color=grad[tnorm(bk.vxpow)], lw=1.3, markershape=:none)
        end
        ylims !== nothing && ylims!(fig[sp], ylims)
    end

    # colorbar as a manual heatmap in its own bbox = subplot 6 (full position control)
    ny = 128; yv = range(vmin, vmax, length=ny)
    zg = repeat(reshape(collect(yv), ny, 1), 1, 2)        # ny x 2 vertical gradient
    heatmap!(fig, [0.0, 1.0], yv, zg; inset=(1, boxcb), subplot=6,
             c=:lipari10, colorbar=false, framestyle=:box,
             xticks=false, xlims=(0,1), ymirror=true, ylabel=L"\log_{10}V")

    savefig(fig, fname*".pdf"); savefig(fig, fname*".png"); return fig
end

profiles(:norm, L"y/y_{\mathrm{size}}", (0,1),  "couette_profiles_normalized")
profiles(:phys, L"y-y_c~[\mathrm{m}]", (-1,0), "couette_profiles_physical")

println("Wrote: couette_tau_vs_V, couette_width_vs_V, couette_profiles_normalized, couette_profiles_physical (.pdf/.png)")

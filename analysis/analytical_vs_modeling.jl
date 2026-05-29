# =============================================================================
#  Analytical_vs_modeling_v2.jl   (paper-production version)
#  Numerical (H-MEC) vs analytical (LSA) shear-band thickness.
#  Theory: LSA_derivation_manuscript.tex  /  LSA_dispersion_analysis.jl
#
#     h = ell_c / Lambda_geom ,
#     ell_c = sqrt( (1-phi) eta_phi kp / (Lw etaf) ),  eta_phi = eta_s/phi
#     Lw    = f*p_e/(n(C+f*p_e)) - 1
#  Lambda_geom: ONE calibrated constant (linear marginal wavelength ->
#  nonlinear band width; analogue of the 6.9 in Barras & Brantut Eq. 8).
#
#  ON POROSITY (phi): the data are from a phi = 1% run; eta_s and p_e are
#  MEASURED from that run.  ell_c already contains phi (via eta_phi=eta_s/phi),
#  so in the h-vs-ell_c panel all points fall on one line regardless of phi.
#  Substituting phi=3%/5% into the formula while keeping the phi=1% eta_s is
#  physically inconsistent -> we do NOT overlay 3%/5% curves.
#
#  Inputs (run in the output dir):
#     EVO_Vslip.txt  EVO_viscosity.txt  EVO_press_eff.txt  fault.txt
#     (EVO_SigmaY.txt optional, used only if EVO_press_eff.txt is absent)
# =============================================================================

using DelimitedFiles, Printf, Statistics, Plots, LaTeXStrings, Measures
# Read H-MEC output and write figures in the shared results directory.
const RESULTS_DIR = get(ENV, "HMEC_RESULTS",
                        abspath(joinpath(@__DIR__, "..", "results")))
mkpath(RESULTS_DIR)
gr(); cd(RESULTS_DIR); print("\033c")

# ---- parameters (must match h_mec_1D.jl) ----
const phi   = 0.01
const C     = 5.0e6
const fric  = 0.3
const nexp  = 0.03
const kp    = 1e-18
const etaf  = 1e-3
const G     = 30e9
const bS    = 2.5e-11
const bF    = 4e-10
const VSLIP0= 0.5e-9
const eps0  = VSLIP0/1000
const bmin  = 5e-5
const V_LOC = 1e-8          # localisation filter: only slices with Vmax>V_LOC

GB  = G/phi; bdr=(1/GB+bS)/(1-phi); KBW=1-bS/bdr
KSK = (bdr-bS)/(bdr-bS+phi*(bF-bS)); S=bdr*KBW/KSK; D=kp/(etaf*S)

# ---- V thresholds (2 per decade) ----
VTH=Float64[]; for ex in -12:0; push!(VTH,1.0*10.0^ex); ex<0 && push!(VTH,2.8*10.0^ex); end
sort!(VTH)

# ---- read EVO ----
Vraw   = readdlm("EVO_Vslip.txt")[:, 3:end]
ETAraw = readdlm("EVO_viscosity.txt")[:, 3:end]
PEok   = isfile("EVO_press_eff.txt")
PEraw  = PEok ? readdlm("EVO_press_eff.txt")[:, 3:end] : zeros(size(Vraw))
yp     = vec(readdlm("fault.txt"))
Nt,Ny  = size(Vraw); yV = yp[1:Ny]
ic  = argmax(vec(sum(abs.(Vraw),dims=1)))
ieta= min(ic,size(ETAraw,2)); ipe=min(ic,size(PEraw,2))
SYraw = (!PEok && isfile("EVO_SigmaY.txt")) ? readdlm("EVO_SigmaY.txt")[:,3:end] : zeros(size(Vraw))

# ---- band thickness from max |dV/dy| ----
function band_thickness(Vk,y)
    nlen=length(Vk); icen=argmax(abs.(Vk))
    g=abs.((Vk[2:end].-Vk[1:end-1])./(y[2:end].-y[1:end-1]))
    it = icen>1 ? argmax(@view g[1:icen-1]) : 1
    ib = icen<nlen ? icen+argmax(@view g[icen:end])-1 : nlen-1
    return abs(0.5*(y[ib]+y[ib+1]) - 0.5*(y[it]+y[it+1]))
end

# ---- walk slices ----
Vmax=[maximum(abs.(Vraw[k,:])) for k in 1:Nt]
slc=Int[]; Vr=Float64[]; hn=Float64[]; etas=Float64[]; pe=Float64[]
nt=1
for k in 1:Nt
    global nt
    nt>length(VTH) && break
    if Vmax[k]>=VTH[nt]
        push!(slc,k); push!(Vr,Vmax[k]); push!(hn,band_thickness(Vraw[k,:],yV))
        push!(etas,ETAraw[k,ieta])
        if PEok
            push!(pe, PEraw[k,ipe])
        else
            tau=SYraw[k,min(ic,size(SYraw,2))]; gd=Vmax[k]/(2*bmin)
            sy0=tau/((gd/eps0)^nexp); push!(pe,max((sy0-C)/fric,1e5))
        end
        nt+=1
    end
end

# ---- LSA prediction ----
etaphi = etas ./ phi
Lw     = fric.*pe ./ (nexp.*(C .+ fric.*pe)) .- 1.0
loc    = (Vr .> V_LOC) .& (Lw .> 0)              # localised & unstable
ell    = sqrt.((1-phi).*etaphi[loc].*kp ./ (Lw[loc].*etaf))
hL     = hn[loc]
Lgeom  = exp(mean(log.(ell ./ hL)))              # single calibrated constant
hpred  = ell ./ Lgeom

# ---- diagnostics ----
slope(x,y)=(lx=log.(x);ly=log.(y);sum((lx.-mean(lx)).*(ly.-mean(ly)))/sum((lx.-mean(lx)).^2))
rcorr(x,y)=(lx=log.(x);ly=log.(y);
            sum((lx.-mean(lx)).*(ly.-mean(ly)))/sqrt(sum((lx.-mean(lx)).^2)*sum((ly.-mean(ly)).^2)))

println("="^78)
println("  H-MEC shear-band thickness:  numerical vs LSA prediction")
println("="^78)
@printf "  phi=%.3f  C=%.2e Pa  f=%.2f  n=%.3f  kp=%.1e m^2  etaf=%.1e Pa.s\n" phi C fric nexp kp etaf
@printf "  S = beta_dr*KBW/KSK = %.3e 1/Pa   D = kp/(etaf*S) = %.3e m^2/s\n" S D
@printf "  KBW(Biot)=%.4f   KSK(Skempton)=%.4f\n" KBW KSK
@printf "  fault centre: V column %d / %d\n" ic Ny
PEok || println("  [warn] EVO_press_eff.txt not found -> p_e reconstructed from EVO_SigmaY")
println("-"^78)

# ---- per-iteration table ----
@printf "%-5s %-11s %-11s %-11s %-9s %-7s %-11s %-11s %-8s\n" "slc" "V[m/s]" "h_num[m]" "eta_phi" "p_e[MPa]" "Lw" "ell_c[m]" "h_pred[m]" "pred/num"
println("-"^78)
for (m,k) in enumerate(findall(loc))
    @printf "%-5d %-11.3e %-11.3e %-11.3e %-9.2f %-7.2f %-11.3e %-11.3e %-8.3f\n" slc[k] Vr[k] hn[k] etaphi[k] pe[k]/1e6 Lw[k] ell[m] hpred[m] hpred[m]/hn[k]
end
println("-"^78)
@printf "  Lambda_geom (fitted)    = %.2f      (cf. 4*pi = %.2f)\n" Lgeom 4pi
@printf "  ell_c/h_num spread      = [%.2f, %.2f]\n" minimum(ell./hL) maximum(ell./hL)
@printf "  exponent h ~ eta_phi^p  : data %.3f , LSA %.3f   (pure sqrt = 0.5)\n" slope(etaphi[loc],hL) slope(etaphi[loc],ell)
@printf "  corr log(ell_c)-log(h)  : r = %.4f   (shape validation)\n" rcorr(ell,hL)
@printf "  localised slices kept   : %d / %d\n" count(loc) length(Vr)
println("="^78)

# =============================================================================
#  PLOTTING  (paper production)
# =============================================================================
# RGB colours
const C_NUM  = RGB(1.00, 1.00, 1.00)   # numerical markers (blue) RGB(0.16, 0.30, 0.66)
const C_EDGE = RGB(0.00, 0.00, 0.00)   # marker edge (orange) RGB(0.95, 0.55, 0.10)
const C_PRED = RGB(0.16, 0.30, 0.66)   # LSA prediction 

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

# numerical data: scatter (blue fill, orange edge); prediction: line only
mknum = (seriestype=:scatter, markercolor=C_NUM, markerstrokecolor=C_EDGE)
lpred = (color=C_PRED, lw=2, markershape=:none)
Vloc=Vr[loc]; ephiloc=etaphi[loc]

# (1) h vs log10 V  (linear y)
sp1 = sortperm(Vloc)

p1 = plot(log10.(Vloc)[sp1], hpred[sp1];
    xlabel=L"\log_{10}(V)~[\mathrm{m/s}]",
    ylabel=L"h~[\mathrm{m}]",
    xlims = (-9, -1),
    xticks = -9:2:-1,
    yticks = 0:0.04:0.2,
    label="Linear stability analysis",
    lpred...)

scatter!(p1, log10.(Vloc), hL;
    label="Numerical",
    mknum...)

# (2) h vs eta_phi  (log x, linear y)
sp2 = sortperm(ephiloc)

p2 = plot(ephiloc[sp2], hpred[sp2];
    xscale=:log10,
    xlabel=L"\eta_\phi~[\mathrm{Pa\,s}]",
    ylabel=L"h~[\mathrm{m}]",
    yticks = 0:0.04:0.2,
    xticks=([1e7, 1e9, 1e11, 1e13],[L"10^7", L"10^9", L"10^{11}", L"10^{13}"]),
    label="Linear stability analysis",
    legend=:topleft,
    lpred...)

scatter!(p2, ephiloc, hL;
    label="Numerical",
    mknum...)

annotate!(p1, -9.85, 0.193, text(L"\mathrm{a}", 15))
annotate!(p2, 1.0e5, 0.193, text(L"\mathrm{b}", 15))

#=
# (3) clean test: h vs ell_c  (log x, linear y), line slope 1/Lgeom
sp3 = sortperm(ell)

p3 = plot(ell[sp3], (ell ./ Lgeom)[sp3];
    xscale=:log10,
    xlabel=L"\ell_c~[\mathrm{m}]",
    ylabel=L"h~[\mathrm{m}]",
    yticks = 0:0.04:0.2,
    label=L"h=\ell_c/%$(round(Lgeom,digits=1))",
    legend=:topleft,
    lpred...)

scatter!(p3, ell, hL;
    label="Numerical",
    mknum...)

=#

fig = plot(p1, p2,
    layout=(1,3),
    size=(1100,420),
    left_margin=10mm,
    right_margin=6mm,
    bottom_margin=9mm,
    top_margin=6mm)


#savefig(fig,"Analytical_vs_modeling_v2.png")
savefig(fig,"analytical_vs_modeling.pdf")
savefig(fig,s"analytical_vs_modeling.svg")

display(fig)

# ---- CSV ----
open("analytical_vs_modeling.csv","w") do io
    @printf io "# slice, V[m/s], h_num[m], eta_phi[Pas], p_e[Pa], Lw, ell_c[m], h_pred[m]\n"
    for (m,k) in enumerate(findall(loc))
        @printf io "%d, %.4e, %.4e, %.4e, %.4e, %.4f, %.4e, %.4e\n" slc[k] Vr[k] hn[k] etaphi[k] pe[k] Lw[k] ell[m] hpred[m]
    end
end

println("\nWrote Analytical_vs_modeling_v2.png/.pdf/.csv")

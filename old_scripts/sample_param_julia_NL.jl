#!/usr/bin/env julia
using Printf
using DataStructures: OrderedDict
using NCDatasets
using DataFrames
using CSV, DataFrames
# using Plots
using Printf
# using LaTeXStrings
using LinearAlgebra
# using Profile
using Statistics 
using DifferentialEquations
using IterTools
using Base.Threads

Threads.nthreads()

# hr2s = 1/3600
# wm = 0.5 * hr2s
# wd = 0.05 * hr2s


# From Sharples and Ross 
wd = 1.2e-5 # m/s
wm = 1e-4 # m/s
growth_m = 1.08e-5 
growth_d = 3.6e-5 




# PO2 ~ 0.02 mg/L 


# gamma_n = 0.0425. # mmol nutrient/ m3 
# nutrient content / cell --> 1e-9 nutrient/cell 

# Umax Maximum C-specific nitrogen uptake rate 0.95 mg N (mg C d)–1

# cells / m3 


function create_range_over(value::Float64, mult::Int, N::Int)
    return LinRange(value/mult, value*mult, N)
end



mult = 10 
N = 4

# # old 
ld = 0.006 * hr2s
lm  = 0.004 * hr2s
# growth_m = 0.05 * hr2s
# growth_d = 0.008 * hr2s



p1 = 0.02 # mg P / L # (from data)

# uptake = umax * (1 - Q/Qmax) * (N / (N + KN))

# From Reynolds 

diatom_v_umax = 0.51            # µmol P / (mol cell C second)  (Asterionella formosa)
half_sat_ku_diatom = 1.9        # 1.9- 2.8 , µmol P / L (Asterionella formosa)
half_sat_ku_microcystis = 0.3   # µmol P / L   (Microcystis aeruginosa)
microcystis_v_umax = 1.95       # µmol P / (mol cell C second)     (Microcystis aeruginosa)

# Unit conversion -> mg P/L --> µmol P 
mgP_2_µmol = 32.27  # µmol P / L
µmol_2_mgP = 1 / mgP_2_µmol


# Dry weight 
# carbon makes up roughly 50% of dry weight
microcystis_weight = 32 / 2  # pg/cell         
diatom_weight = 292 /2  # 292- 349 pg/cell


# # Loss 
# loss_diatoms = create_range_over(0.006 * hr2s, mult, 2)
# loss_microcystis = create_range_over(0.004 * hr2s, mult, 2)

# # Swimming / sinking 
# velocity_diatoms = create_range_over(wd, mult, N)
# velocity_microcystis = create_range_over(wm, mult, N)

# # Growth
# growth_diatoms = create_range_over(0.05*hr2s, mult, N)
# growth_microcystis = create_range_over(0.008*hr2s, mult, N)

# Diffusivity 
# kappa = np.logspace(-2, -9, N, base=10)

Kappa = logrange(10^-1, 10^-7, N)

# Ratio between depths
Depth_ratio = LinRange(0.1, 3, N) 
Total_depth = LinRange(4, 11, N) 
Drawdown = LinRange(1, 100, N)

matrix_out_m = zeros(N, N, N)
matrix_out_d = zeros(N, N, N)


# Uptake of nitrogen 
uptake_nitrogen = (0.95 * 1/86400) # mg N / (mg C s)
cell2mgC = 2e-6 # mg C / cell


function system!(du, u, p, t)
    m1, m2, d1, d2, n1, n2 = u
    h1, h2, kappa, wm, lm, growth_m, wd, ld, growth_d, draw_d = p

    # Half saturation constant for nutrient-limited production of algae
    gamma_m = 3 # mg N/m^3  # https://www.int-res.com/articles/meps2007/347/m347p021.pdf
    gamma_d = 3 # mg N/m^3  or mg P/ m^3

    # Add in monod function 
    monod_m = n1 / (n1 + gamma_m)  
    monod_d = n1 / (n1 + gamma_d)

    # uptake .. .# From Reynolds 
    # Units : mg P / (mol cell C second)

    diatom_v_umax = 0.51            # µmol P / (mol cell C second)  (Asterionella formosa)
    half_sat_ku_diatom = 1.9        # 1.9- 2.8 , µmol P / L (Asterionella formosa)
    half_sat_ku_microcystis = 0.3   # µmol P / L   (Microcystis aeruginosa)
    microcystis_v_umax = 1.95       # µmol P / (mol cell C second)     (Microcystis aeruginosa)
    
    uptake_d = (diatom_v_umax * µmol_2_mgP) *  ((n1 / (n1 + half_sat_ku_diatom)))
    uptake_m = (microcystis_v_umax * µmol_2_mgP) * (n1 / (n1 + half_sat_ku_microcystis)))

    # Dry weight 
    # carbon makes up roughly 50% of dry weight
    # microcystis_weight = 32 / 2  # pg/cell 
    # diatom_weight = 292 /2  # 292- 349 pg/cell

    # uptake_d = diatom_v_umax *  (n1 / (n1 + half_sat_ku_diatom))
    # uptake_m = microcystis_v_umax * (n1 / (n1 + half_sat_ku_microcystis))

    uptake_d = uptake_d * 2.46e-5 * d1 
    uptake_m = uptake_m * 5.5e-13 * m1

    du[1] =  1/h1 * wm * m2 + kappa/h1 * (m2 - m1) - lm * m1 + (growth_m * monod_m) 
    du[2] = -1/h2 * wm * m2 + kappa/h2 * (m1 - m2) - lm * m2

    du[3] = -1/h1 * wd * d1 + kappa/h1 * (d2 - d1) - ld * d1 + (growth_d * monod_d)
    du[4] =  1/h2 * wd * d1 + kappa/h1 * (d1 - d1) - ld * d2

    du[5] = kappa/h1 * (n2 - n1) - uptake_m - uptake_d
    du[6] = 0.001 #kappa/h2 * (n1 - n2)
end

T = 3600*10
tspan = (0, T) 

# println("Running $(length(index_tuples)) simulations...")


# Let's loop over kappa, R, and total depth 
# index_tuples = IterTools.product(1:N, 1:N, 1:N, 1:N)
println("Running simulation for T = $T seconds...")
for idx in IterTools.product(1:N, 1:N, 1:N)
    i, j, k = idx
    kappa = Kappa[i]
    ratio = Depth_ratio[j]
    depth = Total_depth[k]
    draw_d = 3

    h2 = depth/(1 + ratio)
    h1 = ratio * h2 
    p = (h1, h2, kappa, wm, lm, growth_m, wd, ld, growth_d, draw_d)

    init_n = 0.02 # mg P / L
    init  = 8e5  # 40,000 to 120,000 cells per liter
    init = [init*(h1/depth), init*(h2/depth), init*(h1/depth), init*(h2/depth), h1*init_n, h2*init_n]
    # init = [h1, h2, h1, h2, h1*init_n, h2*init_n]
    prob = ODEProblem(system!, init, tspan, p)
    sol = solve(prob, saveat=range(0, T, length=1200), abstol=1e-8, reltol=1e-8)
    m1, m2, d1, d2, n1, n2 = sol.u[end]
    matrix_out_m[i, j, k] = m1 + m2 
    matrix_out_d[i, j, k] = d1 + d2
end
println("Finished running simulation for T = $T seconds")

fout = "population_dataset_units.nc"
ds = NCDataset(fout,"c")

defDim(ds, "ratio", N)
defDim(ds, "kappa", N) 
defDim(ds, "depth", N)
# defDim(ds, "drawdown", N)

v = defVar(ds, "ratio", Float32, ("ratio",), attrib = OrderedDict("units" => "[-]"))
v[:] = Depth_ratio 

v = defVar(ds, "kappa", Float32, ("kappa",), attrib = OrderedDict("units" => "m^2/s"))
v[:] = Kappa

v = defVar(ds, "depth", Float32, ("depth",), attrib = OrderedDict("units" => "m"))
v[:] = Total_depth

# v = defVar(ds, "drawdown", Float32, ("drawdown",), attrib = OrderedDict("units" => "[-]"))
# v[:] = Drawdown
 
v = defVar(ds, "biomass_d", Float64,("kappa","ratio", "depth"), attrib = OrderedDict(
"units" =>  "biomass", "long_name" => "biomass_of_diatoms"))
v[:,:,:] = matrix_out_d ; 

v = defVar(ds, "biomass_m", Float64,("kappa","ratio", "depth"), attrib = OrderedDict(
"units" =>  "biomass", "long_name" => "biomass_of_microcystis"))
v[:,:,:] = matrix_out_m ; 



# for var in var2save
#     v = defVar(ds, var, Float64,("z","time"), attrib = OrderedDict(
#     "units" =>  units_dict[var], "long_name" => var2name[var]))
#     v[:,:] = output[var];
# end


# # n = norm(grad)
# # println("Norm is $(n)")
# eps = step_size #0.2 # 0.1 working well 1 #0.5 #0.01 #2 #15 #2e4 * n 
# println("Step size is $(eps)")
# # [1] Adjust growth rates 
# grad = ds_algae["gamma1"][:,:] .* output["lambda1"]
# new_gamma = @. ds_algae["gamma1"][:,:] - grad*eps 
# v1 = defVar(ds, "gamma1", Float64,("z","time"), attrib = OrderedDict(
#     "units" =>  "-", "long_name" => "gradient descent parameterized growth1"))
# v1[:,:] = new_gamma;



# println("Norm of grad is $(norm(grad))")
#     # [2] Adjust growth rates 
# grad = ds_algae["gamma2"][:,:] .* output["lambda2"]
# new_gamma = @. ds_algae["gamma2"][:,:] - grad*eps 
# v2 = defVar(ds, "gamma2", Float64,("z","time"), attrib = OrderedDict(
#     "units" =>  "-", "long_name" => "gradient descent parameterized growth2"))
# v2[:,:] = new_gamma;


print("Saved $fout \n")
close(ds)
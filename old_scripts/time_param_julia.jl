#!/usr/bin/env julia
using Printf
using DataStructures: OrderedDict
using NCDatasets
using DataFrames
using CSV, DataFrames
using Plots
using Printf
using LaTeXStrings
using LinearAlgebra
# using Profile
using Statistics 
using DifferentialEquations
using IterTools
using Base.Threads

Threads.nthreads()

hr2s = 1/3600
wm = 0.5 * hr2s
wd = 0.05 * hr2s

# function create_range_over(value::Float64, mult::Int, N::Int)
#     return LinRange(value/mult, value*mult, N)
# end


ld = 0.006 * hr2s
lm  = 0.004 * hr2s

mult = 10 
N = 17

growth_d = 0.05 * hr2s
growth_m = 0.008 * hr2s
         
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

Kappa = logrange(10^-2, 10^-5, N)

# Ratio between depths
Depth_ratio = LinRange(0.1, 3, N) 
photic_ratio = LinRange(0.1, 0.9, N)
Total_depth = LinRange(4, 11, N) 
Drawdown = LinRange(1, 100, N)

NT = 300 
matrix_out_m1 = zeros(N, N, N, NT)
matrix_out_m2 = zeros(N, N, N, NT)
matrix_out_d1 = zeros(N, N, N, NT)
matrix_out_d2 = zeros(N, N, N, NT)
matrix_out_n1 = zeros(N, N, N, NT)



function system!(du, u, p, t)
    m1, m2, d1, d2, n1 = u
    h1, h2, kappa, wm, lm, growth_m, wd, ld, growth_d, n2 = p

            # advection     diffusion +         loss        growth
    du[1] =  (wm/h1)*m2   +   kappa/h1*(m2 - m1)  - lm*m1 + (m1*growth_m*n1)
    du[2] = -(wm/h2)*m2   +   kappa/h2*(m1 - m2)  - lm*m2

    du[3] = -(wd/h1)*d1 + kappa/h1*(d2 - d1) - ld*d1 + (d1*growth_d*n1)
    du[4] =  (wd/h2)*d1 + kappa/h2*(d1 - d1) - ld*d2

    du[5] = kappa/h1 * (n2 - n1) - (growth_m*n1*m1) - (growth_d*n1*d1)
    # du[6] = 1e-5 #kappa/h2 * (n1 - n2)
end

T = 3600*200 # 30
tspan = (0, T) 
    # T = 3600*30
    # tspan = (0, T) 
# println("Running $(length(index_tuples)) simulations...")


# Let's loop over kappa, R, and total depth 
# index_tuples = IterTools.product(1:N, 1:N, 1:N, 1:N)
println("Running simulation for T = $T seconds...")
for idx in IterTools.product(1:N, 1:N, 1:N)
    i, j, k = idx
    kappa = Kappa[i]
    # ratio = Depth_ratio[j]
    pratio = photic_ratio[j]
    depth = Total_depth[k]
    draw_d = 10 #3

    h1 = depth * pratio 
    h2 = depth - h1

    init_n = 1 #1e-2 # was 1e-6
    init  = 1e-3 #1e-1  # was 1e-2

    p = (h1, h2, kappa, wm, lm, growth_m, wd, ld, growth_d, init_n)

               #  m1            m2                 d1            d2             n1         
    init = [init*(h1/depth), init*(h2/depth), init*(h1/depth)*2, init*(h2/depth), (h1/depth)*init_n]
    prob = ODEProblem(system!, init, tspan, p)
    sol = solve(prob, saveat=range(0, T, length=NT), abstol=1e-9, reltol=1e-9, maxiters=20000)
    # m1, m2, d1, d2, n1, n2 = sol.u[:]
    m1 = [u[1] for u in sol.u]
    m2 = [u[2] for u in sol.u]
    d1 = [u[3] for u in sol.u]
    d2 = [u[4] for u in sol.u]
    n1 = [u[5] for u in sol.u]

    matrix_out_m1[i, j, k, :] = m1
    matrix_out_m2[i, j, k, :] = m2 
    matrix_out_d1[i, j, k, :] = d1 
    matrix_out_d2[i, j, k, :] = d2
    matrix_out_n1[i, j, k, :] = n1 
end
println("Finished running simulation for T = $T seconds")

fout = "population_dataset_time_INIT1.nc"
ds = NCDataset(fout,"c")

defDim(ds, "ratio", N)
defDim(ds, "kappa", N) 
defDim(ds, "depth", N)
defDim(ds, "t", NT)
# defDim(ds, "drawdown", N)

v = defVar(ds, "ratio", Float32, ("ratio",), attrib = OrderedDict("units" => "[-]"))
v[:] = photic_ratio #Depth_ratio 

v = defVar(ds, "kappa", Float32, ("kappa",), attrib = OrderedDict("units" => "m^2/s"))
v[:] = Kappa

v = defVar(ds, "depth", Float32, ("depth",), attrib = OrderedDict("units" => "m"))
v[:] = Total_depth

v = defVar(ds, "t", Int, ("t",), attrib = OrderedDict("units" => "s"))
v[:] = collect(range(0, T, length=NT))
print(range(0, T, length=NT))

# v = defVar(ds, "drawdown", Float32, ("drawdown",), attrib = OrderedDict("units" => "[-]"))
# v[:] = Drawdown
 
v = defVar(ds, "d1", Float64,("kappa","ratio", "depth", "t"), attrib = OrderedDict(
"units" =>  "biomass", "long_name" => "biomass_of_diatoms"))
v[:,:,:,:] = matrix_out_d1 ; 

v = defVar(ds, "d2", Float64,("kappa","ratio", "depth", "t"), attrib = OrderedDict(
"units" =>  "biomass", "long_name" => "biomass_of_diatoms"))
v[:,:,:,:] = matrix_out_d2 ; 

v = defVar(ds, "m1", Float64,("kappa","ratio", "depth", "t"), attrib = OrderedDict(
"units" =>  "biomass", "long_name" => "biomass_of_microcystis"))
v[:,:,:,:] = matrix_out_m1 ; 

v = defVar(ds, "m2", Float64,("kappa","ratio", "depth", "t"), attrib = OrderedDict(
"units" =>  "biomass", "long_name" => "biomass_of_microcystis"))
v[:,:,:,:] = matrix_out_m2 ; 

v = defVar(ds, "n1", Float64,("kappa","ratio", "depth", "t"), attrib = OrderedDict(
"units" =>  "nutrient", "long_name" => "nutrient"))
v[:,:,:,:] = matrix_out_n1 ; 


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
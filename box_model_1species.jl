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

hr2s = 1/3600
# From Sharples and Ross 
wd = 1.2e-5 # m/s
wm = 1e-4 # m/s
growth_m = 1.08e-5 
growth_d = 3.6e-5 


function create_range_over(value::Float64, mult::Int, N::Int)
    return LinRange(value/mult, value*mult, N)
end


mult = 10 
N = 15

# # old 
ld = 0.006 * hr2s
lm  = 0.004 * hr2s

# growth_m = 0.05 * hr2s
# growth_d = 0.008 * hr2s

p1 = 0.02 # mg P / L # (from data)



# Diffusivity 
# kappa = np.logspace(-2, -9, N, base=10)

Kappa = logrange(10^-1, 10^-7, N)

# Ratio between depths
Depth_ratio = LinRange(0.1, 3, N) 
Total_depth = LinRange(4, 11, N) 
Drawdown = LinRange(1, 100, N)

matrix_out_m = zeros(N, N, N)
matrix_out_d = zeros(N, N, N)


function system!(du, u, p, t)
    m1, m2, n1 = u
    h1, h2, kappa, wm, lm  = p


    # uptake_d = uptake_d * 2.46e-5 * d1 
    # uptake_m = uptake_m * 5.5e-13 * m1

    growth_m = 1.08e-5              # 1/s 
    half_sat_m1 = 0.18              # µmol P / L
    max_uptake_m1 = 2.23e-12        # µmol P / cell s 
    half_sat_n1_m = 1.23            # µmol P / L

    # m1     advection         diffusion       loss                  growth
    du[1] =  1/h1*wm*m2 + kappa/h1*(m2 - m1) - lm*m1 + (growth_m * m1*(n1/(half_sat_m1 + n1))) 

    # m2       advection     diffusion          loss        
    du[2] = -1/h2*wm*m2 + kappa/h2*(m1 - m2) - lm*m2

    # n1 
    n2 = 0.05       
                # diffusion                 # uptake     
    du[3] = kappa/h1 *(n2 - n1) - m1*max_uptake_m1*(n1 / (n1 + half_sat_n1_m))
end

T = 3600*10 #*10
tspan = (0, T) 

NT = 300 
matrix_out_m1 = zeros(N, N, N, NT)
matrix_out_m2 = zeros(N, N, N, NT)
matrix_out_d1 = zeros(N, N, N, NT)
matrix_out_d2 = zeros(N, N, N, NT)
matrix_out_n1 = zeros(N, N, N, NT)


# Let's loop over kappa, R, and total depth 
# index_tuples = IterTools.product(1:N, 1:N, 1:N, 1:N)
println("Running simulation for T = $T seconds...")
for idx in IterTools.product(1:N, 1:N, 1:N)
    println("1")
    i, j, k = idx

    println("2")
    kappa = Kappa[i]
    ratio = Depth_ratio[j]
    depth = Total_depth[k]

    println("3")
    h2 = depth/(1 + ratio)
    h1 = ratio * h2 
    p = (h1, h2, kappa, wm, lm)

    println("4")
    init_n = 0.129 # µmol P / L
    init  = 8e5  # 40,000 to 120,000 cells per liter
    init = [init, init,  init_n]
    # init = [h1, h2, h1, h2, h1*init_n, h2*init_n]
    println("5")
    prob = ODEProblem(system!, init, tspan, p)

    println("6")
    sol = solve(prob, saveat=range(0, T, length=NT), abstol=1e-8, reltol=1e-8)
    m1, m2, n1 = sol.u[end]
    matrix_out_m[i, j, k] = m1 + m2 


    m1 = [u[1] for u in sol.u]
    m2 = [u[2] for u in sol.u]
    n1 = [u[3] for u in sol.u]

    matrix_out_m1[i, j, k, :] = m1
    matrix_out_m2[i, j, k, :] = m2 
    matrix_out_n1[i, j, k, :] = n1 
    # matrix_out_d[i, j, k] = d1 + d2
end
println("Finished running simulation for T = $T seconds")

fout = "population_dataset_1species.nc"
fout = "population_dataset_time_INIT1.nc"
ds = NCDataset(fout,"c")

defDim(ds, "ratio", N)
defDim(ds, "kappa", N) 
defDim(ds, "depth", N)
defDim(ds, "t", NT)
# defDim(ds, "drawdown", N)

v = defVar(ds, "ratio", Float32, ("ratio",), attrib = OrderedDict("units" => "[-]"))
v[:] = Depth_ratio #Depth_ratio 

v = defVar(ds, "kappa", Float32, ("kappa",), attrib = OrderedDict("units" => "m^2/s"))
v[:] = Kappa

v = defVar(ds, "depth", Float32, ("depth",), attrib = OrderedDict("units" => "m"))
v[:] = Total_depth

v = defVar(ds, "t", Int, ("t",), attrib = OrderedDict("units" => "s"))
v[:] = collect(range(0, T, length=NT))
print(range(0, T, length=NT))

# v = defVar(ds, "drawdown", Float32, ("drawdown",), attrib = OrderedDict("units" => "[-]"))
# v[:] = Drawdown
 

v = defVar(ds, "m1", Float64,("kappa","ratio", "depth", "t"), attrib = OrderedDict(
"units" =>  "biomass", "long_name" => "biomass_of_microcystis"))
v[:,:,:,:] = matrix_out_m1 ; 

v = defVar(ds, "m2", Float64,("kappa","ratio", "depth", "t"), attrib = OrderedDict(
"units" =>  "biomass", "long_name" => "biomass_of_microcystis"))
v[:,:,:,:] = matrix_out_m2 ; 

v = defVar(ds, "n1", Float64,("kappa","ratio", "depth", "t"), attrib = OrderedDict(
"units" =>  "nutrient", "long_name" => "nutrient"))
v[:,:,:,:] = matrix_out_n1 ; 


print("Saved $fout \n")
close(ds)
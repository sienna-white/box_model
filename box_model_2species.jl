#!/usr/bin/env julia
using Printf
using DataStructures: OrderedDict
using NCDatasets
using DataFrames
using CSV, DataFrames
# module load  julia/1.11.7 
using Printf
using LinearAlgebra
using Statistics 
using DifferentialEquations
using IterTools
using Base.Threads

Threads.nthreads()

hr2s = 1/3600
# # From Sharples and Ross 
# wd = 1.2e-5 # m/s
# wm = 1e-4 # m/s
# growth_m = 1.08e-5 
# growth_d = 3.6e-5 

# Settling / swimming velocities
wm = 0.5 * hr2s
wd = 0.05 * hr2s

# Respiration  
ld = 0.008 * hr2s # 0.006
lm  = 0.004 * hr2s

# half saturation for growth 
γ_m = 0.18 # half saturation constant for microcystis growth
γ_d = 0.097 # half saturation constant for diatom growth

# maximum uptake rates 
uptake_d = 4e-12 # nutrient uptake rate for diatoms
uptake_m = 2.23e-12 # [µmol P / cell s] nutrient uptake rate for microcystis

# half saturation for uptake 
γ_nm = 1.23    # [µmol P / L] half saturation constant for nutrient uptake microcystis
γ_nd = 2.8     # [µmol P / L] half saturation constant for nutrient uptake diatoms    

# Growth
# α = 0.008 * hr2s
α = 0.27/24  * hr2s # 0.27 per day 0.27/24
β = 2/24  * hr2s    # 2 per day 
# β = 0.05 * hr2s

# uptake_d = uptake_d * 2.46e-5 * d1 
# uptake_m = uptake_m * 5.5e-13 * m1

# growth_m = 1.08e-5              # 1/s 
# γ_m = 0.18              # µmol P / L
# max_uptake_m1 = 2.23e-12        # µmol P / cell s 
# γ_mn = 1.23            # µmol P / L





function create_range_over(value::Float64, mult::Int, N::Int)
    return LinRange(value/mult, value*mult, N)
end


mult = 10 
N = 20



# p1 = 0.02 # mg P / L # (from data)

# Diffusivity 
Kappa = logrange(10^-1, 10^-7, N)

# Ratio between depths
Depth_ratio = LinRange(0.05, 0.95, N) 
Total_depth = LinRange(4, 15, N) 

matrix_out_m = zeros(N, N, N)
matrix_out_d = zeros(N, N, N)

function system!(du, u, p, t)

    # Unpack state variables 
    m1, m2, d1, d2, n1 = u

    # Unpack parameters
    h1, h2, κ, wm, wd, lm, ld, α, β, γ_m, γ_d, uptake_m, uptake_d, γ_nm, γ_nd = p 

    # bottom nutrients 
    n2 = 3 # 0.129

    # println(h1, h2, κ, wm, wd, lm, ld, α, β, γ_m, γ_d, uptake_m, uptake_d, γ_nm, γ_nd)
    
    # m1     advection    diffusion         loss   growth
    du[1] =  (wm/h1)*m2 + κ/h1*(m2 - m1) - lm*m1 + (α * m1*(n1/(γ_m + n1))) 
    # m2     advection    diffusion        loss        
    du[2] = -(wm/h2)*m2 + κ/h2*(m1 - m2) - lm*m2
    # d1     advection    diffusion        loss    growth
    du[3] = -(wd/h1)*d1 + κ/h1*(d2 - d1) - ld*d1 + β*d1*(n1/(γ_d + n1))
    # d2     advection    diffusion        loss
    du[4] =  (wd/h2)*d1 + κ/h2*(d1 - d2) - ld*d2
    # n1     diffusion         # m1 uptake                         #d1 uptake                
    du[5] =  κ/h1 *(n2 - n1) - m1*uptake_m*(n1/(n1 + γ_nm)) - d1*uptake_d*(n1/(n1 + γ_nd))
end

T = 3600*10 #*10
tspan = (0, T) 

NT = 300 
matrix_out_m1 = zeros(N, N, N, NT)
matrix_out_m2 = zeros(N, N, N, NT)
matrix_out_d1 = zeros(N, N, N, NT)
matrix_out_d2 = zeros(N, N, N, NT)
matrix_out_n1 = zeros(N, N, N, NT)


# Let's loop over κ, R, and total depth 
# index_tuples = IterTools.product(1:N, 1:N, 1:N, 1:N)
println("Running simulation for T = $T seconds...")


for idx in IterTools.product(1:N, 1:N, 1:N)
    i, j, k = idx

    κ = Kappa[i]
    ratio = Depth_ratio[j]
    depth = Total_depth[k]
    # h2 = depth/(1 + ratio)
    h1 = ratio * depth
    h2 = depth - h1

     # Pack parameters 

    p = (h1, h2, κ, wm, wd, lm, ld, α, β, γ_m, γ_d, uptake_m, uptake_d, γ_nm, γ_nd)

    init_n = 2 # µmol P / L
    init  = 5e6  # 40,000 to 120,000 cells per liter
    init = [init, init, init, init, init_n]
    # init = [h1, h2, h1, h2, h1*init_n, h2*init_n]

    prob = ODEProblem(system!, init, tspan, p)


    sol = solve(prob, saveat=range(0, T, length=NT), abstol=1e-8, reltol=1e-8)
    m1, m2, d1, d2, n1 = sol.u[end]


    matrix_out_m[i, j, k] = m1 + m2 
    matrix_out_d[i, j, k] = d1 + d2 

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

@info "Saving results to NetCDF file..."
fout = "population_dataset_time_2s_3umol.nc"
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

v = defVar(ds, "d1", Float64,("kappa","ratio", "depth", "t"), attrib = OrderedDict(
"units" =>  "biomass", "long_name" => "surface diatoms"))
v[:,:,:,:] = matrix_out_d1 ; 

v = defVar(ds, "d2", Float64,("kappa","ratio", "depth", "t"), attrib = OrderedDict(
"units" =>  "biomass", "long_name" => "bottom diatoms"))
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


println("Saved $fout \n")
close(ds)
#!/usr/bin/env julia
using Printf
using DataStructures: OrderedDict
using NCDatasets
using DataFrames
using CSV, DataFrames
using Printf
using LinearAlgebra
using Statistics 
using DifferentialEquations
using IterTools
using Base.Threads
using Sundials 
# module load  julia/1.11.7 


hr2s = 1/3600
day2s = 1/(3600*24)
# # From Sharples and Ross 
# wd = 1.2e-5 # m/s
# wm = 1e-4 # m/s
# growth_m = 1.08e-5 
# growth_d = 3.6e-5 
println("Using $(Threads.nthreads()) threads.")
# Settling / swimming velocities
wm = 0.5 * hr2s
wd = 0.05 * hr2s

# Respiration  
ld  = 0.008 * hr2s * 1.1 # 0.006
lm  = 0.004 * hr2s

# half saturation for growth (nitrogen)
γ_m = 0.6 #$0.1 # half saturation constant for microcystis growth (0.1–37.8)
γ_d = 1.8 #0.097 # half saturation constant for diatom growth


# half saturation for uptake  (nitrogen)
γ_nm = 0.6    # [µmol N / L] half saturation constant for nutrient uptake microcystis
γ_nd = 0.483 # - 7     # [µmol N / L] half saturation constant for nutrient uptake diatoms    

# maximum uptake rates  (nitrate)
uptake_d =  4.16e-12 * 17 # nutrient uptake rate for diatoms
uptake_m = 2.23e-12 * 16 # [µmol P / cell s] nutrient uptake rate for microcystis



# # half saturation for growth (phosphorus)
# γ_m = 0.18 # half saturation constant for microcystis growth
# γ_d = 0.097 # half saturation constant for diatom growth

# # maximum uptake rates  (phosphorus)
# uptake_d = 4.16e-12 # nutrient uptake rate for diatoms
# uptake_m = 2.23e-12 # [µmol P / cell s] nutrient uptake rate for microcystis

# # half saturation for uptake  (phosphorus)
# γ_nm = 1.23    # [µmol P / L] half saturation constant for nutrient uptake microcystis
# γ_nd = 0.7     # [µmol P / L] half saturation constant for nutrient uptake diatoms    
#         # ^ (0.7 - 2.8) 

# Growth
# α = 0.008 * hr2s
α = 0.8 * day2s       # 0.1–1.0
β = 1.5 * day2s       # 2 per day 
# β = 0.05 * hr2s

# uptake_d = uptake_d * 2.46e-5 * d1 
# uptake_m = uptake_m * 5.5e-13 * m1


function create_range_over(value::Float64, mult::Int, N::Int)
    return LinRange(value/mult, value*mult, N)
end


mult = 10 
N = 11



# p1 = 0.02 # mg P / L # (from data)

# Diffusivity 
Kappa = logrange(10^-1, 10^-8, N)
# Ratio between depths
Depth_ratio = LinRange(0.05, 0.95, N) 
# Depth of water column
Total_depth = LinRange(4, 15, N) 
# Initial population of cells
Starting_population = LinRange(1, 10, N) # cells / L

# Nutrient concentration in bottom layer 
Available_nutrients = LinRange(5, 60, N) # µmol P / L


function system!(du, u, p, t)

    # Unpack state variables 
    m1, m2, d1, d2, n1 = u

    # Unpack parameters
    h1, h2, κ, wm, wd, lm, ld, α, β, γ_m, γ_d, uptake_m, uptake_d, γ_nm, γ_nd, n2 = p 

    # bottom nutrients 
    # n2 = 20 # 0.129

    # m1     advection    diffusion         loss        growth
    du[1] =  (wm/h1)*m2 + κ/h1*(m2 - m1) - lm*m1 + (α * m1*(n1/(γ_m + n1))) 
    # m2     advection    diffusion        loss        
    du[2] = -(wm/h2)*m2 + κ/h2*(m1 - m2) - lm*m2
    # d1     advection    diffusion        loss         growth
    du[3] = -(wd/h1)*d1 + κ/h1*(d2 - d1) - ld*d1 + β * d1*(n1/(γ_d + n1))

    # d2     advection    diffusion        loss
    du[4] =  (wd/h2)*d1 + κ/h2*(d1 - d2) - ld*d2
    # n1     diffusion         # m1 uptake                         #d1 uptake  
    du[5] =  κ/h1 *(n2 - n1) - m1*uptake_m*(n1/(γ_m + n1)) - d1*uptake_d*(n1/(γ_d + n1))   
    # du[5] =  κ/h1 *(n2 - n1) - m1*uptake_m*((n1*h1)/((n1*h1) + γ_nm)) - d1*uptake_d*((n1*h1)/((n1*h1) + γ_nd))
    # println("t = $t, m1 = $m1, m2 = $m2, d1 = $d1, d2 = $d2, n1 = $n1")
end

T = 3600*12 #*10
tspan = (0, T) 

NT = 500 
matrix_out_m1 = zeros(N, N, N, N, N, NT)
matrix_out_m2 = zeros(N, N, N, N, N, NT)
matrix_out_d1 = zeros(N, N, N, N, N, NT)
matrix_out_d2 = zeros(N, N, N, N, N, NT)
matrix_out_n1 = zeros(N, N, N, N, N, NT)


# Let's loop over κ, R, and total depth 
# index_tuples = IterTools.product(1:N, 1:N, 1:N, 1:N)
println("Running simulation for T = $T seconds...")

t1 = Dates.now()
println("0Time = ", t1)

    # chla2cell_mc =  1e-6/0.36  # ug chl-a/ml --> pg chl-a/ml --> 0.36  pg chl-a/cell Microcystis
    # chla2cell_diatom = 1e-6/4 # ug chl-a/ml --> pg chl-a/ml --> 4 pg chl-a/cell Diatom
    # cell2chla_mc = 1/chla2cell_mc
    # cell2chla_diatom = 1/chla2cell_diatom
    # @info "Starting with $(init*0.36*1e-6) ug/L chl-a for microcystis and $(init*4*1e-6) ug/L chl-a for diatoms"

@threads for idx in IterTools.product(1:N, 1:N, 1:N, 1:N, 1:N)
    a, b, c, d, e = idx # Kappa, Ratio, Depth, Initial population, Nutrient concentration

    κ = Kappa[a]
    ratio = Depth_ratio[b]
    depth = Total_depth[c]
    init = Starting_population[d]
    n2 = Available_nutrients[e]


    # h2 = depth/(1 + ratio)
    h1 = ratio * depth
    h2 = depth - h1

    init  = init * 1e6  # 40,000 to 120,000 cells per liter

    init_n = n2 #30 # µmol P / L

     # Pack parameters 
    p = (h1, h2, κ, wm, wd, lm, ld, α, β, γ_m, γ_d, uptake_m, uptake_d, γ_nm, γ_nd, init_n)

    # Exit if any variable hits < 0 
    condition(u, t, integrator) = minimum(u)  # triggers when crosses 0
    function affect!(integrator)
        terminate!(integrator)
    end

    cb = ContinuousCallback(condition, affect!)


    init = [init, init, init, init, init_n]

    prob = ODEProblem(system!, init, tspan, p)

    sol = solve(prob, saveat=range(0, T, length=NT), abstol=1e-9, reltol=1e-9, CVODE_BDF(), callback=cb)
    m1, m2, d1, d2, n1 = sol.u[end]

    # matrix_out_m[i, j, k] = m1 + m2 
    # matrix_out_d[i, j, k] = d1 + d2 

    m1 = [u[1] for u in sol.u]
    m2 = [u[2] for u in sol.u]
    d1 = [u[3] for u in sol.u]
    d2 = [u[4] for u in sol.u]
    n1 = [u[5] for u in sol.u]

    lvar = length(m1) 
    if lvar <= 500
        matrix_out_m1[a, b, c, d, e, 1:lvar] = m1
        matrix_out_m2[a, b, c, d, e, 1:lvar] = m2 
        matrix_out_d1[a, b, c, d, e, 1:lvar] = d1
        matrix_out_d2[a, b, c, d, e, 1:lvar] = d2 
        matrix_out_n1[a, b, c, d, e, 1:lvar] = n1 
    else 
        println("Length of solution: $lvar")
    end 
end

println("Finished running simulation for T = $T seconds")

println("Time2 = $(Dates.now() - t1)")
@info "Saving results to NetCDF file..."
fout = "population_dataset_NO3_30umol.nc"
ds = NCDataset(fout,"c")

defDim(ds, "ratio", N)
defDim(ds, "kappa", N) 
defDim(ds, "depth", N)
defDim(ds, "n2", N)
defDim(ds, "init", N)

defDim(ds, "t", NT)
# defDim(ds, "drawdown", N)

v = defVar(ds, "ratio", Float32, ("ratio",), attrib = OrderedDict("units" => "[-]"))
v[:] = Depth_ratio #Depth_ratio 

v = defVar(ds, "kappa", Float32, ("kappa",), attrib = OrderedDict("units" => "m^2/s"))
v[:] = Kappa

v = defVar(ds, "depth", Float32, ("depth",), attrib = OrderedDict("units" => "m"))
v[:] = Total_depth

v = defVar(ds, "n2", Float32, ("n2",), attrib = OrderedDict("units" => "µmol N / L"))
v[:] = Available_nutrients

v = defVar(ds, "init", Float32, ("init",), attrib = OrderedDict("units" => "10e6 cells / L"))
v[:] = Starting_population

v = defVar(ds, "t", Int, ("t",), attrib = OrderedDict("units" => "s"))
v[:] = collect(range(0, T, length=NT))

v = defVar(ds, "d1", Float64,("kappa","ratio", "depth", "init", "n2", "t"), attrib = OrderedDict(
"units" =>  "biomass", "long_name" => "surface diatoms"))
v[:,:,:,:,:,:] = matrix_out_d1 ; 

v = defVar(ds, "d2", Float64,("kappa","ratio", "depth", "init", "n2",  "t"), attrib = OrderedDict(
"units" =>  "biomass", "long_name" => "bottom diatoms"))
v[:,:,:,:,:,:] = matrix_out_d2 ; 

v = defVar(ds, "m1", Float64,("kappa","ratio", "depth", "init", "n2",  "t"), attrib = OrderedDict(
"units" =>  "biomass", "long_name" => "biomass_of_microcystis"))
v[:,:,:,:,:,:] = matrix_out_m1 ; 

v = defVar(ds, "m2", Float64,("kappa","ratio", "depth", "init", "n2",  "t"), attrib = OrderedDict(
"units" =>  "biomass", "long_name" => "biomass_of_microcystis"))
v[:,:,:,:,:,:] = matrix_out_m2 ; 

v = defVar(ds, "n1", Float64,("kappa","ratio", "depth", "init", "n2",  "t"), attrib = OrderedDict(
"units" =>  "nutrient", "long_name" => "nutrient"))
v[:,:,:,:,:,:] = matrix_out_n1 ; 


println("Saved $fout \n")
close(ds)
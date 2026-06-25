using JuMP, MosekTools, Random, CSV, DataFrames, Dates, MathOptInterface, LinearAlgebra

include("../src/instance.jl")
include("../src/utils.jl")
include("../src/solvers.jl")

const MOI = MathOptInterface

"""
MÉTHODE SDP : Relaxation SDP avec Mosek.
"""

using JuMP, MosekTools, Random, CSV, DataFrames, Dates, MathOptInterface, LinearAlgebra

include("../src/instance.jl")
include("../src/utils.jl")
include("../src/solvers.jl")

const MOI = MathOptInterface

function solve_p_median_sdp_mosek(n_clients, n_sites, p, d, f, Q)
    
    model = Model(Mosek.Optimizer)
    set_silent(model)

    @variable(model, y[1:n_sites])
    @variable(model, 0 <= x[1:n_clients, 1:n_sites] <= 1)
    @variable(model, Y[1:n_sites+1, 1:n_sites+1], Symmetric)

    # SDP : Y ⪰ 0
    @constraint(model, Y in PSDCone())

    # Coin bas-droit = 1
    @constraint(model, Y[n_sites+1, n_sites+1] == 1)

    # Dernière colonne = y 
    @constraint(model, [j in 1:n_sites], Y[j, n_sites+1] == y[j])

    # Diagonale : relaxation de y_j² = y_j
    @constraint(model, [j in 1:n_sites], Y[j, j] == y[j])

    # p-median constraints
    @constraint(model, [i in 1:n_clients], sum(x[i, j] for j in 1:n_sites) == 1)
    @constraint(model, sum(y[j] for j in 1:n_sites) == p)
    @constraint(model, [i in 1:n_clients, j in 1:n_sites], x[i, j] <= y[j])

    # Objectif
    @objective(model, Min,
        sum(f[j] * y[j] for j in 1:n_sites) +
        sum(d[i, j] * x[i, j] for i in 1:n_clients, j in 1:n_sites) +
        sum(Q[j, jp] * Y[j, jp] for j in 1:n_sites, jp in (j+1):n_sites)
    )

    t_start = time()
    optimize!(model)
    t_solve = time() - t_start

    val_relaxation = -1.0
    if primal_status(model) == MOI.FEASIBLE_POINT
        val_relaxation = JuMP.objective_value(model)
    end
    
    return val_relaxation, t_solve
end


folder_path = joinpath(@__DIR__, "..", "Instances", "Euclid")
file_name = "1011EuclS.txt"
instance_path = joinpath(folder_path, file_name)
println("Loading instance: $instance_path")
inst = readInstance(instance_path)

n_sites   = 60
n_clients = 60
p         = 15
f         = Float64.(inst.c)
d         = Float64.(inst.d')  
Q         = Float64.(inst.a)
n = 60

println("  n_sites=$n_sites, n_clients=$n_clients, p=$p")
println("================================================")

val_relaxation, t_solve = solve_p_median_sdp_mosek(n_clients, n_sites, p, d[1:n, 1:n], f, Q[1:n, 1:n])
println("  relaxation  = $val_relaxation")
println("  t_solve   = $t_solve")

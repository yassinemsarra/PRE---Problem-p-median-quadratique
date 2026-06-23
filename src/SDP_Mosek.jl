using JuMP, MosekTools, Random, CSV, DataFrames, Dates, MathOptInterface, LinearAlgebra

include("../src/instance.jl")
include("../src/utils.jl")
include("../src/solvers.jl")

const MOI = MathOptInterface

"""
MÉTHODE SDP : Relaxation SDP avec Mosek.

Reformulation :
    - On pose Y = y * y^T  (matrice n_sites × n_sites)
    - Le terme quadratique Σ_{j<j'} q_jj' * y_j * y_j' = ½ <Q, Y>
    - On relaxe Y = y*y^T par la contrainte SDP (et en utilisant Schur):
        M = [Y  y ] ≽ 0
            [y^T 1]
    - Contrainte diagonale : Y_jj = y_j  ∀j  (cohérence, vient de y_j² = y_j)
    - Relaxation des binaires : y_j ∈ [0,1]

"""
function solve_p_median_sdp_mosek(n_clients, n_sites, p, d, f, Q)
    model = Model(Mosek.Optimizer)
    set_silent(model)

    # y_j ∈ [0,1]  
    @variable(model, y[1:n_sites])

    # x_ij ∈ [0,1]  
    @variable(model, 0 <= x[1:n_clients, 1:n_sites] <= 1)

    # Y : matrice symétrique n_sites × n_sites
    @variable(model, Y[1:n_sites, 1:n_sites], Symmetric)

    # ----------------------------------------------------------------
    # Contrainte SDP : M = [Y  y; y^T  1] >= 0
    # Taille : (n_sites + 1) × (n_sites + 1)
    # ----------------------------------------------------------------
    n = n_sites
    @variable(model, M[1:(n+1), 1:(n+1)], Symmetric)

    # Lier M à Y et y
    # Bloc supérieur gauche : M[1:n, 1:n] = Y
    @constraint(model, [j in 1:n, jp in j:n], M[j, jp] == Y[j, jp])

    # Dernière colonne/ligne : M[j, n+1] = y[j]
    @constraint(model, [j in 1:n], M[j, n+1] == y[j])

    # Coin bas-droit : M[n+1, n+1] = 1
    @constraint(model, M[n+1, n+1] == 1)

    # Contrainte SDP : M >= 0
    @constraint(model, M in PSDCone())

    # ----------------------------------------------------------------
    # Contrainte diagonale : Y_jj = y_j  (car y_j² = y_j sur {0,1})
    # ----------------------------------------------------------------
    @constraint(model, [j in 1:n_sites], Y[j, j] == y[j])

    # ----------------------------------------------------------------
    # Contraintes du problème p-médian
    # ----------------------------------------------------------------

    # Chaque client raccordé à exactement 1 site
    @constraint(model, [i in 1:n_clients], sum(x[i, j] for j in 1:n_sites) == 1)

    # Exactement p sites ouverts
    @constraint(model, sum(y[j] for j in 1:n_sites) == p)

    # On ne peut raccorder i à j que si j est ouvert
    @constraint(model, [i in 1:n_clients, j in 1:n_sites], x[i, j] <= y[j])

    # ----------------------------------------------------------------
    # Objectif : min  Σ f_j*y_j  +  Σ c_ij*x_ij  +  ½ <Q, Y>
    # <Q, Y> = tr(Q * Y) = Σ_{j,j'} Q_jj' * Y_jj'
    # Comme Q et Y sont symétriques à diagonale nulle :
    #   ½ <Q, Y> = ½ Σ_{j,j'} Q_jj' * Y_jj'
    #            = Σ_{j<j'} Q_jj' * Y_jj'   (car diag = 0 et symétrie)
    # ----------------------------------------------------------------
    @objective(model, Min,
        sum(f[j] * y[j] for j in 1:n_sites) +
        sum(d[i, j] * x[i, j] for i in 1:n_clients, j in 1:n_sites) +
        sum(Q[j, jp] * Y[j, jp] for j in 1:n_sites, jp in (j+1):n_sites)
    )

    # ----------------------------------------------------------------
    # Résolution
    # ----------------------------------------------------------------
    optimize!(model)

    val_relaxation = -1.0
    t_solve = solve_time(model)

    if primal_status(model) == MOI.FEASIBLE_POINT
        val_relaxation = JuMP.objective_value(model)
    end
    
    for i in 1:n_clients
        for j in 1:n_clients
            print(round(JuMP.value(Y[i, j]), digits=2), "\t")
        end
        println()
    end
    
    for i in 1:n_clients
            print(round(JuMP.value(y[i]), digits=2), "\t")
        end
        println()
    


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

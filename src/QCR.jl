using JuMP, MosekTools, Random, CSV, DataFrames, Dates, MathOptInterface, LinearAlgebra

include("../src/instance.jl")
include("../src/utils.jl")
include("../src/solvers.jl")

const MOI = MathOptInterface

function solve_p_median_qcr(n_clients, n_sites, p, d, f, Q)
    
    # ----------------------------------------------------------------
    # PHASE 1 : Relaxation SDP pour extraire les duaux u
    # ----------------------------------------------------------------
    model_sdp  = Model(Mosek.Optimizer)
    set_silent(model_sdp)

    @variable(model_sdp , y_sdp[1:n_sites])
    @variable(model_sdp , 0 <= x_sdp[1:n_clients, 1:n_sites] <= 1)
    @variable(model_sdp , Y[1:n_sites+1, 1:n_sites+1], Symmetric)

    # SDP : Y ⪰ 0
    @constraint(model_sdp , Y in PSDCone())

    # Coin bas-droit = 1
    @constraint(model_sdp , Y[n_sites+1, n_sites+1] == 1)

    # Dernière colonne = y 
    @constraint(model_sdp , [j in 1:n_sites], Y[j, n_sites+1] == y_sdp[j])

    # Diagonale : relaxation de y_j² = y_j
    diag_cstr = @constraint(model_sdp, [j in 1:n_sites], Y[j, j] == y_sdp[j])

    # p-median constraints
    @constraint(model_sdp , [i in 1:n_clients], sum(x_sdp[i, j] for j in 1:n_sites) == 1)
    @constraint(model_sdp , sum(y_sdp[j] for j in 1:n_sites) == p)
    @constraint(model_sdp , [i in 1:n_clients, j in 1:n_sites], x_sdp[i, j] <= y_sdp[j])

    # Objectif
    @objective(model_sdp , Min,
        sum(f[j] * y_sdp[j] for j in 1:n_sites) +
        sum(d[i, j] * x_sdp[i, j] for i in 1:n_clients, j in 1:n_sites) +
        sum(Q[j, jp] * Y[j, jp] for j in 1:n_sites, jp in (j+1):n_sites)
    )

    t_start = time()
    optimize!(model_sdp)
    t_sdp = time() - t_start

    val_sdp  = -1.0
    if primal_status(model_sdp ) == MOI.FEASIBLE_POINT
        val_sdp  = JuMP.objective_value(model_sdp)
    end
    # ----------------------------------------------------------------
    # PHASE 2 : Extraire u et construire Q_cvx = Q + diag(u)
    # ----------------------------------------------------------------
    u = -dual.(diag_cstr)  

    Q_cvx = copy(Q)
    for j in 1:n_sites
        Q_cvx[j, j] += 10 * u[j]
    end

    # Vérification que Q_cvx est bien SDP 
    eigvals_min = minimum(eigvals(Q_cvx))
    println("  λ_min(Q + U) = $(round(eigvals_min, digits=6))")
    if eigvals_min < -1e-6
        @warn "Q + U n'est pas SDP ! λ_min = $eigvals_min"
        eigvals_Q = minimum(eigvals(Q))
        println("λ_Q = $eigvals_Q")
        min_u = minimum(u)
        max_u = maximum(u)
        println("min_u = $min_u)")
        println("max_u = $max_u")
    end

    # ----------------------------------------------------------------
    # PHASE 3 : MIQP convexe avec Gurobi 
    # ----------------------------------------------------------------
    model_qcr = Model(Gurobi.Optimizer)
    set_silent(model_qcr)

    @variable(model_qcr, y[1:n_sites], Bin)
    @variable(model_qcr, x[1:n_clients, 1:n_sites], Bin)

    @constraint(model_qcr, [i in 1:n_clients], sum(x[i, j] for j in 1:n_sites) == 1)
    @constraint(model_qcr, sum(y[j] for j in 1:n_sites) == p)
    @constraint(model_qcr, [i in 1:n_clients, j in 1:n_sites], x[i, j] <= y[j])

    # Objectif : f^T y  +  d·x  +  y^T (Q+U) y  -  u^T y
    # Le terme quadratique est maintenant convexe
    @objective(model_qcr, Min,
        sum(f[j] * y[j] for j in 1:n_sites) +
        sum(d[i, j] * x[i, j] for i in 1:n_clients, j in 1:n_sites) +
        sum(Q[j, jp] * y[j] * y[jp] for j in 1:n_sites, jp in (j+1):n_sites) +
        sum(u[j] * (y[j]^2) for j in 1:n_sites) -                                         
        sum(u[j] * y[j] for j in 1:n_sites)                                          
    )

    t_start = time()
    optimize!(model_qcr)
    t_qcr = time() - t_start

    # Tracking des performances
    bound, objective, gap = -1.0, -1.0, -1.0
    nodes = 0

    if primal_status(model_qcr) == MOI.FEASIBLE_POINT
        objective = JuMP.objective_value(model_qcr)
        nodes = Int(round(JuMP.node_count(model_qcr)))

        if termination_status(model_qcr) == MOI.OPTIMAL
            gap = 0.0
            bound = objective
        else
            bound = JuMP.objective_bound(model_qcr)
            gap = 100.0 * abs(objective - bound) / (objective + 1e-4)
        end
    end

    return val_sdp, t_sdp, objective, bound, gap, nodes, t_qcr
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

val_sdp, t_sdp, objective, bound, gap, nodes, t_qcr = solve_p_median_qcr(n_clients, n_sites, p, d[1:n, 1:n], f, Q[1:n, 1:n])
println("  val_sdp  = $val_sdp")
println("  objective  = $objective")
println("  t_solve   = $(t_sdp + t_qcr)")
println("  bound  = $bound")
println("  gap  = $gap")
println("  nodes  = $nodes")


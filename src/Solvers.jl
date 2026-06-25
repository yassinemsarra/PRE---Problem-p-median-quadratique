using JuMP
using Gurobi
using LinearAlgebra

const MOI = JuMP.MOI

include("Utils.jl")

"""
MÉTHODE 1 : Linéarisation manuelle de Fortet.
"""
function solve_p_median_manual_linearization(n_clients, n_sites, p, d, f, Q)
    model = Model(Gurobi.Optimizer)
    set_silent(model)
    set_attribute(model, "TimeLimit", 1000.0)

    # Variables de décision 
    @variable(model, y[1:n_sites], Bin) # y[j] = 1 si le site j est ouvert
    @variable(model, x[1:n_clients, 1:n_sites], Bin) # x[i,j] = 1 si le client i est raccordé au site j
    @variable(model, z[j in 1:n_sites, jp in (j+1):n_sites], Bin) # z[j,jp] = y[j] * y[jp]
    
    # Fonction objectif
    @objective(model, Min, 
        sum(f[j] * y[j] for j in 1:n_sites) + 
        sum(d[i,j] * x[i,j] for i in 1:n_clients, j in 1:n_sites) +
        sum(Q[j,jp] * z[j,jp] for j in 1:n_sites, jp in (j+1):n_sites)
        )
    
    # Contraintes structurelles
    @constraint(model, [i in 1:n_clients], sum(x[i,j] for j in 1:n_sites) == 1)
    @constraint(model, sum(y[j] for j in 1:n_sites) == p)
    @constraint(model, [i in 1:n_clients, j in 1:n_sites], x[i,j] <= y[j])

    # Contraintes de linéarisation de Fortet
    @constraint(model, [j in 1:n_sites, jp in (j+1):n_sites], z[j,jp] <= y[j])
    @constraint(model, [j in 1:n_sites, jp in (j+1):n_sites], z[j,jp] <= y[jp])
    @constraint(model, [j in 1:n_sites, jp in (j+1):n_sites], z[j,jp] >= y[j] + y[jp] - 1)

    # Relaxation continue pour obtenir la borne racine
    relax_v = relax_integrality(model)
    optimize!(model)
    val_relaxation = -1.0
    if primal_status(model) == MOI.FEASIBLE_POINT
        val_relaxation = JuMP.objective_value(model)
    end

    # Restauration des variables binaires et résolution exacte
    relax_v()

    t_start = time()
    optimize!(model)
    t_solve = time() - t_start

    # Initialisation des variables de performance
    bound, objective, gap = -1.0, -1.0, -1.0
    nodes = 0

    if primal_status(model) == MOI.FEASIBLE_POINT
        objective = JuMP.objective_value(model)
        nodes = Int(round(JuMP.node_count(model))) 
        
        if termination_status(model) == MOI.OPTIMAL
            gap = 0.0
            bound = objective
        else
            bound = JuMP.objective_bound(model)
            gap = 100.0 * abs(objective - bound) / (objective + 1e-4)
        end
    end

    return val_relaxation, objective, bound, gap, nodes, t_solve
end

"""
MÉTHODE 1 : Linéarisation avec Gurobi.
"""
function solve_p_median_quadratic_gurobi(n_clients, n_sites, p, d, f, Q, prelinearize_val)
    model = Model(Gurobi.Optimizer)
    set_attribute(model, "PreQLinearize", prelinearize_val)
    set_attribute(model, "TimeLimit", 1000.0)
    set_silent(model)


    @variable(model, y[1:n_sites], Bin)
    @variable(model, x[1:n_clients, 1:n_sites], Bin)

    @objective(model, Min,
        sum(f[j] * y[j] for j in 1:n_sites) +
        sum(d[i,j] * x[i,j] for i in 1:n_clients, j in 1:n_sites) +
        sum(Q[j,jp] * y[j] * y[jp] for j in 1:n_sites, jp in (j+1):n_sites)
    )

    @constraint(model, [i in 1:n_clients], sum(x[i,j] for j in 1:n_sites) == 1)
    @constraint(model, sum(y[j] for j in 1:n_sites) == p)
    @constraint(model, [i in 1:n_clients, j in 1:n_sites], x[i,j] <= y[j])

    # Capturer la borne à la racine via callback
    root_bound = Ref(-1.0)
    root_captured = Ref(false)

    function my_callback(cb_data, cb_where::Cint)
        if !root_captured[] && cb_where == Gurobi.GRB_CB_MIPNODE
            depth = Ref{Cint}()
            Gurobi.GRBcbget(cb_data, cb_where, Gurobi.GRB_CB_MIPNODE_NODCNT, depth)
            if depth[] == 0
                bd = Ref{Cdouble}()
                Gurobi.GRBcbget(cb_data, cb_where, Gurobi.GRB_CB_MIPNODE_OBJBND, bd)
                root_bound[] = bd[]
                root_captured[] = true
            end
        end
    end

    MOI.set(model, Gurobi.CallbackFunction(), my_callback)

    t_start = time()
    optimize!(model)
    t_solve = time() - t_start

    val_relaxation = root_bound[]

    bound, objective, gap = -1.0, -1.0, -1.0
    nodes = 0

    if primal_status(model) == MOI.FEASIBLE_POINT
        objective = JuMP.objective_value(model)
        nodes = Int(round(JuMP.node_count(model)))

        if termination_status(model) == MOI.OPTIMAL
            gap = 0.0
            bound = objective
        else
            bound = JuMP.objective_bound(model)
            gap = 100.0 * abs(objective - bound) / (abs(objective) + 1e-4)
        end
    end

    return val_relaxation, objective, bound, gap, nodes, t_solve
end

"""
MÉTHODE 2 : Convexification avec la plus petite valeur propre.
"""
function solve_p_median_quadratic_convex(n_clients, n_sites, p, d, f, Q)
    model = Model(Gurobi.Optimizer)
    set_silent(model)
    set_attribute(model, "TimeLimit", 1000.0)

    set_attribute(model, "PreQLinearize", 0)

    # Calcul de la plus petite valeur propre de Q
    λ_1 = minimum(eigvals(Q))
    λ = λ_1 < 0 ? -λ_1 : 0.0

    # Variables de décision
    @variable(model, y[1:n_sites], Bin) # y[j] = 1 si le site j est ouvert, 0 sinon 
    @variable(model, x[1:n_clients, 1:n_sites], Bin) # x[i,j] = 1 si le client i est raccordé au site j, 0 sinon

    # Fonction objectif
    @objective(model, Min,
        sum(f[j] * y[j] for j in 1:n_sites) +
        sum(d[i,j] * x[i,j] for i in 1:n_clients, j in 1:n_sites) +
        sum(Q[j,jp] * y[j] * y[jp] for j in 1:n_sites, jp in (j+1):n_sites) +
        0.5 * λ * sum(y[j]^2 - y[j] for j in 1:n_sites)
    )

    # Contraintes structurelles
    @constraint(model, [i in 1:n_clients], sum(x[i,j] for j in 1:n_sites) == 1)
    @constraint(model, sum(y[j] for j in 1:n_sites) == p)
    @constraint(model, [i in 1:n_clients, j in 1:n_sites], x[i,j] <= y[j])

    # Relaxation continue pour obtenir la borne racine
    relax_v = relax_integrality(model) 
    optimize!(model)
    val_relaxation = -1.0
    if primal_status(model) == MOI.FEASIBLE_POINT
        val_relaxation = JuMP.objective_value(model)
    else
        return -1.0, -1.0, -1.0, -1.0, 0, -1.0
    end
    
    # Restauration des variables binaires et résolution exacte
    relax_v()

    t_start = time()
    optimize!(model)
    t_solve = time() - t_start

    # Initialisation des variables de performance
    bound, objective, gap = -1.0, -1.0, -1.0
    nodes = 0

    if primal_status(model) == MOI.FEASIBLE_POINT
        objective = JuMP.objective_value(model)
        nodes = Int(round(JuMP.node_count(model)))

        if termination_status(model) == MOI.OPTIMAL
            gap = 0.0
            bound = objective
        else
            bound = JuMP.objective_bound(model)
            gap = 100.0 * abs(objective - bound) / (objective + 1e-4)
        end
    end

    return val_relaxation, objective, bound, gap, nodes, t_solve
end

"""
MÉTHODE 3 : Projection de la matrice Q sur le cone SDP.
"""
function solve_p_median_quadratic_sdp(n_clients, n_sites, p, d, f, Q)
    model = Model(Gurobi.Optimizer)
    set_silent(model)
    
    set_attribute(model, "PreQLinearize", 0)
    set_attribute(model, "TimeLimit", 1000.0)

    # Décomposition de Q en une partie PSD convexe
    Q_prime = project_to_sdp(Q)

    # Variables de décision 
    @variable(model, y[1:n_sites], Bin) # y[j] = 1 si le site j est ouvert
    @variable(model, x[1:n_clients, 1:n_sites], Bin) # x[i,j] = 1 si le client i est raccordé au site j
    @variable(model, z[j in 1:n_sites, jp in 1:n_sites], Bin) # z[j,jp] = y[j] * y[jp]

    # Fonction objectif
    @objective(model, Min,
        sum(f[j] * y[j] for j in 1:n_sites) +
        sum(d[i,j] * x[i,j] for i in 1:n_clients, j in 1:n_sites) +
        sum(Q_prime[j,jp] * y[j] * y[jp] for j in 1:n_sites, jp in 1:n_sites) / 2 +
        sum((Q[j,jp] - Q_prime[j,jp]) * z[j,jp] for j in 1:n_sites, jp in 1:n_sites) / 2
    )

    # Contraintes structurelles
    @constraint(model, [i in 1:n_clients], sum(x[i,j] for j in 1:n_sites) == 1)
    @constraint(model, sum(y[j] for j in 1:n_sites) == p)
    @constraint(model, [i in 1:n_clients, j in 1:n_sites], x[i,j] <= y[j])

    # Contraintes de linéarisation de Fortet
    @constraint(model, [j in 1:n_sites, jp in 1:n_sites], z[j,jp] <= y[j])
    @constraint(model, [j in 1:n_sites, jp in 1:n_sites], z[j,jp] <= y[jp])
    @constraint(model, [j in 1:n_sites, jp in 1:n_sites], z[j,jp] >= y[j] + y[jp] - 1)

    # Relaxation continue pour obtenir la borne racine
    relax_v = relax_integrality(model)
    optimize!(model)
    val_relaxation = -1.0
    if primal_status(model) == MOI.FEASIBLE_POINT
        val_relaxation = JuMP.objective_value(model)
    end

    # Restauration des variables binaires et résolution exacte
    relax_v()

    t_start = time()
    optimize!(model)
    t_solve = time() - t_start

    # Initialisation des variables de performance
    bound, objective, gap = -1.0, -1.0, -1.0
    nodes = 0

    if primal_status(model) == MOI.FEASIBLE_POINT
        objective = JuMP.objective_value(model)
        nodes = Int(round(JuMP.node_count(model)))

        if termination_status(model) == MOI.OPTIMAL
            gap = 0.0
            bound = objective
        else
            bound = JuMP.objective_bound(model)
            gap = 100.0 * abs(objective - bound) / (objective + 1e-4)
        end
    end

    return val_relaxation, objective, bound, gap, nodes, t_solve
end

"""
MÉTHODE 4 : Convexification de l'objectif.
"""
function solve_p_median_quadratic_convex_obj(n_clients, n_sites, p, d, f, Q)
    model = Model(Gurobi.Optimizer)
    set_silent(model)

    set_attribute(model, "PreQLinearize", 0)
    set_attribute(model, "TimeLimit", 1000.0)

    # Variables de décision 
    @variable(model, y[1:n_sites], Bin) # y[j] = 1 si le site j est ouvert
    @variable(model, x[1:n_clients, 1:n_sites], Bin) # x[i,j] = 1 si le client i est raccordé au site j

    # Fonction objectif
    @objective(model, Min,
        sum(f[j] * y[j] for j in 1:n_sites) +
        sum(d[i,j] * x[i,j] for i in 1:n_clients, j in 1:n_sites) +
        sum(Q[j,jp] * 0.5 * ((y[j] + y[jp])^2 - y[j] - y[jp]) for j in 1:n_sites, jp in (j+1):n_sites)
    )

    # Contraintes structurelles
    @constraint(model, [i in 1:n_clients], sum(x[i,j] for j in 1:n_sites) == 1)
    @constraint(model, sum(y[j] for j in 1:n_sites) == p)
    @constraint(model, [i in 1:n_clients, j in 1:n_sites], x[i,j] <= y[j])

    # Relaxation continue pour obtenir la borne racine
    relax_v = relax_integrality(model)
    optimize!(model)
    val_relaxation = -1.0
    if primal_status(model) == MOI.FEASIBLE_POINT
        val_relaxation = JuMP.objective_value(model)
    end

    # Restauration des variables binaires et résolution exacte
    relax_v()
    
    t_start = time()
    optimize!(model)
    t_solve = time() - t_start

    # Initialisation des variables de performance
    bound, objective, gap = -1.0, -1.0, -1.0
    nodes = 0

    if primal_status(model) == MOI.FEASIBLE_POINT
        objective = JuMP.objective_value(model)
        nodes = Int(round(JuMP.node_count(model)))

        if termination_status(model) == MOI.OPTIMAL
            gap = 0.0
            bound = objective
        else
            bound = JuMP.objective_bound(model)
            gap = 100.0 * abs(objective - bound) / (objective + 1e-4)
        end
    end

    return val_relaxation, objective, bound, gap, nodes, t_solve
end
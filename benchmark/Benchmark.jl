using JuMP, Gurobi, Random, CSV, DataFrames, Dates, LinearAlgebra

"""
MÉTHODE 1 : Linéarisation manuelle de Fortet.
"""
function solve_p_median_manual_linearization(n_clients, n_sites, p, d, f, Q)
    model = Model(Gurobi.Optimizer)
    set_silent(model)
    set_attribute(model, "TimeLimit", 120.0)

    @variable(model, y[1:n_sites], Bin)
    @variable(model, x[1:n_clients, 1:n_sites], Bin)
    @variable(model, z[j in 1:n_sites, jp in (j+1):n_sites], Bin)

    @objective(model, Min, 
    sum(f[j] * y[j] for j in 1:n_sites) + 
    sum(d[i,j] * x[i,j] for i in 1:n_clients, j in 1:n_sites) +
    sum(Q[j,jp] * z[j,jp] for j in 1:n_sites, jp in (j+1):n_sites)
    )
    
    @constraint(model, [i in 1:n_clients], sum(x[i,j] for j in 1:n_sites) == 1)
    @constraint(model, sum(y[j] for j in 1:n_sites) == p)
    @constraint(model, [i in 1:n_clients, j in 1:n_sites], x[i,j] <= y[j])

    @constraint(model, [j in 1:n_sites, jp in (j+1):n_sites], z[j,jp] <= y[j])
    @constraint(model, [j in 1:n_sites, jp in (j+1):n_sites], z[j,jp] <= y[jp])
    @constraint(model, [j in 1:n_sites, jp in (j+1):n_sites], z[j,jp] >= y[j] + y[jp] - 1)

    
    relax_v = relax_integrality(model) 
    optimize!(model)
    
    val_relaxation = -1.0
    if primal_status(model) == MOI.FEASIBLE_POINT
        val_relaxation = JuMP.objective_value(model)
    else
        return -1.0, -1.0, -1.0, -1.0, 0, -1.0
    end
    
    # Annulation de relaxation
    relax_v()

    optimize!(model)

    # Initialisation des variables de performance
    bound, objective, gap = -1.0, -1.0, -1.0
    nodes = 0
    t_solve = solve_time(model)

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
    set_silent(model)
    set_attribute(model, "TimeLimit", 120.0)

    set_attribute(model, "PreQLinearize", prelinearize_val)

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

    relax_v = relax_integrality(model) 
    optimize!(model)
    
    val_relaxation = -1.0
    if primal_status(model) == MOI.FEASIBLE_POINT
        val_relaxation = JuMP.objective_value(model)
    else
        return -1.0, -1.0, -1.0, -1.0, 0, -1.0
    end
    
    # Annulation de relaxation
    relax_v()

    optimize!(model)

    bound, objective, gap = -1.0, -1.0, -1.0
    nodes = 0
    t_solve = solve_time(model)

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
MÉTHODE 2 : Convexification avec la plus petite valeur propre.
"""

function solve_p_median_quadratic_convex(n_clients, n_sites, p, d, f, Q)
    model = Model(Gurobi.Optimizer)
    set_silent(model)
    set_attribute(model, "TimeLimit", 120.0)

    set_attribute(model, "PreQLinearize", 0)

    λ_1 = minimum(eigvals(Q))
    λ = λ_1 < 0 ? -λ_1 : 0.0

    @variable(model, y[1:n_sites], Bin)
    @variable(model, x[1:n_clients, 1:n_sites], Bin)

    @objective(model, Min,
        sum(f[j] * y[j] for j in 1:n_sites) +
        sum(d[i,j] * x[i,j] for i in 1:n_clients, j in 1:n_sites) +
        sum(Q[j,jp] * y[j] * y[jp] for j in 1:n_sites, jp in (j+1):n_sites) +
        λ * sum(y[j]^2 - y[j] for j in 1:n_sites)
    )

    @constraint(model, [i in 1:n_clients], sum(x[i,j] for j in 1:n_sites) == 1)
    @constraint(model, sum(y[j] for j in 1:n_sites) == p)
    @constraint(model, [i in 1:n_clients, j in 1:n_sites], x[i,j] <= y[j])

    relax_v = relax_integrality(model) 
    optimize!(model)
    
    val_relaxation = -1.0
    if primal_status(model) == MOI.FEASIBLE_POINT
        val_relaxation = JuMP.objective_value(model)
    else
        return -1.0, -1.0, -1.0, -1.0, 0, -1.0
    end
    
    # Annulation de relaxation
    relax_v()

    optimize!(model)

    bound, objective, gap = -1.0, -1.0, -1.0
    nodes = 0
    t_solve = solve_time(model)

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



function project_to_sdp(A::AbstractMatrix{<:Real})
    # Vérifier que la matrice est carrée
    size(A, 1) == size(A, 2) || error("La matrice doit être carrée")

    # Symétrisation (utile si A n'est pas exactement symétrique)
    A_sym = (A + A') / 2

    # Décomposition en valeurs propres (spectrale)
    F = eigen(A_sym)

    # Remplacement des valeurs propres négatives par 0
    Λ_proj = Diagonal(map(x -> max(x, 0), F.values))

    # Reconstruction de la matrice projetée
    A_proj = F.vectors * Λ_proj * F.vectors'
    return A_proj
end

"""
MÉTHODE 3 : Projection de la matrice Q sur le cone SDP.
"""

function solve_p_median_quadratic_sdp(n_clients, n_sites, p, d, f, Q)
    model = Model(Gurobi.Optimizer)
    set_silent(model)
    set_attribute(model, "TimeLimit", 120.0)

    set_attribute(model, "PreQLinearize", 0)

    Q_prime = project_to_sdp(Q)

    @variable(model, y[1:n_sites], Bin)
    @variable(model, x[1:n_clients, 1:n_sites], Bin)
    @variable(model, z[j in 1:n_sites, jp in 1:n_sites], upper_bound=1, lower_bound=0)

    @objective(model, Min,
        sum(f[j] * y[j] for j in 1:n_sites) +
        sum(d[i,j] * x[i,j] for i in 1:n_clients, j in 1:n_sites) +
        sum(Q_prime[j,jp] * y[j] * y[jp] for j in 1:n_sites, jp in 1:n_sites) / 2 +
        sum((Q[j,jp] - Q_prime[j,jp]) * z[j,jp] for j in 1:n_sites, jp in 1:n_sites) / 2
    )

    @constraint(model, [i in 1:n_clients], sum(x[i,j] for j in 1:n_sites) == 1)
    @constraint(model, sum(y[j] for j in 1:n_sites) == p)
    @constraint(model, [i in 1:n_clients, j in 1:n_sites], x[i,j] <= y[j])

    # Contraintes de linéarisation 
    @constraint(model, [j in 1:n_sites, jp in 1:n_sites], z[j,jp] <= y[j])
    @constraint(model, [j in 1:n_sites, jp in 1:n_sites], z[j,jp] <= y[jp])
    @constraint(model, [j in 1:n_sites, jp in 1:n_sites], z[j,jp] >= y[j] + y[jp] - 1)


    relax_v = relax_integrality(model) 
    optimize!(model)
    
    val_relaxation = -1.0
    if primal_status(model) == MOI.FEASIBLE_POINT
        val_relaxation = JuMP.objective_value(model)
    else
        return -1.0, -1.0, -1.0, -1.0, 0, -1.0
    end
    
    # Annulation de relaxation
    relax_v()

    optimize!(model)

    bound, objective, gap = -1.0, -1.0, -1.0
    nodes = 0
    t_solve = solve_time(model)

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
    set_attribute(model, "TimeLimit", 120.0)

    set_attribute(model, "PreQLinearize", 0)

    @variable(model, y[1:n_sites], Bin)
    @variable(model, x[1:n_clients, 1:n_sites], Bin)

    # Remplacer dans l'objectif yj * yjp par 1/2 * ((yj + yjp)^2 - yj - yjp)
    @objective(model, Min,
        sum(f[j] * y[j] for j in 1:n_sites) +
        sum(d[i,j] * x[i,j] for i in 1:n_clients, j in 1:n_sites) +
        sum(Q[j,jp] * 0.5 * ((y[j] + y[jp])^2 - y[j] - y[jp]) for j in 1:n_sites, jp in (j+1):n_sites)
    )

    @constraint(model, [i in 1:n_clients], sum(x[i,j] for j in 1:n_sites) == 1)
    @constraint(model, sum(y[j] for j in 1:n_sites) == p)
    @constraint(model, [i in 1:n_clients, j in 1:n_sites], x[i,j] <= y[j])

    relax_v = relax_integrality(model) 
    optimize!(model)
    
    val_relaxation = -1.0
    if primal_status(model) == MOI.FEASIBLE_POINT
        val_relaxation = JuMP.objective_value(model)
    else
        return -1.0, -1.0, -1.0, -1.0, 0, -1.0
    end

    # Annulation de relaxation
    relax_v()

    optimize!(model)

    bound, objective, gap = -1.0, -1.0, -1.0
    nodes = 0
    t_solve = solve_time(model)

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


# ==============================================================================

function generate_instance(n_clients, n_sites, p; seed=nothing)
    if seed !== nothing Random.seed!(seed) end
    d = rand(1:100, n_clients, n_sites)
    f = rand(1:100, n_sites)
 
    Q = rand(n_sites, n_sites) .* 20
    Q = (Q + Q') / 2
    for j in 1:n_sites Q[j,j] = 0.0 end

    return n_clients, n_sites, p, d, f, Q
end

# ==============================================================================

function run_benchmark(instances)
    results = DataFrame(
        n_clients = Int[], n_sites = Int[], p = Int[], method = String[],
        val_relaxation = Float64[], objective = Float64[], bound = Float64[],
        gap = Float64[], nodes = Int[], t_solve = Float64[]
    )

    # Noms mis à jour pour expliciter les configurations de PreQLinearize
    methods = [
        ("Fortet",                 (nc, ns, p, d, f, Q) -> solve_p_median_manual_linearization(nc, ns, p, d, f, Q)),
        ("Gurobi_PreQLinearize_2", (nc, ns, p, d, f, Q) -> solve_p_median_quadratic_gurobi(nc, ns, p, d, f, Q, 2)),
        ("Gurobi_PreQLinearize_1", (nc, ns, p, d, f, Q) -> solve_p_median_quadratic_gurobi(nc, ns, p, d, f, Q, 1)), 
        ("Gurobi_PreQLinearize_0", (nc, ns, p, d, f, Q) -> solve_p_median_quadratic_gurobi(nc, ns, p, d, f, Q, 0)),
        ("Eigenvalue",             (nc, ns, p, d, f, Q) -> solve_p_median_quadratic_convex(nc, ns, p, d, f, Q)),
        ("SDP",                    (nc, ns, p, d, f, Q) -> solve_p_median_quadratic_sdp(nc, ns, p, d, f, Q)),
        ("Convex_obj",             (nc, ns, p, d, f, Q) -> solve_p_median_quadratic_convex_obj(nc, ns, p, d, f, Q)),
    ]

    n_total = length(instances) * length(methods)
    n_done = 0

    println("Début du benchmark : $(length(instances)) instances à tester ($(n_total) résolutions).")

    for (idx, (nc, ns, p, d, f, Q)) in enumerate(instances)
        println("\nInstance $idx/$(length(instances)) | Clients: $nc, Sites: $ns, p: $p")
        ref_objective = nothing

        for (name, solve_fn) in methods
            n_done += 1
            print("  -> Execution: $name... ")
            flush(stdout)

            try
                val_relax, obj, bnd, gap, nodes, t = solve_fn(nc, ns, p, d, f, Q)

                if obj > 0
                    if ref_objective === nothing
                        ref_objective = obj
                    elseif abs(obj - ref_objective) > 1e-1
                        print("[WARNING: divergence obj] ")
                    end
                end

                println("OK")
                println("     Obj: $(round(obj, digits=1)) | Relax: $(round(val_relax, digits=1)) | Gap: $(round(gap, digits=2))% | Noeuds: $nodes | Temps: $(round(t, digits=2))s")

                push!(results, (nc, ns, p, name, val_relax, obj, bnd, gap, nodes, t))
            catch e
                println("ERROR")
                println("     Détails : $e")
                push!(results, (nc, ns, p, name, -1.0, -1.0, -1.0, -1.0, 0, -1.0))
            end
        end
    end

    println("\nBenchmark terminé.")
    return results
end

# --- MAIN CONTROLLER ---
function main()
    configs = [
        (20, 10),
        (50, 20),
        (100, 30),
        (200, 50),
    ]

    instances = [
        generate_instance(nc, ns, div(ns, 3), seed=i)
        for (nc, ns) in configs
        for i in 1:4
    ]

    results = run_benchmark(instances)

    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    filename = "benchmark_$timestamp.csv"
    CSV.write(filename, results)
    println("Résultats exportés dans : $filename")
end

main()
using JuMP, Gurobi, Random, CSV, DataFrames, Dates, MathOptInterface

"""
MÉTHODE 1 : Linéarisation avec Gurobi.
"""

function solve_p_median_quadratic_gurobi(n_clients, n_sites, p, d, f, Q, prelinearize_val)
    model = Model(Gurobi.Optimizer)
    set_silent(model)

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

function main()
    # Instances (n_clients, n_sites, p)
    instances = [
        (10, 15, 3),
        (20, 30, 5),
        (30, 40, 6),
        (40, 50, 8)
    ]

    results = DataFrame(
        methode          = String[],
        n_clients        = Int[],
        n_sites          = Int[],
        p                = Int[],
        relax_racine     = Float64[],
        valeur_optimale  = Float64[],
        borne_inferieure = Float64[],
        gap_pourcent     = Float64[],
        nombre_noeuds    = Int[],
        temps_s          = Float64[]
    )
    for (n_clients, n_sites, p) in instances

        # Boucle pour tester chaque valeur de PreLinearize
        for p_val in [0, 1, 2]

            Random.seed!(42)
            clients = rand(n_clients, 2) .* 100
            sites = rand(n_sites, 2) .* 100
            d = [sqrt(sum((clients[i, :] .- sites[j, :]).^2)) for i in 1:n_clients, j in 1:n_sites]
            f = rand(n_sites) .* 50

            Q = rand(n_sites, n_sites) .* 20
            Q = (Q + Q') / 2
            for j in 1:n_sites
                Q[j, j] = 0.0
            end

            
            # Résolution
            relax, obj, bnd, gp, nds, t = solve_p_median_quadratic_gurobi(n_clients, n_sites, p, d, f, Q, p_val)

            # Enregistrement
            push!(results, ("Gurobi (prelinearization = $p_val)", n_clients, n_sites, p, relax, obj, bnd, gp, nds, t))
        end
    end

    # Exportation CSV unique pour ce fichier
    filename = "PQNE_M1_Auto" * Dates.format(now(), "yyyymmdd_HHMM") * ".csv"
    CSV.write(filename, results)
    println(results)
end

# Lancement du code
main()

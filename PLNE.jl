using JuMP, Gurobi, Random, CSV, DataFrames, Dates

"""
Résout le problème du p-médian classique (Linéaire Entier)
"""

function solve_p_median_linear(n_clients, n_sites, p, d, f)
    # 1. Initialisation du modèle
    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "OutputFlag", 0)
    set_silent(model) 

    # Déclaration des variables 
    @variable(model, y[1:n_sites], Bin)
    @variable(model, x[1:n_clients, 1:n_sites], Bin)

    # Objectif
    @objective(model, Min, 
        sum(f[j] * y[j] for j in 1:n_sites) + 
        sum(d[i,j] * x[i,j] for i in 1:n_clients, j in 1:n_sites))

    # Contraintes 
    @constraint(model, [i in 1:n_clients], sum(x[i,j] for j in 1:n_sites) == 1)
    @constraint(model, sum(y[j] for j in 1:n_sites) == p)
    @constraint(model, [i in 1:n_clients, j in 1:n_sites], x[i,j] <= y[j])

    # 2. Résolution 
    optimize!(model)

    # 3. Récupération des résultats
    bound = -1.0
    objective = -1.0
    gap = -1.0
    sites_ouverts = Int[]

    if primal_status(model) == MOI.FEASIBLE_POINT
        objective = JuMP.objective_value(model)
        
        # Verification que les sites sont ouverts
        sites_ouverts = findall(JuMP.value.(y) .>= 0.99)

        if termination_status(model) == MOI.OPTIMAL
            gap = 0.0
            bound = objective
        else
            bound = JuMP.objective_bound(model)
            gap = 100.0 * abs(objective - bound) / (objective + 10^-4)
        end
    end

    time_sol = solve_time(model)

    return objective, bound, gap, time_sol
end

# ==============================================================================
# BOUCLE SUR LES INSTANCES
# ==============================================================================

# Instances (n_clients, n_sites, p)
instances = [
    (10, 15, 3),
    (20, 30, 5),
    (30, 40, 6),
    (40, 50, 8)
]

results = DataFrame(
    n_clients  = Int[],
    n_sites    = Int[],
    p          = Int[],
    optimal    = Float64[],
    bound      = Float64[],
    gap        = Float64[],
    time_s     = Float64[]
)

for (n_clients, n_sites, p) in instances

    Random.seed!(42)
    clients = rand(n_clients, 2) .* 100
    sites   = rand(n_sites, 2)   .* 100
    d = [sqrt(sum((clients[i,:] - sites[j,:]).^2)) for i in 1:n_clients, j in 1:n_sites]
    f = rand(n_sites) .* 50

    # Résolution
    objective, bound, gap, time_sol = solve_p_median_linear(n_clients, n_sites, p, d, f)

    # Enregistrement
    push!(results, (n_clients, n_sites, p, objective, bound, gap, time_sol))
end


filename = "PLNE_" * Dates.format(now(), "yyyymmdd") * ".csv"
CSV.write(filename, results)
println("\n", results)

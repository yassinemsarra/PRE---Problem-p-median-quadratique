using JuMP, Gurobi, Random, CSV, DataFrames, Dates, MathOptInterface, LinearAlgebra

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

    set_attribute(model, "PreQLinearize", 0)

    Q_prime = project_to_sdp(Q)

    @variable(model, y[1:n_sites], Bin)
    @variable(model, x[1:n_clients, 1:n_sites], Bin)
    @variable(model, z[j in 1:n_sites, jp in 1:n_sites], Bin)

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
        relax, obj, bnd, gp, nds, t = solve_p_median_quadratic_sdp(n_clients, n_sites, p, d, f, Q)

        # Enregistrement
        push!(results, ("projection SDP", n_clients, n_sites, p, relax, obj, bnd, gp, nds, t))
    end

    # Exportation CSV unique pour ce fichier
    mkpath(joinpath(@__DIR__, "..", "results", "individual"))
    filename = joinpath(@__DIR__, "..", "results", "individual", "PQNE_M3_" * Dates.format(now(), "yyyymmdd") * ".csv")
    CSV.write(filename, results)
    println(results)
end

# Lancement du code
main()
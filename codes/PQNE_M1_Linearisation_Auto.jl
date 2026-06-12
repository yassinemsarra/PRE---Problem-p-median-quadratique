using JuMP, Gurobi, Random, CSV, DataFrames, Dates, MathOptInterface

"""
MÉTHODE 1 : Linéarisation avec Gurobi.
"""
function get_root_relaxation_from_log(model)
    log_lines = String[]
    
    function log_callback(cb_data, cb_where::Int32)
        if cb_where == Gurobi.GRB_CB_MESSAGE
            msg_ptr = Ref{Ptr{Cchar}}()
            Gurobi.GRBcbget(cb_data, cb_where, Gurobi.GRB_CB_MSG_STRING, msg_ptr)
            line = unsafe_string(msg_ptr[])
            push!(log_lines, line)
            # ← pas de print ici = log silencieux pendant le solve
        end
    end
    
    MOI.set(model, Gurobi.CallbackFunction(), log_callback)
    optimize!(model)
    
    val_relaxation = -1.0
    for line in log_lines
        m = match(r"Root relaxation:\s+objective\s+([\d.e+\-]+)", line)
        if m !== nothing
            val_relaxation = parse(Float64, m.captures[1])
            break
        end
    end
    
    return val_relaxation
end

function solve_p_median_quadratic_gurobi(n_clients, n_sites, p, d, f, Q, prelinearize_val)

    model = Model(Gurobi.Optimizer)
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

    val_relaxation = get_root_relaxation_from_log(model)  

    relax_v()
    set_silent(model)
    optimize!(model)


    

    bound, objective, gap = -1.0, -1.0, -1.0
    nodes   = 0
    t_solve = solve_time(model)

    if primal_status(model) == MOI.FEASIBLE_POINT
        objective = JuMP.objective_value(model)
        nodes     = Int(round(JuMP.node_count(model)))

        if termination_status(model) == MOI.OPTIMAL
            gap   = 0.0
            bound = objective
        else
            bound = JuMP.objective_bound(model)
            gap   = 100.0 * abs(objective - bound) / (abs(objective) + 1e-4)
        end
    end

    return val_relaxation, objective, bound, gap, nodes, t_solve
end

function main()
    # Instances (n_clients, n_sites, p)
    instances = [
        (20, 20, 3),
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
    mkpath(joinpath(@__DIR__, "..", "results", "individual"))
    filename = joinpath(@__DIR__, "..", "results", "individual", "PQNE_M1_auto_" * Dates.format(now(), "yyyymmdd") * ".csv")
    CSV.write(filename, results)
    println(results)
end

# Lancement du code
main()

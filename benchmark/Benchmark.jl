using JuMP, Gurobi, Random, CSV, DataFrames, Dates, LinearAlgebra
"""
MÉTHODE 1 : Linéarisation manuelle de Fortet.
"""
function solve_p_median_manual_linearization(n_clients, n_sites, p, d, f, Q)
    model = Model(Gurobi.Optimizer)
    set_silent(model)
    set_attribute(model, "TimeLimit", 7200.0)

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

""""
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

"""
MÉTHODE 2 : Convexification avec la plus petite valeur propre.
"""

function solve_p_median_quadratic_convex(n_clients, n_sites, p, d, f, Q)
    model = Model(Gurobi.Optimizer)
    set_silent(model)
    set_attribute(model, "TimeLimit", 7200.0)

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
    set_attribute(model, "TimeLimit", 7200.0)

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
    set_attribute(model, "TimeLimit", 7200.0)

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

function load_instance_from_csv(folder_path, instance_name)
    try
        params_df = CSV.read(joinpath(folder_path, "$(instance_name)_params.csv"), DataFrame)
        n_clients = params_df[1, "n_clients"]
        n_sites   = params_df[1, "n_sites"]
        p         = params_df[1, "p"]
 
        d = Matrix(CSV.read(joinpath(folder_path, "$(instance_name)_d.csv"), DataFrame, header=false))
        f = vec(Matrix(CSV.read(joinpath(folder_path, "$(instance_name)_f.csv"), DataFrame, header=false)))
        Q = Matrix(CSV.read(joinpath(folder_path, "$(instance_name)_Q.csv"), DataFrame, header=false))
 
        return (n_clients, n_sites, p, d, f, Q)
    catch e
        println("!!! Error loading $instance_name: $e")
        return nothing
    end
end
 
 
function run_benchmark(instances, instance_names)
 
    methods = [
        ("manual_linearization",     (nc,ns,p,d,f,Q) -> solve_p_median_manual_linearization(nc,ns,p,d,f,Q)),
        ("gurobi_preqlin_0",         (nc,ns,p,d,f,Q) -> solve_p_median_quadratic_gurobi(nc,ns,p,d,f,Q, 0)),
        ("gurobi_preqlin_1",         (nc,ns,p,d,f,Q) -> solve_p_median_quadratic_gurobi(nc,ns,p,d,f,Q, 1)),
        ("gurobi_preqlin_2",         (nc,ns,p,d,f,Q) -> solve_p_median_quadratic_gurobi(nc,ns,p,d,f,Q, 2)),
        ("convex_eigenvalue",        (nc,ns,p,d,f,Q) -> solve_p_median_quadratic_convex(nc,ns,p,d,f,Q)),
        ("sdp_projection",           (nc,ns,p,d,f,Q) -> solve_p_median_quadratic_sdp(nc,ns,p,d,f,Q)),
        ("convex_obj_reformulation", (nc,ns,p,d,f,Q) -> solve_p_median_quadratic_convex_obj(nc,ns,p,d,f,Q)),
    ]
 
    n_total = length(instances) * length(methods)
    counter = 0
    rows    = []
 
    println("Running: $(length(instances)) instances × $(length(methods)) methods = $n_total runs\n")
 
    for (idx, (nc, ns, p, d, f, Q)) in enumerate(instances)
        name = instance_names[idx]
 
        for (method_name, solver) in methods
            counter += 1
            print("[$counter/$n_total] $name ($nc×$ns) | $method_name ... ")
            flush(stdout)
 
            local val_relax, obj, bound, gap, nodes, t_solve
            try
                val_relax, obj, bound, gap, nodes, t_solve = solver(nc, ns, p, d, f, Q)
            catch e
                println("ERROR: $e")
                val_relax, obj, bound, gap, nodes, t_solve = -1.0, -1.0, -1.0, -1.0, 0, -1.0
            end
 
            status = obj > 0 ? (gap == 0.0 ? "OPTIMAL" : "FEASIBLE") : "FAILED"
            println("$status  obj=$(round(obj,digits=1))  gap=$(round(max(gap,0.0),digits=2))%  t=$(round(t_solve,digits=2))s")
 
            push!(rows, (
                instance       = name,
                n_clients      = nc,
                n_sites        = ns,
                p              = p,
                method         = method_name,
                status         = status,
                val_relaxation = val_relax,
                objective      = obj,
                bound          = bound,
                gap_pct        = gap,
                nodes          = nodes,
                t_solve_s      = t_solve,
            ))
        end
    end
 
    return DataFrame(rows)
end
 
 
function main()
    instances_folder = "./instances"
 
    # Same list as in generate_instances.jl
    # Comment out the sizes you don't want to run
    instance_names = [
        # Small (5 reps)
        "small_1", "small_2", "small_3", "small_4", "small_5",
        # Medium (5 reps)
        "medium_1", "medium_2", "medium_3", "medium_4", "medium_5",
        # Large (4 reps)
        "large_1", "large_2", "large_3", "large_4",
        # XLarge (3 reps)
        "xlarge_1", "xlarge_2", "xlarge_3",
        # XXLarge (2 reps)
        "xxlarge_1", "xxlarge_2",
        # Huge (2 reps) — comment out if short on time
        "huge_1", "huge_2",
    ]
 
    println("Loading instances...")
    instances = []
    loaded_names = []
    for name in instance_names
        inst = load_instance_from_csv(instances_folder, name)
        if inst !== nothing
            push!(instances, inst)
            push!(loaded_names, name)
            nc, ns, p, _, _, _ = inst
            println("  ✓ $name  ($nc clients × $ns sites, p=$p)")
        end
    end
 
    if isempty(instances)
        println("!!! No instances loaded. Did you run generate_instances.jl first?")
        return
    end
 
    println("\n" * "="^70)
    results = run_benchmark(instances, loaded_names)
 
    results_folder = "../results/benchmark"
    if !isdir(results_folder)
        mkpath(results_folder)
    end
 
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    filename  = joinpath(results_folder, "benchmark_$(timestamp).csv")
    CSV.write(filename, results)
    println("\nResults exported to: $filename")
end
 
main()
 
 

# main_benchmark.jl
# Runs all methods on all instances and saves results to results/
# Run from the benchmark/ folder:
#   julia main_benchmark.jl

using JuMP, Gurobi, LinearAlgebra, CSV, DataFrames, Dates

include(joinpath(@__DIR__, "..", "src", "instance.jl"))
include(joinpath(@__DIR__, "..", "src", "Utils.jl"))
include(joinpath(@__DIR__, "..", "src", "Solvers.jl"))


function load_instance(path::String)
    inst      = readInstance(path)
    n_sites   = inst.n
    n_clients = inst.m
    p         = max(1, round(Int, 0.1 * n_sites))
    f         = Float64.(inst.c)
    d         = Float64.(inst.d')  
    Q         = Float64.(inst.a)   
    return n_clients, n_sites, p, d, f, Q
end


function run_benchmark(instance_folder::String)

    # Read all .txt files from the instances folder
    files = sort(filter(f -> endswith(f, ".txt"), readdir(instance_folder, join=true)))

    if isempty(files)
        println("!!! No .txt files found in $instance_folder")
        return
    end

    methods = [
        ("manual_linearization",     (nc,ns,p,d,f,Q) -> solve_p_median_manual_linearization(nc,ns,p,d,f,Q)),
        ("gurobi_preqlin_0",         (nc,ns,p,d,f,Q) -> solve_p_median_quadratic_gurobi(nc,ns,p,d,f,Q, 0)),
        ("gurobi_preqlin_1",         (nc,ns,p,d,f,Q) -> solve_p_median_quadratic_gurobi(nc,ns,p,d,f,Q, 1)),
        ("gurobi_preqlin_2",         (nc,ns,p,d,f,Q) -> solve_p_median_quadratic_gurobi(nc,ns,p,d,f,Q, 2)),
        ("convex_eigenvalue",        (nc,ns,p,d,f,Q) -> solve_p_median_quadratic_convex(nc,ns,p,d,f,Q)),
        ("sdp_projection",           (nc,ns,p,d,f,Q) -> solve_p_median_quadratic_sdp(nc,ns,p,d,f,Q)),
        ("convex_obj_reformulation", (nc,ns,p,d,f,Q) -> solve_p_median_quadratic_convex_obj(nc,ns,p,d,f,Q)),
    ]

    n_total = length(files) * length(methods)
    counter = 0
    rows    = []

    println("="^70)
    println("BENCHMARK: $(length(files)) instances × $(length(methods)) methods = $n_total runs")
    println("="^70 * "\n")

    for fpath in files
        fname = basename(fpath)
        println("--- $fname ---")

        local n_clients, n_sites, p, d, f, Q
        try
            n_clients, n_sites, p, d, f, Q = load_instance(fpath)
            println("  n_sites=$n_sites, n_clients=$n_clients, p=$p")
        catch e
            println("!!! Error loading $fname: $e")
            continue
        end

        for (method_name, solver) in methods
            counter += 1
            print("  [$counter/$n_total] $method_name ... ")
            flush(stdout)

            local val_relax, obj, bound, gap, nodes, t_solve
            try
                val_relax, obj, bound, gap, nodes, t_solve = solver(n_clients, n_sites, p, d, f, Q)
            catch e
                println("ERROR: $e")
                val_relax, obj, bound, gap, nodes, t_solve = -1.0, -1.0, -1.0, -1.0, 0, -1.0
            end

            status = obj > 0 ? (gap == 0.0 ? "OPTIMAL" : "FEASIBLE") : "FAILED"
            println("$status  obj=$(round(obj,digits=1))  gap=$(round(max(gap,0.0),digits=2))%  t=$(round(t_solve,digits=2))s")

            push!(rows, (
                instance       = fname,
                n_sites        = n_sites,
                n_clients      = n_clients,
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

        println()

        # Save after each instance in case the server crashes
        results_folder = "../results"
        if !isdir(results_folder)
            mkpath(results_folder)
        end
        timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
        CSV.write(joinpath(results_folder, "benchmark_$(timestamp).csv"), DataFrame(rows))
    end

    # Final save
    results_folder = joinpath(@__DIR__, "..", "results")
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    filename  = joinpath(results_folder, "benchmark_$(timestamp).csv")
    CSV.write(filename, DataFrame(rows))
    println("="^70)
    println("Results saved to: $filename")
    println("="^70)
end

instance_path = joinpath(@__DIR__, "..", "Instances", "Euclid")
run_benchmark(instance_path)
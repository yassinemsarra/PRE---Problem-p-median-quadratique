# main_benchmark.jl
# One instance, varying n and p — like the paper.
# Run from the benchmark/ folder:
#   julia main_benchmark.jl

using JuMP, Gurobi, LinearAlgebra, CSV, DataFrames, Dates

include(joinpath(@__DIR__, "..", "src", "instance.jl"))
include(joinpath(@__DIR__, "..", "src", "Utils.jl"))
include(joinpath(@__DIR__, "..", "src", "Solvers.jl"))


function load_instance(path::String, n::Int, p::Int)
    inst = readInstance(path)
    f    = fill(3000.0, n)
    d    = Float64.(inst.d')[1:n, 1:n]
    Q    = Float64.(inst.d)[1:n, 1:n]
    return n, n, p, d, f, Q
end


function run_benchmark(instance_path::String)

    configs = [
        (60,  [15, 20, 25, 30]),
        (70,  [15, 20, 25, 30]),
        (80,  [15, 20, 25, 30]),
        (90,  [15, 20, 25, 30]),
        (100, [15, 20, 25, 30]),
    ]

    methods = [
        ("manual_linearization",     (nc,ns,p,d,f,Q) -> solve_p_median_manual_linearization(nc,ns,p,d,f,Q)),
        ("gurobi_preqlin_0",         (nc,ns,p,d,f,Q) -> solve_p_median_quadratic_gurobi(nc,ns,p,d,f,Q, 0)),
        ("gurobi_preqlin_1",         (nc,ns,p,d,f,Q) -> solve_p_median_quadratic_gurobi(nc,ns,p,d,f,Q, 1)),
        ("gurobi_preqlin_2",         (nc,ns,p,d,f,Q) -> solve_p_median_quadratic_gurobi(nc,ns,p,d,f,Q, 2)),
        ("convex_eigenvalue",        (nc,ns,p,d,f,Q) -> solve_p_median_quadratic_convex(nc,ns,p,d,f,Q)),
        ("sdp_projection",           (nc,ns,p,d,f,Q) -> solve_p_median_quadratic_sdp(nc,ns,p,d,f,Q)),
        ("convex_obj_reformulation", (nc,ns,p,d,f,Q) -> solve_p_median_quadratic_convex_obj(nc,ns,p,d,f,Q)),
    ]

    n_total  = sum(length(ps) for (_, ps) in configs) * length(methods)
    counter  = 0
    rows     = []
    fname    = basename(instance_path)

    results_folder = joinpath(@__DIR__, "..", "results")
    if !isdir(results_folder); mkpath(results_folder); end
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    filename  = joinpath(results_folder, "benchmark_$(splitext(fname)[1])_$(timestamp).csv")

    println("="^70)
    println("BENCHMARK — instance: $fname")
    println("$(sum(length(ps) for (_,ps) in configs)) configs × $(length(methods)) methods = $n_total runs")
    println("="^70 * "\n")

    for (n, p_values) in configs
        for p in p_values
            println("--- n=$n, p=$p ---")

            local n_clients, n_sites, d, f, Q
            try
                n_clients, n_sites, _, d, f, Q = load_instance(instance_path, n, p)
            catch e
                println("!!! Error loading (n=$n, p=$p): $e")
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
                    n              = n,
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

            CSV.write(filename, DataFrame(rows))
            println()
        end
    end

    println("="^70)
    println("Results saved to: $filename")
    println("="^70)
end


instance_path = joinpath(@__DIR__, "..", "instances", "Euclid", "1011EuclS.txt")
run_benchmark(instance_path)
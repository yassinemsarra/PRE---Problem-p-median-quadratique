# main_benchmark_mosek.jl
# One instance, varying n and p — Mosek SDP relaxation only.
# Run from the benchmark/ folder:
#   julia main_benchmark_mosek.jl

using JuMP, MosekTools, LinearAlgebra, CSV, DataFrames, Dates, MathOptInterface

include(joinpath(@__DIR__, "..", "src", "instance.jl"))
include(joinpath(@__DIR__, "..", "src", "SDP_Mosek.jl"))


function load_instance(path::String, n::Int, p::Int)
    inst = readInstance(path)
    f    = fill(3000.0, n)
    d    = Float64.(inst.d')[1:n, 1:n]
    Q    = Float64.(inst.d)[1:n, 1:n]
    return n, n, p, d, f, Q
end


function run_benchmark_mosek(instance_path::String)

    configs = [
        (60,  [15, 20, 25, 30]),
        (70,  [15, 20, 25, 30]),
        (80,  [15, 20, 25, 30]),
        (90,  [15, 20, 25, 30]),
        (100, [15, 20, 25, 30]),
    ]

    n_total  = sum(length(ps) for (_, ps) in configs)
    counter  = 0
    rows     = []
    fname    = basename(instance_path)

    results_folder = joinpath(@__DIR__, "..", "results")
    if !isdir(results_folder); mkpath(results_folder); end
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    filename  = joinpath(results_folder, "benchmark_mosek_$(splitext(fname)[1])_$(timestamp).csv")

    println("="^70)
    println("BENCHMARK MOSEK SDP — instance: $fname")
    println("$n_total configs")
    println("="^70 * "\n")

    for (n, p_values) in configs
        for p in p_values
            counter += 1
            print("[$counter/$n_total] n=$n, p=$p ... ")
            flush(stdout)

            local n_clients, n_sites, d, f, Q
            try
                n_clients, n_sites, _, d, f, Q = load_instance(instance_path, n, p)
            catch e
                println("ERROR loading: $e")
                continue
            end

            local val_relax, t_solve
            try
                val_relax, t_solve = solve_p_median_sdp_mosek(n_clients, n_sites, p, d, f, Q)
            catch e
                println("ERROR solving: $e")
                val_relax, t_solve = -1.0, -1.0
            end

            println("relax=$(round(val_relax, digits=2))  t=$(round(t_solve, digits=2))s")

            push!(rows, (
                instance       = fname,
                n              = n,
                p              = p,
                val_relaxation = val_relax,
                t_solve_s      = t_solve,
            ))

            CSV.write(filename, DataFrame(rows))
        end
    end

    println("\n" * "="^70)
    println("Results saved to: $filename")
    println("="^70)
end


instance_path = joinpath(@__DIR__, "..", "instances", "Euclid", "1011EuclS.txt")
run_benchmark_mosek(instance_path)
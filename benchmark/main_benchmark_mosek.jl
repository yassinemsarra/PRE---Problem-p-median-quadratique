# main_benchmark_mosek.jl
# Benchmark Mosek SDP sur toutes les instances Euclid, n=100 avec p variant.
# Un CSV par instance sauvegardé dans results/.
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


function run_benchmark_mosek(instances_folder::String)

    files = sort(filter(f -> endswith(f, ".txt"), readdir(instances_folder, join=true)))

    if isempty(files)
        println("!!! No .txt files found in $instances_folder")
        return
    end

    # Fixed n=100, varying p
    n = 100
    p_values = [10]

    results_folder = joinpath(@__DIR__, "..", "results")
    if !isdir(results_folder)
        mkpath(results_folder)
    end

    n_total   = length(p_values)
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")

    println("="^70)
    println("BENCHMARK MOSEK SDP: $(length(files)) instances × $(length(p_values)) p values")
    println("Fixed n=$n, p values = $p_values")
    println("="^70 * "\n")

    for fpath in files
        fname    = basename(fpath)
        counter  = 0
        rows     = []
        filename = joinpath(results_folder, "benchmark_mosek_$(splitext(fname)[1])_$(timestamp).csv")

        println("\n" * "="^70)
        println("Instance: $fname")
        println("="^70)

        for p in p_values
            counter += 1
            print("  [$counter/$n_total] n=$n, p=$p ... ")
            flush(stdout)

            local n_clients, n_sites, d, f, Q
            try
                n_clients, n_sites, _, d, f, Q = load_instance(fpath, n, p)
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

            # Sauvegarde après chaque p
            CSV.write(filename, DataFrame(rows))
        end

        println("  → Saved: $filename")
    end

    println("\n" * "="^70)
    println("BENCHMARK MOSEK COMPLETE")
    println("="^70)
end


instances_folder = joinpath(@__DIR__, "..", "instances", "Euclid")
run_benchmark_mosek(instances_folder)
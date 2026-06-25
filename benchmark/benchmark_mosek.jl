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

    all_files = sort(filter(f -> endswith(f, ".txt"), readdir(instances_folder, join=true)))
    files     = all_files[1:min(10, length(all_files))]

    configs = [
        (60,  [15, 20, 25, 30]),
        (70,  [15, 20, 25, 30]),
        (80,  [15, 20, 25, 30]),
        (90,  [15, 20, 25, 30]),
        (100, [15, 20, 25, 30]),
    ]

    n_total = length(files) * sum(length(ps) for (_, ps) in configs)
    counter = 0

    results_folder = joinpath(@__DIR__, "..", "results")
    if !isdir(results_folder); mkpath(results_folder); end
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    filename  = joinpath(results_folder, "benchmark_mosek_$(timestamp).csv")

    println("="^70)
    println("BENCHMARK MOSEK SDP — $(length(files)) instances × $(sum(length(ps) for (_,ps) in configs)) configs = $n_total runs")
    println("Saving to: $filename")
    println("="^70 * "\n")

    first_write = true

    for fpath in files
        fname = basename(fpath)

        for (n, p_values) in configs
            for p in p_values
                counter += 1
                print("[$counter/$n_total] $fname | n=$n p=$p ... ")
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

                status = val_relax > 0 ? "OPTIMAL" : "FAILED"
                println("$status  relax=$(round(val_relax, digits=2))  t=$(round(t_solve, digits=2))s")

                row = DataFrame([(
                    instance       = fname,
                    n              = n,
                    p              = p,
                    method         = "mosek_sdp",
                    status         = status,
                    val_relaxation = val_relax,
                    t_solve_s      = t_solve,
                )])

                if first_write
                    CSV.write(filename, row)
                    first_write = false
                else
                    CSV.write(filename, row, append=true)
                end
            end
        end
    end

    println("\n" * "="^70)
    println("DONE. Results saved to: $filename")
    println("="^70)
end


instances_folder = joinpath(@__DIR__, "..", "Instances", "Euclid")
run_benchmark_mosek(instances_folder)
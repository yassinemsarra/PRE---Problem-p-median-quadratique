using CSV, DataFrames, Random, LinearAlgebra

"""
    generate_instances(output_folder="./instances")

Generates ALL instances and saves them as CSV files.
Run this once before the benchmark:
    julia generate_instances.jl
"""
function generate_instances(output_folder="./instances")

    if !isdir(output_folder)
        mkdir(output_folder)
        println("Created folder: $output_folder")
    end

    # Each line is: (n_clients, n_sites, p, instance_name)
    #
    # Sizing rationale (for a server run):
    #   Small   (20×10)   : 5 reps — very fast, good for sanity checks
    #   Medium  (50×20)   : 5 reps — a few minutes each, still cheap
    #   Large   (100×40)  : 4 reps — starts to take time, 4 is enough
    #   XLarge  (200×60)  : 3 reps — potentially 30+ min each
    #   XXLarge (300×80)  : 2 reps — could be hours, 2 reps is the minimum to average
    #   Huge    (500×100) : 2 reps — only if you have time, comment out if not
    #
    # Total: 23 instances × 7 methods = 161 runs
    configs = [
        # Small (5 reps)
        (20,  10,  3,  "small_1"),
        (20,  10,  3,  "small_2"),
        (20,  10,  3,  "small_3"),
        (20,  10,  3,  "small_4"),
        (20,  10,  3,  "small_5"),
        # Medium (5 reps)
        (50,  20,  6,  "medium_1"),
        (50,  20,  6,  "medium_2"),
        (50,  20,  6,  "medium_3"),
        (50,  20,  6,  "medium_4"),
        (50,  20,  6,  "medium_5"),
        # Large (4 reps)
        (100, 40,  13, "large_1"),
        (100, 40,  13, "large_2"),
        (100, 40,  13, "large_3"),
        (100, 40,  13, "large_4"),
        # XLarge (3 reps)
        (200, 60,  20, "xlarge_1"),
        (200, 60,  20, "xlarge_2"),
        (200, 60,  20, "xlarge_3"),
        # XXLarge (2 reps)
        (300, 80,  26, "xxlarge_1"),
        (300, 80,  26, "xxlarge_2"),
        # Huge (2 reps) — comment these out if you are short on time
        (500, 100, 33, "huge_1"),
        (500, 100, 33, "huge_2"),
    ]

    println("\n" * "="^70)
    println("GENERATING INSTANCES")
    println("="^70 * "\n")

    for (i, (n_clients, n_sites, p, instance_name)) in enumerate(configs)
        println("[$i/$(length(configs))] $instance_name — $n_clients clients × $n_sites sites, p=$p")

        # Different seed per instance so they are all different even within the same size
        Random.seed!(i * 137)

        # Distance matrix: integer costs in [1, 100]
        d = rand(1:100, n_clients, n_sites)

        # Fixed opening cost per site
        f = rand(1:100, n_sites)

        # Quadratic interaction matrix — symmetric, zero diagonal
        # Values in [-10, 10] so some pairs have synergy (negative) and some have conflict (positive)
        Q_raw = rand(n_sites, n_sites) .* 20 .- 10
        Q = (Q_raw + Q_raw') / 2
        for j in 1:n_sites
            Q[j,j] = 0.0
        end

        CSV.write(joinpath(output_folder, "$(instance_name)_params.csv"),
                  DataFrame(n_clients=[n_clients], n_sites=[n_sites], p=[p]))

        CSV.write(joinpath(output_folder, "$(instance_name)_d.csv"),
                  DataFrame(d, :auto), header=false)

        CSV.write(joinpath(output_folder, "$(instance_name)_f.csv"),
                  DataFrame(reshape(f, :, 1), :auto), header=false)

        CSV.write(joinpath(output_folder, "$(instance_name)_Q.csv"),
                  DataFrame(Q, :auto), header=false)
    end

    println("\n" * "="^70)
    println("DONE — $(length(configs)) instances saved in '$output_folder/'")
    println("="^70)
end

generate_instances("./instances")
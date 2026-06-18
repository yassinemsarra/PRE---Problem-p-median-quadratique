using JuMP, Gurobi, LinearAlgebra, CSV, DataFrames, Dates, Random

include("../src/instance.jl")
include("../src/utils.jl")
include("../src/solvers.jl")

function main()
    # ── Option 1 : real instance

    # Load a single instance to test    
    # folder_path = joinpath(@__DIR__, "..", "Instances", "Euclid")
    # file_name = "3011EuclS.txt"
    # instance_path = joinpath(folder_path, file_name)
    # println("Loading instance: $instance_path")
    # inst = readInstance(instance_path)

     # ── Option 2 : random instance
    Random.seed!(42)
    n_clients, n_sites = 20, 20
    f = rand(1:100, n_sites)          
    d = rand(1:100, n_sites, n_clients) 
    Q_raw = rand(-10:10, n_sites, n_sites)
    Q = (Q_raw + Q_raw') .÷ 2 
    for j in 1:n_sites; Q[j,j] = 0.0; end
    inst = Instance(f, d, Q)

    n_sites   = inst.n
    n_clients = inst.m
    p         = max(1, round(Int, 0.1 * n_sites))
    f         = Float64.(inst.c)
    d         = Float64.(inst.d')  
    Q         = Float64.(inst.a)
 
    println("  n_sites=$n_sites, n_clients=$n_clients, p=$p")
    println("")

    # Test one method
    println("Testing manual_linearization...")
    # M1 : solve_p_median_manual_linearization(n_clients, n_sites, p, d, f, Q)
    # M1 : solve_p_median_quadratic_gurobi(n_clients, n_sites, p, d, f, Q, prelinearize_val=0)
    # M2 : solve_p_median_quadratic_convex(n_clients, n_sites, p, d, f, Q)
    # M3 : solve_p_median_quadratic_sdp(n_clients, n_sites, p, d, f, Q)
    # M4 : solve_p_median_quadratic_convex_obj(n_clients, n_sites, p, d, f, Q)
    val_relax, obj, bound, gap, nodes, t_solve = solve_p_median_quadratic_sdp(n_clients, n_sites, p, d, f, Q)

    println("  relaxation  = $val_relax")
    println("  objective   = $obj")
    println("  bound       = $bound")
    println("  gap         = $(gap)%")
    println("  nodes       = $nodes")
    println("  time        = $(t_solve)s")
end

main()
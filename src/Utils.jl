using LinearAlgebra
using JuMP
using Gurobi

const MOI = JuMP.MOI

"""
Projette une matrice carrée sur le cône semi-défini positif (PSD).
"""
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
Intercepte les logs de Gurobi via un callback pour extraire la borne de relaxation continue.
"""
function get_root_relaxation_from_log(model)
    log_lines = String[]
    
    function log_callback(cb_data, cb_where::Int32)
        if cb_where == Gurobi.GRB_CB_MESSAGE
            msg_ptr = Ref{Ptr{Cchar}}()
            Gurobi.GRBcbget(cb_data, cb_where, Gurobi.GRB_CB_MSG_STRING, msg_ptr)
            line = unsafe_string(msg_ptr[])
            push!(log_lines, line)
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

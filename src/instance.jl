mutable struct Instance
    n::Int64 # Number of facilities
    m::Int64 # Number of clients
    c::Vector{Int64} # Opening costs
    d::Matrix{Int64} # d[i, j] = connexion costs between facility i and client j
    a::Matrix{Int64} # a[i, j] = connexion costs between facilities i and j

    function Instance()
        return new()
    end
end

# Constructeur de la structure
function Instance(c::Vector{Int64}, d::Matrix{Int64}, a::Matrix{Int64})

    this = Instance()
    this.c = copy(c)
    this.d = copy(d)
    this.a = copy(a)
    this.n, this.m = size(d)

    return this
end

function Instance(path::String)
    return readInstance(path)
end 


function readInstance(path::String)

    datafile = open(path) 
    data = readlines(datafile)
    close(datafile)
    
    n=-1
    m=-1
    c = nothing
    d = nothing
    
    for line in data
        if n == -1
            if !occursin("FILE", line)
                sLine = parse.(Int64, split(line))
                n = sLine[1]
                m = sLine[2]
                c = Vector{Int64}(undef, m)
                d = Matrix{Int64}(undef, n, m)
            end 
        else
            sLine = parse.(Int64, split(line))
            c[sLine[1]]= sLine[2]
            d[sLine[1], :] = sLine[3:end]
        end 
    end

    return Instance(c, d, d)
end

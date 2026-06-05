# \# Problème P-Médian Quadratique

# 

# Stage de recherche en optimisation combinatoire — implémentation et comparaison de méthodes de résolution pour le problème p-médian quadratique (PQNE).

# 

# \## Structure du projet

PRE---Problem-p-median-quadratique/

├── code/

│   ├── PLNE.jl

│   ├── PQNE\_M1\_Linearisation\_Manuelle.jl

│   ├── PQNE\_M1\_Linearisation\_Auto.jl

│   ├── PQNE\_M2\_Convexification.jl

│   ├── PQNE\_M3\_Projection\_SDP.jl

│   └── PQNE\_M4\_Convexifier\_Obj.jl

├── benchmark/

│   └── Benchmark.jl

├── results/

│   ├── individual/

│   └── benchmark/

└── README.md



\## Méthodes implémentées



1\. \*\*Fortet\*\* — Linéarisation manuelle des termes quadratiques via les contraintes de Fortet

2\. \*\*Gurobi PreQLinearize\*\* — Linéarisation automatique par Gurobi (paramètres 0, 1, 2)

3\. \*\*Eigenvalue\*\* — Convexification par la plus petite valeur propre de Q

4\. \*\*SDP\*\* — Projection de la matrice Q sur le cône des matrices semi-définies positives

5\. \*\*Convex\_obj\*\* — Convexification directe de l'objectif



\## Prérequis



\- Julia 1.x

\- Gurobi (licence académique)

\- Packages Julia : JuMP, Gurobi, CSV, DataFrames, LinearAlgebra, Dates, Random



\## Utilisation



Lancer une méthode individuelle :

```julia

include("code/PQNE\_M2\_Convexification.jl")

```



Lancer le benchmark complet :

```julia

include("benchmark/Benchmark.jl")

```



Les résultats sont sauvegardés automatiquement dans `results/individual/` ou `results/benchmark/`.






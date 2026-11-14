# Projet HPC : décomposition de domaine (méthode de Schwarz)

Projet du cours *High Performance Computing* du M2 MMS (Université de Pau et des Pays de l'Adour), projet n°1 « Domain decomposition ».

## Objectif

On résout le problème de Poisson sur un domaine formé d'un disque et d'un rectangle qui se recouvrent :

    -Δu = f  dans Ω,     u = 0  sur le bord de Ω

Le code fourni résout ce problème de deux façons : sur le domaine entier, et par la méthode de Schwarz alternée (on résout tour à tour sur le disque et sur le rectangle, en échangeant les valeurs sur les bords fictifs).

Le projet consiste à :

1. mesurer et profiler le code fourni ;
2. optimiser le code séquentiel ;
3. écrire la méthode de Schwarz parallèle avec MPI ;
4. étudier les performances, en particulier en fonction de la taille du recouvrement.

## Fichiers

`dr_mesh.py` Génère les maillages avec gmsh : `dr.msh` (domaine entier), `d.msh` (disque), `r.msh` (rectangle) 
`dr_solve.py`  Assemblage éléments finis P1 (`build_system`) et résolution sur le domaine entier 
`dr_solve_dd.py` Méthode de Schwarz alternée sur les deux sous-domaines 
`job_projet.sh` Job Slurm pour lancer les deux solveurs sur le cluster Zen 

Les maillages (`*.msh`) et les résultats (`*.xdmf`, `*.h5`) ne sont pas versionnés : ils se régénèrent avec les scripts.

## Utilisation

Profilage avec Score-P, puis visualisation avec Cube :

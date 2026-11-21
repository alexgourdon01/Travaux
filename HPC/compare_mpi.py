# Compare Schwarz parallele MPI (2 processus) a Schwarz parallele sur 1 processus,
# sans copier de fichiers. A lancer juste apres un "mpirun -np 2 python3 dr_solve_dd_mpi.py".
import meshio
from dr_solve_dd_par import solve_dd_par

# 1) Solutions ecrites par le dernier lancement MPI, gardees en memoire
u_mpi_d = meshio.read('d.xdmf').point_data['u']
u_mpi_r = meshio.read('r.xdmf').point_data['u']

# 2) Schwarz parallele sur 1 processus : reecrit d.xdmf et r.xdmf
solve_dd_par('d', 'r')
u_par_d = meshio.read('d.xdmf').point_data['u']
u_par_r = meshio.read('r.xdmf').point_data['u']

# 3) Meme algorithme, meme calcul : on attend un ecart nul
print('disque    : ecart max MPI / 1 processus =', abs(u_mpi_d - u_par_d).max())
print('rectangle : ecart max MPI / 1 processus =', abs(u_mpi_r - u_par_r).max())
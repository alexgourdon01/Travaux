# Compare Schwarz parallele a Schwarz alterne, sans copier de fichiers
import meshio
from dr_solve_dd import solve_dd
from dr_solve_dd_par import solve_dd_par

# 1) Schwarz alterne : ecrit d.xdmf et r.xdmf, qu'on garde en memoire
solve_dd('d', 'r')
u_alt_d = meshio.read('d.xdmf').point_data['u']
u_alt_r = meshio.read('r.xdmf').point_data['u']

# 2) Schwarz parallele : reecrit d.xdmf et r.xdmf
solve_dd_par('d', 'r')
u_par_d = meshio.read('d.xdmf').point_data['u']
u_par_r = meshio.read('r.xdmf').point_data['u']

# 3) Ecart maximal entre les deux solutions, sur chaque sous-domaine
print('disque    : ecart max parallele / alterne =', abs(u_par_d - u_alt_d).max())
print('rectangle : ecart max parallele / alterne =', abs(u_par_r - u_alt_r).max())
print('max |u|   :', abs(u_alt_d).max())
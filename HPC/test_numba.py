import meshio, time
from dr_solve_np import assemble   # version compilée (@njit)

msh = meshio.read('dr.msh')
triangles = msh.cells_dict['triangle'][msh.cell_sets_dict['Domain']['triangle']]
points = msh.points

t0 = time.time()
Arow0, Acol0, Aval0, b0 = assemble.py_func(points, triangles)   # même fonction, SANS Numba
t1 = time.time()
Arow1, Acol1, Aval1, b1 = assemble(points, triangles)           # 1er appel : compilation + calcul
t2 = time.time()
Arow1, Acol1, Aval1, b1 = assemble(points, triangles)           # 2e appel : calcul seul
t3 = time.time()

print('écart Aval :', abs(Aval1 - Aval0).max(), ' écart b :', abs(b1 - b0).max(),
      ' indices identiques :', (Arow1 == Arow0).all() and (Acol1 == Acol0).all())
print(f'Python : {t1-t0:.3f} s   Numba 1er appel : {t2-t1:.3f} s   Numba 2e appel : {t3-t2:.4f} s')
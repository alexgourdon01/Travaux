import meshio, time
from dr_solve import build_system
from dr_solve_np import build_system_np

msh = meshio.read('dr.msh')

t0 = time.time(); A0, b0 = build_system(msh);    t1 = time.time()
A1, b1 = build_system_np(msh);                   t2 = time.time()

print('écart A :', abs(A1 - A0).max(), ' écart b :', abs(b1 - b0).max())
print(f'original : {t1-t0:.2f} s   tableaux : {t2-t1:.2f} s')
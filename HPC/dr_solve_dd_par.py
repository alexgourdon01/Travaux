import meshio
import numpy as np
from math import fabs, sqrt
import scipy
import time

from dr_solve_np import build_system_np as build_system

def solve_dd_par(*domains):
    msh=[meshio.read(domain+'.msh') for domain in domains]
    A0,b0 = build_system(msh[0])
    A1,b1 = build_system(msh[1])
    b = [b0, b1]
    A = [A0, A1]

    # Systems modification to enforce Dirichet condition for fictious boundaries
    bc_inside = [[], []] # indices of nodes for Dirichlet condition (reuse later)
    for d in range(len(domains)):
        d_other = (d+1)%len(domains)
        for k in msh[d].cell_sets_dict['BoundariesInside'+domains[d_other].capitalize()]['line']:
            elem = msh[d].cells_dict['line'][k]
            for i in elem:
                for j in A[d].indices[A[d].indptr[i]:A[d].indptr[i+1]]:
                    A[d][i,j] = 0. if i!=j else 1.
                bc_inside[d].append(i)
    # Sort indices and remove duplicates
    bc_inside = [sorted(list(set(bc_inside[d])),
                        key=lambda _:msh[d].points[_][0]+1000*msh[d].points[_][1])
                 for d in range(len(domains))]
    bc_other = [[], []] # indices of nodes as other domain's fictious boundary
    for d in range(len(domains)):
        for k in msh[d].cell_sets_dict['BoundariesInside'+domains[d].capitalize()]['line']:
            elem = msh[d].cells_dict['line'][k]
            for i in elem:
                bc_other[d].append(i)
    # Sort indices and remove duplicates
    bc_other = [sorted(list(set(bc_other[d])),
                        key=lambda _:msh[d].points[_][0]+1000*msh[d].points[_][1])
                 for d in range(len(domains))]

    # Initialize solutions sequence
    u = [np.zeros_like(b[d]) for d in range(len(domains))]

    for c in range(100):
        err = 0.
        # Schwarz parallele : on garde les solutions de l'iteration precedente
        u_old = [u[d].copy() for d in range(len(domains))]
        for d in range(len(domains)):
            d_other = (d+1)%len(domains)
            # Dirichlet value from other domain's PREVIOUS solution, keep track of max error
            for i,io in zip(bc_inside[d],bc_other[d_other]):
                err = max(err, abs(b[d][i] - u_old[d_other][io]))
                b[d][i] = u_old[d_other][io]
            u[d]=scipy.sparse.linalg.gmres(A[d],b[d])[0]

        # Stop iterations if convergence tolerance is reached
        if (err<1e-6):
            print("Convergence", c, err)
            break

    for d in range(len(domains)):
        msh[d].point_data={"u":u[d],}
        msh[d].write(domains[d]+'.xdmf')

if __name__=='__main__':
    t0 = time.time()
    solve_dd_par('d','r')
    t1 = time.time()
    print(f'temps de solution (parallele, 1 processus) :{t1-t0} secondes')
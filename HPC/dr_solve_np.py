import numpy as np
import scipy.sparse
from numba import njit

# Gradients des fonctions de base sur le triangle de référence
gradphi = np.array([[-1., -1.],
                    [ 1.,  0.],
                    [ 0.,  1.]])


@njit(cache=True)
def assemble(points, triangles):
    NElem = len(triangles)
    NNodes = len(points)
    Arow = np.empty(9*NElem, dtype=np.int64)   # ligne i de chaque coefficient
    Acol = np.empty(9*NElem, dtype=np.int64)   # colonne j de chaque coefficient
    Aval = np.empty(9*NElem)                   # valeur de chaque coefficient
    b = np.zeros(NNodes)
    n = 0
    for k in range(NElem):
        elem = triangles[k]
        X = [points[i] for i in elem]
        X0, X1, X2 = X
        J = np.array([[X1[0]-X0[0], X1[1]-X0[1]],
                      [X2[0]-X0[0], X2[1]-X0[1]]])
        det = J[0,0]*J[1,1] - J[0,1]*J[1,0]
        detJ = abs(det)
        Jinv = np.array([[J[1,1], -J[0,1]],
                         [-J[1,0], J[0,0]]])/det

        # les 3 gradients sur le vrai triangle, calculés une seule fois
        grad = np.empty((3, 2))
        for dofi in range(3):
            grad[dofi] = Jinv.dot(gradphi[dofi])

        for dofi in range(3):
            i = elem[dofi]
            b[i] += detJ/6
            for dofj in range(3):
                j = elem[dofj]
                Arow[n] = i
                Acol[n] = j
                Aval[n] = grad[dofi].dot(grad[dofj])*detJ*0.5
                n += 1
    return Arow, Acol, Aval, b


def build_system_np(msh):
    triangles = msh.cells_dict['triangle'][msh.cell_sets_dict['Domain']['triangle']]
    Arow, Acol, Aval, b = assemble(msh.points, triangles)
    NNodes = len(b)
    A = scipy.sparse.coo_matrix((Aval, (Arow, Acol)), shape=(NNodes, NNodes)).tocsr()
    # Dirichlet homogène sur le bord extérieur
    lines = msh.cells_dict['line'][msh.cell_sets_dict['BoundariesOutside']['line']]
    for i in np.unique(lines):
        A.data[A.indptr[i]:A.indptr[i+1]] = 0.
        A[i, i] = 1.
        b[i] = 0.
    return A, b


import meshio, time
import scipy.sparse.linalg

def solve_np(domain):
    msh = meshio.read(domain+'.msh')
    A, b = build_system_np(msh)
    u, _ = scipy.sparse.linalg.gmres(A, b)
    msh.point_data = {"u": u, }
    msh.write(domain+'.xdmf')

if __name__ == '__main__':
    t0 = time.time()
    solve_np('dr')
    t1 = time.time()
    print(f'temps solveur global (np) : {t1-t0} secondes')
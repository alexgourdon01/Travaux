import meshio
import numpy as np
from math import fabs, sqrt
import scipy

class Q3pt():
    """Defined on reference triangle"""
    def __init__(self):
        self.weights = [1/6, 1/6, 1/6]
        self.points = [np.array((0.5, 0.5, 0.)),
                       np.array((0.5, 0., 0.5)),
                       np.array((0., 0.5, 0.5))]

phi=[lambda p: p[2], # 1-x-y
     lambda p: p[1], # x
     lambda p: p[0]] # y
gradphi=[lambda p: (-1, -1),
         lambda p: (1, 0),
         lambda p: (0, 1)]
f = lambda x: 1.

def build_system(msh):
    NNodes=msh.points.shape[0]
    ## A=np.zeros((NNodes,NNodes)) # A is dense
    ARowdata = [{} for _ in range(NNodes)] # A is sparse
    b=np.zeros(NNodes)
    q=Q3pt()
    for k in msh.cell_sets_dict["Domain"]['triangle']:
        elem = msh.cells_dict['triangle'][k]
        X = [msh.points[i] for i in elem]
        X0,X1,X2 = X
        J = np.asarray((((X1[0]-X0[0]), (X1[1]-X0[1])),
                        ((X2[0]-X0[0]), (X2[1]-X0[1]))))
        """
        detJ = 0.5*fabs(J[0,0]*J[1,1] - J[0,1]*J[1,0])
        Jinv = np.asarray((((J[1,1], -J[0,1]),
                            (-J[0,1], J[0,0]))))/detJ
        """
        det  = J[0,0]*J[1,1] - J[0,1]*J[1,0]
        detJ = fabs(det)                              
        Jinv = np.asarray((((J[1,1], -J[0,1]),
                    (-J[1,0], J[0,0]))))/det    
        
        for w,p in zip(q.weights, q.points):
            xqf = p.dot(X)
            wq = detJ*w
            for dofi,i in enumerate(elem):
                b[i] += (phi[dofi](p))*f(xqf)*wq
                for dofj,j in enumerate(elem):
                    if not j in ARowdata[i]:
                        ARowdata[i][j] = 0.  # New coefficient in A
                    ARowdata[i][j] += Jinv.dot(gradphi[dofi](p)).dot(Jinv.dot(gradphi[dofj](p)))*wq
                    ## A[i,j] += Jinv.dot(gradphi[dofi](p)).dot(Jinv.dot(gradphi[dofj](p)))*wq

    # Dirichlet homogène sur le bord extérieur
    for k in msh.cell_sets_dict['BoundariesOutside']['line']:
        elem = msh.cells_dict['line'][k]
        for i in elem:
            ## A[i,:]=0.
            ## A[i,i]=1.
            for j in ARowdata[i]:
                ARowdata[i][j] = 0. if i!=j else 1.
            b[i]=0
    A=scipy.sparse.csr_matrix(([d for line in ARowdata for d in line.values()],
                               [j for line in ARowdata for j in line.keys()],
                               np.cumsum([0,] + [len(line) for line in ARowdata])))
    return A,b

def solve(domain):
    msh=meshio.read(domain+'.msh')
    A,b = build_system(msh)
    u,_=scipy.sparse.linalg.gmres(A,b)
    msh.point_data={"u":u,}
    msh.write(domain+'.xdmf')

if __name__=='__main__':
    solve('dr')
    solve('d')
    solve('r')

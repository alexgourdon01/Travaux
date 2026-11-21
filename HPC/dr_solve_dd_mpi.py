import meshio
import numpy as np
import scipy
import scipy.sparse.linalg
import mpi4py.MPI as MPI

from dr_solve_np import build_system_np as build_system

def solve_dd_mpi(*domains):
    comm = MPI.COMM_WORLD   # default communicator
    size = comm.size        # size of the communicator
    rank = comm.rank        # rank of the current process in the communicator
    assert size == len(domains), "il faut un processus par sous-domaine"

    d = rank                        # sous-domaine traite par ce processus
    d_other = (d+1)%len(domains)    # sous-domaine voisin = rang du voisin

    # Chaque processus ne lit et n'assemble que SON sous-domaine
    msh = meshio.read(domains[d]+'.msh')
    A, b = build_system(msh)

    # Bord fictif de d : condition de Dirichlet (comme dans dr_solve_dd.py)
    bc_inside = []
    for k in msh.cell_sets_dict['BoundariesInside'+domains[d_other].capitalize()]['line']:
        elem = msh.cells_dict['line'][k]
        for i in elem:
            for j in A.indices[A.indptr[i]:A.indptr[i+1]]:
                A[i,j] = 0. if i!=j else 1.
            bc_inside.append(i)
    bc_inside = sorted(list(set(bc_inside)),
                       key=lambda _:msh.points[_][0]+1000*msh.points[_][1])

    # Noeuds de d situes sur le bord fictif du voisin : valeurs a lui envoyer
    bc_other = []
    for k in msh.cell_sets_dict['BoundariesInside'+domains[d].capitalize()]['line']:
        elem = msh.cells_dict['line'][k]
        for i in elem:
            bc_other.append(i)
    bc_other = sorted(list(set(bc_other)),
                      key=lambda _:msh.points[_][0]+1000*msh.points[_][1])

    u = np.zeros_like(b)
    sendbuf = np.zeros(len(bc_other), dtype=np.float64)   # valeurs envoyees au voisin
    recvbuf = np.zeros(len(bc_inside), dtype=np.float64)  # valeurs recues du voisin
    err = np.zeros(1, dtype=np.float64)                   # erreur locale
    err_glob = np.zeros(1, dtype=np.float64)              # erreur globale

    for c in range(100):
        # Echange des valeurs d'interface de l'iteration precedente
        sendbuf[:] = u[bc_other]
        comm.Sendrecv(sendbuf, dest=d_other, recvbuf=recvbuf, source=d_other)

        # Dirichlet value from other domain's solution, keep track of max error
        err[0] = 0.
        for i, val in zip(bc_inside, recvbuf):
            err[0] = max(err[0], abs(b[i] - val))
            b[i] = val
        u = scipy.sparse.linalg.gmres(A, b)[0]

        # Erreur globale (maximum sur tous les processus) : tous s'arretent ensemble
        comm.Allreduce(err, err_glob, op=MPI.MAX)
        if (err_glob[0] < 1e-6):
            if rank == 0:
                print("Convergence", c, err_glob[0])
            break

    msh.point_data = {"u": u, }
    msh.write(domains[d]+'.xdmf')

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    comm.Barrier()
    t0 = MPI.Wtime()
    solve_dd_mpi('d', 'r')
    comm.Barrier()
    t1 = MPI.Wtime()
    if comm.rank == 0:
        print(f'temps de solution (MPI, {comm.size} processus) : {t1-t0} secondes')
import datetime
import time
import scipy.sparse as sparse


def iLUTP(H):
    print("iLUTP preconditioning.")
    st = time.time()

    N = len(H)
    sH = sparse.csc_matrix(H)
    sH_iLU = sparse.linalg.spilu(sH, fill_factor=3)
    M = sparse.linalg.LinearOperator((N,N), sH_iLU.solve)

    en = time.time()
    print('Time passed: {}\n'.format(datetime.timedelta(seconds=en - st)))
    return M, sH_iLU
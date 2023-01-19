import datetime
import time
import ilupp
import scipy.sparse as sparse
import numpy as np


def iLU0(H):
    print("iLU0 preconditioning.")
    st = time.time()

    L, U = ilupp.ilu0(sparse.csc_matrix(H))
    L = L.toarray()
    U = U.toarray()
    LU = np.dot(L, U)

    en = time.time()
    print('Time passed: {}\n'.format(datetime.timedelta(seconds=en - st)))
    return np.linalg.inv(LU)

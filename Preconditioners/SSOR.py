import numpy as np
import datetime
import time


def SSOR(A, omega):
    print("SSOR preconditioning.")
    st = time.time()
    L = np.tril(A, -1)
    D = np.zeros(A.shape, dtype=A.dtype)
    np.fill_diagonal(D, np.diag(A))

    LD = (1/omega*D + L)

    M = (omega / (2-omega)) * np.linalg.multi_dot([LD, np.linalg.inv(D), LD.T])
    M = np.linalg.inv(M)
    en = time.time()
    print('Time passed: {}\n'.format(datetime.timedelta(seconds=en - st)))

    return M
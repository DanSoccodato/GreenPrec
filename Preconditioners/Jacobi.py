import datetime
import time
import numpy as np


def Jacobi(H):
    print("Jacobi preconditioning.")
    st = time.time()
    N = len(H)

    P = np.zeros((N, N), dtype=H.dtype)
    for i in range(N):
        P[i, i] = 1. / H[i, i]
    en = time.time()
    print('Time passed: {}\n'.format(datetime.timedelta(seconds=en - st)))

    return P
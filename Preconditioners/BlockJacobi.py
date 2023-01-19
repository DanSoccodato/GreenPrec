import datetime
import time
import numpy as np
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from BlockMat import BlockMat, fill_blocks


def BlockJacobi(H):
    print("Block-Jacobi preconditioning.")
    st = time.time()
    Nblocks = len(H)
    G0 = fill_blocks(BlockMat((Nblocks, Nblocks)), H, 0.)

    for i in range(Nblocks):
        mat = np.linalg.inv(H[i, i])
        G0[i, i] = mat
    en = time.time()
    print('Time passed: {}\n'.format(datetime.timedelta(seconds=en - st)))

    return G0
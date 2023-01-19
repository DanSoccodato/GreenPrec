import numpy as np
import time
import datetime
import sys
import os
import warnings
# getting the name of the directory
# where this file is present.
current = os.path.dirname(os.path.realpath(__file__))
# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)
# adding the parent directory to
# the sys.path.
sys.path.append(parent)
from BlockMat import BlockMat, fill_blocks


def BlockGreen(H, order_diag, order_offdiag, offdiag=False):
    Nblocks = len(H)
    order_diag = min(order_diag, Nblocks)
    order_offdiag = min(order_offdiag, Nblocks)

    s = "Block-Green preconditioning. Off-diagonal blocks: {:}. Diagonal order: {:}."\
        .format(offdiag, order_diag)
    if offdiag:
        s += " Off-diagonal order: {:}".format(order_offdiag)
    print(s)
    st = time.time()

    G0 = fill_blocks(BlockMat((Nblocks, Nblocks)), H, 0.)

    for i in range(Nblocks):
        # Diagonal blocks
        # Initialize list of forbidden indices
        forbidden_neighbours = np.empty(order_diag, dtype=int)
        orderCounter = 1

        Sigma_i = sum_on_neighbours(H, i, forbidden_neighbours, orderCounter, order_diag, False)

        try:
            to_invert = -H[i, i] - Sigma_i
            assert not np.all(to_invert == 0)
            mat = np.linalg.inv(to_invert)
        except np.linalg.LinAlgError:
            warnings.warn("Matrix to invert is computationally singular; "
                          "adding small identity factor to regularize", RuntimeWarning)
            mat = np.linalg.inv(-H[i, i] - Sigma_i + np.eye(Sigma_i.shape[0]) * 1e-16)

        G0[i, i] = mat

        # Off-diagonal blocks
        if offdiag:
            for j in range(Nblocks):
                if i != j:
                    forbidden_neighbours = np.empty((order_offdiag + 1), dtype=int)
                    forbidden_neighbours[0] = i
                    i_Sigma_j = sum_on_neighbours(H, j, forbidden_neighbours,
                                                  orderCounter, order_offdiag + 1, False)
                    try:
                        i_G_jj = np.linalg.inv(-H[j, j] - i_Sigma_j)
                    except np.linalg.LinAlgError:
                        warnings.warn("Matrix to invert is computationally singular; "
                                      "adding small identity factor to regularize", RuntimeWarning)

                        i_G_jj = np.linalg.inv(-H[j, j] - i_Sigma_j +
                                               np.eye(i_Sigma_j.shape[0]) * 1e-16)

                    assert not np.all(G0[i, i] == 0)
                    mat = np.linalg.multi_dot([G0[i, i], H[i, j], i_G_jj])
                    G0[i, j] = mat

    en = time.time()
    print('Time passed: {}\n'.format(datetime.timedelta(seconds=en - st)))

    return G0


def sum_on_neighbours(Hbl, i, forbidden_neighbours, orderCounter, order, debug=False):
    sum_ = np.zeros(Hbl[i, i].shape, dtype=Hbl[i, i].dtype)
    forbidden_neighbours[orderCounter - 1] = i
    neighbours = get_and_exclude_neighbours(Hbl, i, forbidden_neighbours, orderCounter)

    if debug:
        print('considered index: {}\tforbidden_neighbours: {}\tdepth: {}\tneighbour_list: {} '
              .format(i, forbidden_neighbours[:orderCounter], orderCounter, neighbours))

    if orderCounter < order and len(neighbours) >= 1:
        orderCounter += 1

        for k in neighbours:
            to_invert = -Hbl[k, k] - sum_on_neighbours(Hbl, k, forbidden_neighbours, orderCounter, order)
            Hbl_inv_kk = np.linalg.inv(to_invert)
            sum_ += np.linalg.multi_dot([Hbl[i, k], Hbl_inv_kk, Hbl[k, i]])

    return sum_


def get_and_exclude_neighbours(Hbl, i, index_list, orderCounter):
    # Get neighbours of i
    neighbours = []
    for z in range(len(Hbl)):
        if np.count_nonzero(Hbl[i, z]) != 0:
            neighbours.append(z)
    neighbours = np.array(neighbours)

    # Exclude neighbours in index_list
    for n in range(0, min(orderCounter, len(index_list))):
        neighbours = np.delete(neighbours, np.where(neighbours == index_list[n]))

    return neighbours

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
from Graphs import get_and_exclude_neighbours, find_cycles_recursive, find_simple_paths


def BlockGreen(H, max_depth_diag, max_depth_offdiag,
               max_length_diag, max_length_offdiag,
               offdiag=False):

    s = "Block-Green preconditioning. Off-diagonal blocks: {:}. Diagonal order: {:}."\
        .format(offdiag, max_depth_diag)
    if offdiag:
        s += " Off-diagonal order: {:}".format(max_depth_offdiag)
    print(s)
    st = time.time()

    Nblocks = len(H)
    max_depth_diag = min(max_depth_diag, Nblocks)
    max_depth_offdiag = min(max_depth_offdiag, Nblocks)
    max_length_diag = min(max_length_diag, Nblocks)
    max_length_offdiag = min(max_length_offdiag, Nblocks)

    G0 = fill_blocks(BlockMat((Nblocks, Nblocks)), H, 0.)

    for i in range(Nblocks):
        # Diagonal blocks
        # Initialize list of forbidden indices
        forbidden_neighbours = np.empty(max_depth_diag, dtype=int)
        depth = 1

        # Sum over neighbours
        Sigma_i = sum_on_neighbours(H, i, forbidden_neighbours, depth, max_depth_diag)

        # Sum over loops
        for length in range(3, max_length_diag + 1):
            Sigma_i += sum_on_loops(H, i, length, max_depth_diag)

        mat = regularized_inv(-H[i, i] - Sigma_i)

        G0[i, i] += mat

        # Off-diagonal blocks
        if offdiag:
            for j in range(Nblocks):
                if i != j:
                    forbidden_neighbours = np.empty((max_depth_offdiag + 1), dtype=int)
                    forbidden_neighbours[0] = i
                    depth = 2

                    # Sum over neighbours
                    i_Sigma_j = sum_on_neighbours(H, j, forbidden_neighbours,
                                                  depth, max_depth_offdiag + 1)

                    i_G_jj = regularized_inv(-H[j, j] - i_Sigma_j)
                    assert not np.all(G0[i, i] == 0)
                    mat = np.linalg.multi_dot([G0[i, i], H[i, j], i_G_jj])
                    G0[i, j] += mat

                    # Sum over paths
                    sum_on_paths(H, G0, i, j, max_length_offdiag, max_depth_offdiag)

    en = time.time()
    print('Time passed: {}\n'.format(datetime.timedelta(seconds=en - st)))

    return G0


def sum_on_neighbours(Hbl, i, forbidden_neighbours, depth, max_depth, debug=False):
    sum_ = np.zeros(Hbl[i, i].shape, dtype=Hbl[i, i].dtype)
    forbidden_neighbours[depth - 1] = i

    neighbours = get_and_exclude_neighbours(Hbl, i, forbidden_neighbours, depth)

    if debug:
        print('considered index: {}\tforbidden_neighbours: {}\tdepth: {}\tneighbour_list: {} '
              .format(i, forbidden_neighbours[:depth], depth, neighbours))

    if depth < max_depth and len(neighbours) >= 1:
        depth += 1

        for k in neighbours:
            to_invert = -Hbl[k, k] - sum_on_neighbours(Hbl, k, forbidden_neighbours, depth, max_depth, debug)
            Hbl_inv_kk = regularized_inv(to_invert)
            sum_ += np.linalg.multi_dot([Hbl[i, k], Hbl_inv_kk, Hbl[k, i]])

    return sum_


def sum_on_loops(Hbl, i, length, max_depth):
    sum_ = np.zeros(Hbl[i, i].shape, dtype=Hbl[i, i].dtype)
    loops = list(find_cycles_recursive(Hbl, length, [i]))
    for loop in loops:
        v = loop[0]
        j = 0
        prod = np.eye(Hbl[i, i].shape[0])

        # print("\nloop = {:}".format(loop))
        forbidden_neighbours = np.empty(max_depth + len(loop) - 1, dtype=int)
        while j < len(loop) - 1:
            forbidden_neighbours[:j + 1] = loop[:j + 1]
            depth = 2 + j
            k = loop[j + 1]
            sigma = sum_on_neighbours(Hbl, k, forbidden_neighbours, depth, max_depth + j + 1)
            G_limited = regularized_inv(-Hbl[k, k] - sigma)
            prod = np.linalg.multi_dot([prod, Hbl[loop[j], k], G_limited])
            j += 1

        prod = np.dot(prod, Hbl[loop[-1], v])
        sum_ += prod
    return sum_


def sum_on_paths(Hbl, Gbl, i, j, max_length, max_depth):
    paths = list(find_simple_paths(Hbl, i, j, max_length))

    for path in paths:
        v = path[0]
        n = 0
        prod = Gbl[v, v]

        forbidden_neighbours = np.empty(max_depth + len(path) - 1, dtype=int)
        while n < len(path) - 1:
            forbidden_neighbours[:n + 1] = path[:n + 1]
            depth = 2 + n
            k = path[n + 1]
            sigma = sum_on_neighbours(Hbl, k, forbidden_neighbours, depth, max_depth + n + 1)
            # G_limited = 1. / (-Hbl[k, k] - sigma)
            G_limited = regularized_inv(-Hbl[k, k] - sigma)
            prod = np.linalg.multi_dot([prod, Hbl[path[n], k], G_limited])
            n += 1
        Gbl[i, j] += prod


def regularized_inv(to_invert):
    try:
        assert not np.all(to_invert == 0)
        mat = np.linalg.inv(to_invert)
    except np.linalg.LinAlgError:
        warnings.warn("Matrix to invert is computationally singular; "
                      "adding small identity factor to regularize", RuntimeWarning)
        mat = np.linalg.inv(to_invert + np.eye(to_invert.shape[0]) * 1e-16)
    return mat

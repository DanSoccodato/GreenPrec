import datetime
import time
import numpy as np
import sys
import os
import warnings
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from Graphs import find_cycles_recursive, get_and_exclude_neighbours, find_simple_paths


def Green(H, max_depth_diag, max_depth_offdiag,
          max_length_diag, max_length_offdiag,
          Verbose, offdiag=False):

    s = "Green preconditioning. Off-diagonal blocks: {:}.\nDiagonal max_depth: {:}.\tDiagonal max_length: {:}" \
        .format(offdiag, max_depth_diag, max_length_diag)
    if offdiag:
        s += "\nOff-diagonal max_depth: {:}.\tOff-diagonal max_length: {:}".format(max_depth_offdiag,
                                                                                   max_length_offdiag)
    print(s)
    st = time.time()

    N = len(H)
    max_depth_diag = min(max_depth_diag, N)
    max_depth_offdiag = min(max_depth_offdiag, N)
    max_length_diag = min(max_length_diag, N)
    max_length_offdiag = min(max_length_offdiag, N)

    if max_length_diag < 3:
        warnings.warn("Parameter max_length_diag set to value < 3, this"
                      " will have no effect. Consider setting the value to 3 or more.", RuntimeWarning)

    if offdiag and max_length_offdiag < 3:
        warnings.warn("Parameter max_length_offdiag set to value < 3, this"
                      " will have no effect. Consider setting the value to 3 or more.", RuntimeWarning)

    G0 = np.zeros((N, N), dtype=H.dtype)

    for i in range(N):
        if Verbose > 0:
            try:
                if i % (int(N / 10)) == 0:
                    print("Progress = {:.2f}%"
                          .format(i / N * 100))
            except ZeroDivisionError:
                raise ZeroDivisionError("Dimension of matrix < 10, please set Verbose = 0 or increase matrix "
                                        "dimension.")

        # Diagonal elements
        # Initialize list of forbidden indices
        forbidden_neighbours = np.empty(max_depth_diag, dtype=int)
        depth = 1

        # Sum over neighbours
        Sigma_i = sum_on_neighbours(H, i, forbidden_neighbours, depth, max_depth_diag)

        # Sum over loops
        for length in range(3, max_length_diag + 1):
            Sigma_i += sum_on_loops(H, i, length, max_depth_diag)

        G0[i, i] += 1. / (-H[i, i] - Sigma_i)

        # Off-diagonal elements
        if offdiag:
            for j in range(N):
                if j != i:
                    forbidden_neighbours = np.empty((max_depth_offdiag + 1), dtype=int)
                    forbidden_neighbours[0] = i
                    depth = 2

                    # Sum over neighbours
                    i_Sigma_j = sum_on_neighbours(H, j, forbidden_neighbours,
                                                  depth, max_depth_offdiag + 1)
                    i_G_jj = 1. / (-H[j, j] - i_Sigma_j)
                    G0[i, j] += G0[i, i] * H[i, j] * i_G_jj

                    # Sum over paths
                    sum_on_paths(H, G0, i, j, max_length_offdiag, max_depth_offdiag)

    en = time.time()
    print('Time passed: {}\n'.format(datetime.timedelta(seconds=en - st)))
    return G0


def sum_on_neighbours(H, i, forbidden_neighbours, depth, max_depth, debug=False):
    sum_ = 0
    forbidden_neighbours[depth - 1] = i

    neighbours = get_and_exclude_neighbours(H, i, forbidden_neighbours, depth)

    if debug:
        print("\ndepth = ", depth, " max_depth = ", max_depth)
        print('considered index: {}\tforbidden_neighbours: {}\tdepth: {}\tneighbour_list: {} '
              .format(i, forbidden_neighbours[:depth], depth, neighbours))

    if depth < max_depth and len(neighbours) >= 1:
        depth += 1
        for k in neighbours:
            sum_ += H[i, k] * 1. / (-H[k, k]
                                    - sum_on_neighbours(H, k, forbidden_neighbours, depth, max_depth, debug)
                                    ) * H[k, i]

    return sum_


def sum_on_loops(H, i, length, max_depth):
    sum_ = 0
    loops = list(find_cycles_recursive(H, length, [i]))
    for loop in loops:
        v = loop[0]
        j = 0
        prod = 1

        # print("\nloop = {:}".format(loop))
        forbidden_neighbours = np.empty(max_depth + len(loop) - 1, dtype=int)
        while j < len(loop) - 1:
            forbidden_neighbours[:j + 1] = loop[:j + 1]
            depth = 2 + j
            k = loop[j + 1]
            sigma = sum_on_neighbours(H, k, forbidden_neighbours, depth, max_depth + j + 1)
            G_limited = 1. / (-H[k, k] - sigma)
            prod = prod * H[loop[j], k] * G_limited
            j += 1

        prod *= H[loop[-1], v]
        sum_ += prod
    return sum_


def sum_on_paths(H, G, i, j, max_length, max_depth):
    paths = list(find_simple_paths(H, i, j, max_length))

    for path in paths:
        v = path[0]
        n = 0
        prod = G[v, v]

        forbidden_neighbours = np.empty(max_depth + len(path) - 1, dtype=int)
        while n < len(path) - 1:
            forbidden_neighbours[:n + 1] = path[:n + 1]
            depth = 2 + n
            k = path[n + 1]
            sigma = sum_on_neighbours(H, k, forbidden_neighbours, depth, max_depth + n + 1)
            G_limited = 1. / (-H[k, k] - sigma)
            prod = prod * H[path[n], k] * G_limited
            n += 1
        G[i, j] += prod





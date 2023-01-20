import datetime
import time
import numpy as np
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from Graphs import find_cycles_recursive


def Green(H, order_diag, order_offdiag, max_length, Verbose, offdiag=False):
    s = "Green preconditioning. Off-diagonal entries: {:}. Diagonal order: {}."\
        .format(offdiag, order_diag)
    if offdiag:
        s += " Off-diagonal order: {}".format(order_offdiag)
    print(s)

    st = time.time()

    # TODO: Define forbidden_neighbours as a list and not as a np.empty().
    #  Use the stash saved in git

    N = len(H)
    order_diag = min(order_diag, N)
    order_offdiag = min(order_offdiag, N)
    max_length = min(max_length, N)

    G0 = np.zeros((N, N), dtype=H.dtype)

    for i in range(N):
        if Verbose > 0:
            try:
                if i % (int(N / 10)) == 0:
                    print("Progress = {:.2f}%"
                          .format(i / N * 100))
            except ZeroDivisionError:
                raise ZeroDivisionError("Dimension of matrix < 10, please set Verbose to 0 or increase matrix "
                                        "dimension.")

        # Diagonal elements
        # Initialize list of forbidden indices
        forbidden_neighbours = np.empty(order_diag, dtype=int)
        depth = 1

        #Sum over neighbours
        Sigma_i = sum_on_neighbours(H, i, forbidden_neighbours, depth, order_diag)

        # Sum over loops
        for length in range(3, max_length + 1):
            loops = list(find_cycles_recursive(H, length, [i]))
            for loop in loops:
                i = loop[0]
                j = 0
                prod = 1

                print("\nloop = {:}".format(loop))
                forbidden_neighbours = np.empty(order_diag + len(loop) - 1, dtype=int)
                while j < len(loop)-1:

                    forbidden_neighbours[:j+1] = loop[:j+1]
                    depth = 2 + j
                    k = loop[j+1]

                    sigma = sum_on_neighbours(H, k, forbidden_neighbours, depth, order_diag + j + 1, True)

                    G_limited = 1. / (-H[k, k] - sigma)


                    prod = prod * H[loop[j], k] * G_limited
                    j += 1

                prod *= H[loop[-1], i]
                Sigma_i += prod

        G0[i, i] = 1. / (-H[i, i] - Sigma_i)

        # Non-diagonal elements
        if offdiag:
            for j in range(N):
                if j != i:  # Must exclude diagonal elements
                    forbidden_neighbours = np.empty((order_offdiag + 1), dtype=int)
                    forbidden_neighbours[0] = i
                    depth = 2

                    i_Sigma_j = sum_on_neighbours(H, j, forbidden_neighbours,
                                                  depth, order_offdiag + 1)
                    i_G_jj = 1. / (-H[j, j] - i_Sigma_j)
                    G0[i, j] = G0[i, i] * H[i, j] * i_G_jj

    en = time.time()
    print('Time passed: {}\n'.format(datetime.timedelta(seconds=en - st)))
    return G0


def sum_on_neighbours(H, i, forbidden_neighbours, depth, order, debug=False):
    sum_ = 0
    forbidden_neighbours[depth - 1] = i

    neighbours = get_and_exclude_neighbours(H, i, forbidden_neighbours, depth)

    if debug:
        print("\ndepth = ", depth, " order = ", order)
        print('considered index: {}\tforbidden_neighbours: {}\tdepth: {}\tneighbour_list: {} '
              .format(i, forbidden_neighbours[:depth], depth, neighbours))

    if depth < order and len(neighbours) >= 1:
        depth += 1
        for k in neighbours:
            sum_ += H[i, k] * 1. / (-H[k, k]
                                    - sum_on_neighbours(H, k, forbidden_neighbours, depth, order, debug)
                                    ) * H[k, i]

    return sum_


def get_and_exclude_neighbours(H, i, index_list, orderCounter):
    # Get neighbours of i
    neighbours = np.nonzero(H[i, :])[0]

    # Exclude neighbours in index_list
    for n in range(0, min(orderCounter, len(index_list))):
        neighbours = np.delete(neighbours, np.where(neighbours == index_list[n]))

    return neighbours

import datetime
import time

import numpy as np


def Green(H, order_diag, order_offdiag, Verbose, offdiag=False):
    s = "Green preconditioning. Off-diagonal entries: {:}. Diagonal order: {}."\
        .format(offdiag, order_diag)
    print("Testing git feature")
    if offdiag:
        s += " Off-diagonal order: {}".format(order_offdiag)
    print(s)

    st = time.time()

    N = len(H)
    order_diag = min(order_diag, N)
    order_offdiag = min(order_offdiag, N)

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
        orderCounter = 1

        # Recursive computation of selfenergy for row i, up to user-defined order
        # print(12* '-~' + ' Entering sum_on_neighbours for row i = {:} '.format(i) + 12* '-~' + '\n')
        Sigma_i = sum_on_neighbours(H, i, forbidden_neighbours, orderCounter, order_diag)
        # print(12* '-~' +' Exiting sum_on_neighbours. Sum is: {:.4f} '.format(Sigma_i)
        #     + 12* '-~' + '\n\n')

        G0[i, i] = 1. / (-H[i, i] - Sigma_i)

        # Non-diagonal elements
        if offdiag:
            for j in range(N):
                if j != i:  # Must exclude diagonal elements
                    forbidden_neighbours = np.empty((order_offdiag + 1), dtype=int)
                    forbidden_neighbours[0] = i
                    orderCounter = 2

                    i_Sigma_j = sum_on_neighbours(H, j, forbidden_neighbours,
                                                  orderCounter, order_offdiag + 1)
                    i_G_jj = 1. / (-H[j, j] - i_Sigma_j)
                    G0[i, j] = G0[i, i] * H[i, j] * i_G_jj

    en = time.time()
    print('Time passed: {}\n'.format(datetime.timedelta(seconds=en - st)))
    return G0


def sum_on_neighbours(H, i, forbidden_neighbours, orderCounter, order):
    sum_ = 0
    forbidden_neighbours[orderCounter - 1] = i
    neighbours = get_and_exclude_neighbours(H, i, forbidden_neighbours, orderCounter)

    # print('considered index: {}\tforbidden_neighbours: {}\tdepth: {}\tneighbour_list: {} '
    #      .format(i, forbidden_neighbours[:orderCounter], orderCounter, neighbours))

    if orderCounter < order and len(neighbours) >= 1:
        orderCounter += 1

        for k in neighbours:
            sum_ += H[i, k] * 1. / (-H[k, k]
                                    - sum_on_neighbours(H, k, forbidden_neighbours, orderCounter, order)
                                    ) * H[k, i]

    return sum_


def get_and_exclude_neighbours(H, i, index_list, orderCounter):
    # Get neighbours of i
    neighbours = np.nonzero(H[i, :])[0]

    # Exclude neighbours in index_list
    for n in range(0, min(orderCounter, len(index_list))):
        neighbours = np.delete(neighbours, np.where(neighbours == index_list[n]))

    return neighbours

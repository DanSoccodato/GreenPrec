import numpy as np
import time
import datetime
from . import BlockMat


def defineBlocks(Nblocks, N, Load=False):
    if not Load:
        block_size = int(N / Nblocks)

        start_blocks = [0]
        for i in range(1, Nblocks):
            start_blocks.append(
                np.random.randint(low=start_blocks[i - 1] + 1,
                                  high=min(start_blocks[i - 1]
                                           + 1.8*block_size, N - 1))  # 1.8 empirical parameter
                                )

        end_blocks = []

        for i in range(Nblocks - 1):
            end_blocks.append(start_blocks[i + 1] - 1)
        end_blocks.append(N - 1)
    else:
        with open("pickles/blocks.txt", "r") as f:
            line = f.readline()
            start_blocks = [int(i) for i in (line.split('[')[-1].split(']')[0].split(','))]
            line = f.readline()
            end_blocks = [int(i) for i in (line.split('[')[-1].split(']')[0].split(','))]

    print('start_blocks:\n{:}'.format(start_blocks))
    print('\nend_blocks:\n{:}'.format(end_blocks))

    return start_blocks, end_blocks


def createTestMatrix(start_blocks, end_blocks, block_sparsity, offDiag_sparsity, symmetry=False):
    Nblocks = len(start_blocks)
    N = end_blocks[-1] + 1

    if not symmetry:
        st = time.time()
        H = np.zeros((N, N), dtype=float)
        Hbl = BlockMat.mat2block(H, start_blocks, end_blocks)

        for i in range(Nblocks):
            for j in range(Nblocks):
                if i == j:
                    Hbl[i, j] = np.random.uniform(0, 1. / block_sparsity, size=Hbl[i, j].shape)
                else:
                    Hbl[i, j] = np.random.uniform(0, 1. / offDiag_sparsity, size=Hbl[i, j].shape)

                Hbl[i, j][np.where(Hbl[i, j] > 1.)] = 0.

        H = BlockMat.block2mat(Hbl, start_blocks, end_blocks)
        np.fill_diagonal(H, 7.0)

        en = time.time()
        print('H_sparse created. Time passed: {}'.format(datetime.timedelta(seconds=en - st)))
        print('Sparsity of H_sparse is: {:.5f}'.format(np.count_nonzero(H) / (N * N)))

    else:
        st = time.time()
        H = np.zeros((N, N), dtype=float)
        Hbl = BlockMat.mat2block(H, start_blocks, end_blocks)

        for i in range(Nblocks):
            Hbl[i, i] = np.random.uniform(0, 1. / block_sparsity, size=Hbl[i, i].shape)
            Hbl[i, i][np.where(Hbl[i, i] > 1.)] = 0.

            # Symmetrize diagonal blocks
            for s in range(len(Hbl[i, i])):
                for m in range(s + 1, len(Hbl[i, i])):
                    Hbl[i, i][m, s] = Hbl[i, i][s, m]

            for j in range(i + 1, Nblocks):
                Hbl[i, j] = np.random.uniform(0, 1. / offDiag_sparsity, size=Hbl[i, j].shape)
                Hbl[i, j][np.where(Hbl[i, j] > 1.)] = 0.
                Hbl[j, i] = Hbl[i, j].T

        H = BlockMat.block2mat(Hbl, start_blocks, end_blocks)
        np.fill_diagonal(H, 7.0)

        en = time.time()
        print('H_sym created. Time passed: {}'.format(datetime.timedelta(seconds=en - st)))
        print('Sparsity of H_sym is: {:.5f}'.format(np.count_nonzero(H) / (N * N)))

    return H

import numpy as np


class BlockMat:

    def __init__(self, shape):

        if type(shape) != tuple or len(shape) != 2:
            raise TypeError("Shape must be a 2D-tuple")

        for i in shape:
            if type(i) != int:
                raise TypeError("Shape must be a tuple of integers")

        self.shape = shape
        self.blocks = [[np.array([]) for j in range(shape[1])] for i in range(shape[0])]
        self._current_index = 0

    def __setitem__(self, index, newValue):
        if type(index) == tuple and len(index) == 2:
            self.blocks[index[0]][index[1]] = newValue
        else:
            raise TypeError("Setting new value in blockMat object: index must be 2-D tuple")

    def __getitem__(self, index):
        if type(index) == int:
            return self.blocks[index]
        elif type(index) == tuple and len(index) == 2:
            return self.blocks[index[0]][index[1]]
        else:
            raise TypeError("Indexing blockMat object: index must be integer or tuple")

    def __iter__(self):
        return self

    def __next__(self):
        if self._current_index < self.shape[0]:
            member = self.blocks[self._current_index]
            self._current_index += 1
            return member
        raise StopIteration

    def __str__(self):
        s = ''
        for i in self.blocks:
            s += '{}\n'.format(i)
        return s

    def __len__(self):
        return len(self.blocks)


def mat2block(mat, start_blocks, end_blocks):

    if len(start_blocks) != len(end_blocks):
        raise ValueError("Length of start_blocks is not the same of end_blocks")

    Nblocks = len(start_blocks)

    blockmat = BlockMat((Nblocks, Nblocks))
    for i in range(Nblocks):
        for j in range(Nblocks):
            blockmat[i, j] = mat[start_blocks[i]:end_blocks[i] + 1, start_blocks[j]:end_blocks[j] + 1]

    return blockmat


def block2mat(blockmat, start_blocks, end_blocks):

    if len(start_blocks) != len(end_blocks):
        raise ValueError("Length of start_blocks is not the same of end_blocks")

    Nblocks = len(start_blocks)
    N = end_blocks[-1] + 1

    mat = np.zeros((N, N), dtype=blockmat[0, 0].dtype)
    for i in range(Nblocks):
        for j in range(Nblocks):
            mat[start_blocks[i]:end_blocks[i] + 1, start_blocks[j]:end_blocks[j] + 1] = blockmat[i, j]

    return mat


def fill_blocks(blockmat1, blockmat2, value=None):

    Nbl = len(blockmat1)
    if value is not None:
        for i in range(Nbl):
            for j in range(Nbl):
                blockmat1[i, j] = value * np.ones(blockmat2[i, j].shape, dtype=blockmat2[i, j].dtype)
    else:
        for i in range(Nbl):
            for j in range(Nbl):
                blockmat1[i, j] = blockmat2[i, j]

    return blockmat1

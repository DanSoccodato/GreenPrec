import numpy as np


def get_and_exclude_neighbours(in_, i, index_list, depth):
    if type(in_) == np.ndarray:
        return _get_and_exclude_neighbours_dns(in_, i, index_list, depth)
    else:
        # TODO: else is too general, find a way to distinguish BlockMat class
        adj = create_adjacency_matrix(in_)
        return _get_and_exclude_neighbours_dns(adj, i, index_list, depth)


def find_cycles_recursive(in_, L, cycle):
    if type(in_) == np.ndarray:
        return _find_cycles_recursive_dns(in_, L, cycle)
    else:
        # TODO: else is too general, find a way to distinguish BlockMat class
        adj = create_adjacency_matrix(in_)
        return _find_cycles_recursive_dns(adj, L, cycle)


def find_simple_paths(in_, source, target, cutoff):
    if type(in_) == np.ndarray:
        return _nx_find_simple_paths_dns(in_, source, target, cutoff)
    # TODO: else is too general, find a way to distinguish BlockMat class
    else:
        adj = create_adjacency_matrix(in_)
        return _nx_find_simple_paths_dns(adj, source, target, cutoff)


def _get_and_exclude_neighbours_dns(mat, i, index_list, depth):
    # Get neighbours of i
    neighbours = np.nonzero(mat[i, :])[0]

    # Exclude neighbours in index_list
    for n in range(0, min(depth, len(index_list))):
        neighbours = np.delete(neighbours, np.where(neighbours == index_list[n]))

    return neighbours


def _find_cycles_recursive_dns(mat, L, cycle):
    successors = np.nonzero(mat[cycle[-1]])[0]
    if len(cycle) == L:
        if cycle[0] in successors:
            yield cycle
    elif len(cycle) < L:
        for v in successors:
            if v in cycle:
                continue
            yield from _find_cycles_recursive_dns(mat, L, cycle + [v])


# Adapted from the networkx library, all_symple_paths():
# https://networkx.org/documentation/stable/_modules/networkx/algorithms/simple_paths.html#all_simple_paths
def _nx_find_simple_paths_dns(mat, source, target, cutoff):
    targets = {target}
    visited = {source: True}
    neighbours = iter(np.nonzero(mat[source])[0])
    stack = [neighbours]
    while stack:
        children = stack[-1]
        child = next(children, None)
        if child is None:
            stack.pop()
            visited.popitem()
        elif len(visited) < cutoff:
            if child in visited:
                continue
            if child in targets and len(visited) != 1:  # Exclude paths of length 2
                yield list(visited) + [child]
            visited[child] = True
            if targets - set(visited.keys()):  # expand stack until find all targets
                stack.append(iter(np.nonzero(mat[child])[0]))
            else:
                visited.popitem()
        else:  # len(visited) == cutoff:
            for target in (targets & (set(children) | {child})) - set(visited.keys()):
                if len(list(visited)) != 1:  # Exclude paths of length 2
                    yield list(visited) + [target]
            stack.pop()
            visited.popitem()


def create_adjacency_matrix(block_mat):

    Nbl = len(block_mat)
    ad = np.zeros((Nbl, Nbl))

    for i in range(Nbl):
        for j in range(Nbl):
            if np.count_nonzero(block_mat[i, j]) != 0:
                ad[i, j] = 1.

    np.fill_diagonal(ad, 0.)
    return ad

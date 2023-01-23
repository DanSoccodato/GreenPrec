import numpy as np


def find_cycles_recursive(graph, L, cycle):
    successors = np.nonzero(graph[cycle[-1]])[0]
    if len(cycle) == L:
        if cycle[0] in successors:
            yield cycle
    elif len(cycle) < L:
        for v in successors:
            if v in cycle:
                continue
            yield from find_cycles_recursive(graph, L, cycle + [v])


def get_and_exclude_neighbours(H, i, index_list, depth):
    # Get neighbours of i
    neighbours = np.nonzero(H[i, :])[0]

    # Exclude neighbours in index_list
    for n in range(0, min(depth, len(index_list))):
        neighbours = np.delete(neighbours, np.where(neighbours == index_list[n]))

    return neighbours


# Adapted from the networkx library, all_symple_paths():
# https://networkx.org/documentation/stable/_modules/networkx/algorithms/simple_paths.html#all_simple_paths
def nx_find_simple_paths(G, source, target, cutoff):
    targets = {target}
    visited = {source: True}
    stack = [iter(np.nonzero(G[source])[0])]
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
                stack.append(iter(np.nonzero(G[child])[0]))
            else:
                visited.popitem()
        else:  # len(visited) == cutoff:
            for target in (targets & (set(children) | {child})) - set(visited.keys()):
                if len(list(visited)) != 1:  # Exclude paths of length 2
                    yield list(visited) + [target]
            stack.pop()
            visited.popitem()
# def find_all_loops(H, length):
#     temp = []
#     h = np.copy(H)
#     np.fill_diagonal(h, 0.)  # to exclude self-loops
#
#     # Use Depth First Search algorithm to find all cycles up to fixed length
#     for l in range(2, length + 1):
#         print("Calling find_cycle() for length = ", l)
#         count, loops_l = find_cycle(h, l)
#         if count != 0:
#             for el in loops_l:
#                 temp.append(el)
#         print("Done.")
#
#     # Perform cyclic permutations
#     loops = []
#     for i in range(len(temp)):
#         if len(temp[i]) > 2:
#             permutations = cyclic_perm(temp[i])
#             for perm in permutations:
#                 loops.append(perm)
#
#     # Create dictionary with vertices as keys
#     groups = {}
#     for l in sorted(loops):
#         groups.setdefault(l[0], []).append(l)
#
#     return groups
#
#
# # find_cycle() returns loops starting only from one node, we need them starting from every node:
# # we must perform cyclic permutations
# def cyclic_perm(a):
#     n = len(a)
#     b = [[a[i - j] for i in range(n)] for j in range(n)]
#     return b
#
#
# def find_cycle(graph, l):
#     V = len(graph)
#     loops = []
#     # all vertex are marked un-visited initially.
#     marked = [False] * V
#
#     # Searching for cycle by using v-l+1 vertices
#     count = 0
#     for i in range(V - (l - 1)):
#         count = DFS(graph, marked, l - 1, i, i, count, [i], loops)
#
#         # ith vertex is marked as visited and
#         # will not be visited again.
#         marked[i] = True
#
#     return (count, loops)
#
#
# def DFS(graph, marked, l, vert, start, count, path, path_collection):
#     V = len(graph)
#
#     # mark the vertex vert as visited
#     marked[vert] = True
#
#     # if the path of length (n-1) is found
#     if l == 0:
#
#         # mark vert as un-visited to make
#         # it usable again.
#         marked[vert] = False
#
#         # Check if vertex vert can end with
#         # vertex start
#         if graph[vert][start] != 0:
#             count = count + 1
#             path_collection.append(path)
#             return count
#         else:
#             return count
#
#     # For searching every possible path of
#     # length (n-1)
#     for i in range(V):
#         if marked[i] == False and graph[vert][i] != 0:
#             # DFS for searching path by decreasing
#             # length by 1
#             next_path = path[:]
#             next_path.append(i)
#             count = DFS(graph, marked, l - 1, i, start, count, next_path, path_collection)
#
#     # marking vert as unvisited to make it
#     # usable again.
#     marked[vert] = False
#     return count

import numpy as np

# TEST FOR GIT
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

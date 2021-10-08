# g = Graph()

# given a list of n-tuples (where tuples correspond to existing cliques in a graph g),
# return a list of all n+1-tuples in the graph by checking all connecting nodes
def n_plus_1_cliques(g, n_cliques):
    np1_cliques = []
    for clique in n_cliques:
        adjacent = []
        for node in clique:
            from_nodes = [x[1] for x in g.edges(from_node=node)]
            adjacent.append(set(from_nodes).difference(clique))
        common_nodes = set.intersection(*adjacent)
        for cn in common_nodes:
            np1_cliques.append(clique + (cn,))
    return set(np1_cliques)

def get_score_of_clique(clique, g):
    score = 0
    for a, b in combinations(clique, 2):
        score += g.edge(a, b)
    return score

# builds graph of overlapping cliques given list of cliques
def build_clique_graph(cg):

    cliques_graph = Graph()
    for c in cg:
        cliques_graph.add_node(c)

    for i, x in enumerate(cg):
        for j, y in enumerate(cg[i:]):
            if not (set(x).intersection(set(y)) == set()):
                cliques_graph.add_edge(x, y, bidirectional=True)

    return cliques_graph

# given a list of sequences, returns only those sequences that are not subsets of any other
def inds_of_subsequences(seq_list, sqs, g):
    seq_list = list(seq_list)
    pat_seqs = [sqs[i] for i in seq_list]
    pat_seqs = sorted(pat_seqs, key=lambda x: (len(x)))
    inds_to_remove = []
    for i, x in enumerate(pat_seqs[:-1]):
        for j, y in enumerate(pat_seqs[i + 1:]):
            if set(x).issubset(set(y)):
                # score_x = len(g.edges(from_node=seq_list[i]))
                # score_y = len(g.edges(from_node=seq_list[j]))
                # worse_ind = i if score_x < score_y else (i + j)
                inds_to_remove.append(i)
    # duplicates_removed = [seq_list[i] for i in range(len(seq_list)) if not (i in inds_to_remove)]
    return inds_to_remove

# print('building graph...')
# g = Graph()
# for i, sq in enumerate(sqs):
#     g.add_node(i)

# for i, sq_pair in enumerate(pairs_to_compare):
#     score = scores[i]
#     if score > dist_thresh:
#         g.add_edge(sq_pair[0], sq_pair[1], score, bidirectional=True)

#     if not i % (len(pairs_to_compare) // 10):
#         print(f'   {i} / {len(pairs_to_compare)} nodes... \n'
#               f'   {len(g.edges())} total edges added')

# orphaned_nodes = set(g.nodes(out_degree=0)).intersection(g.nodes(in_degree=0))
# print(f'removing {len(orphaned_nodes)} orphaned nodes from graph.')
# for x in orphaned_nodes:
#     g.del_node(x)

# # # get all 2-cliques
# cliques2 = [(x[0], x[1]) for x in g.edges()]
# cliques3 = n_plus_1_cliques(g, cliques2)
# cliques4 = n_plus_1_cliques(g, cliques3)
# cliques5 = n_plus_1_cliques(g, cliques4)
# cliques6 = n_plus_1_cliques(g, cliques5)
# cliques7 = n_plus_1_cliques(g, cliques6)

# cg = build_clique_graph(cliques4)
# clique_components = cg.components()

# graph_regions = []
# for clique_group in clique_components:
#     clique_region = set.union(*[set(x) for x in clique_group])
#     graph_regions.append(clique_region)


# # graph_components = g.components()
# # # graph_components = [remove_subsets_from_list_of_seqs(x, sqs) for x in graph_components]

# # refining_steps = 20
# # min_cardinality = 4
# # max_cardinality = 10

# # graph_components = [x for x in g.components() if len(x) > min_cardinality]
# # print([len(x) for x in graph_components])

# # for gc in graph_components:
# #     graph_region = list(gc)
# #     dupes = inds_of_subsequences(graph_region, sqs, g)
# #     print([graph_region[ind] for ind in dupes])
# #     for ind in dupes:
# #         g.del_node(graph_region[ind])
# # graph_components = [x for x in g.components() if len(x) > min_cardinality]
# # print([len(x) for x in graph_components])

# # for f in range(2, refining_steps + 2):

# #     graph_components = [x for x in g.components() if len(x) > min_cardinality]
# #     refine_score_thresh = sorted_scores[int(prop_edges_to_keep * len(scores) / f)]

# #     print([len(x) for x in graph_components])
# #     print(f'refining step {f}')

# #     for gc in graph_components:
# #         if len(gc) < max_cardinality:
# #             continue
# #         edges = []
# #         for node in gc:
# #             edges = g.edges(from_node=node)
# #             for edge in edges:
# #                 if edge[2] < refine_score_thresh:
# #                     g.del_edge(edge[0], edge[1])
# #     graph_components = [x for x in g.components() if len(x) > min_cardinality]






    
    



# # # get all 3-cliques






# # patterns = []
# # for gr in graph_regions:






# # print('finding all paths..')
# # all_paths = g.all_pairs_shortest_paths()

# # duples = {}
# # for x in all_paths.keys():
# #     for y in all_paths[x].keys():
# #         if x == y or all_paths[x][y] == np.inf:
# #             continue
# #         duples[(x, y)] = all_paths[x][y]

# # x = sorted(duples.keys(), key=lambda x: duples[x])

# # good_paths = []
# # for i in range(num_paths_to_check):
# #     p = g.shortest_path(x[i][0], x[i][1], memoize=True)

# #     print(p)

# #     if len(p[1]) < min_occurrences:
# #         continue

# #     # check to make sure this isn't too similar to any other path
# #     similar_paths = [
# #         g for g in good_paths
# #         if len(set(g[1]).intersection(set(p[1]))) > len(p[1]) * 0.5
# #         ]
# #     if not similar_paths:
# #         good_paths.append(p)
# #         continue

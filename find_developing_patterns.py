import music21 as m21
import numpy as np
from collections import Counter, defaultdict
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
from numpy.lib.stride_tricks import as_strided
from itertools import combinations
import affine_needleman_wunsch as nw
from sklearn.cluster import DBSCAN
from scipy.sparse import lil_matrix, dia_matrix

import visualize_motifs as vm
import generate_viewpoints as gv
import markov_probs as mp
import pattern
from importlib import reload
reload(vm)
reload(gv)
reload(mp)
reload(nw)
reload(pattern)

cardinalities = [10]
max_length_difference = 2
min_occurrences = 4
prop_edges_to_keep = 0.05
seq_compare_dist_threshold = 50
tune_name = 'fallinggrace'
keys = ['durs', 'contour', 'melodic_peaks', 'skips', 'dur_contour', 'interval_class']
match_weights = [0, -6]
gap_penalties = [-5, -5, -4, -4]

xml_root = r'.\parker_transcriptions'
fname = r'parker_transcriptions\1947 03 07 All The Things You Are.mxl'

us = m21.environment.UserSettings()
# us['musescoreDirectPNGPath'] = Path(r"C:\Program Files\MuseScore 3\bin\MuseScore3.exe")

print('pre-processing score...')
mus_xml = m21.converter.parse(fname)
x = list(mus_xml.recurse().getElementsByClass('PageLayout'))
mus_xml.remove(x, recurse=True)
mus_xml_copy = gv.collapse_tied_notes(mus_xml)

feat_dict = gv.get_viewpoints(mus_xml_copy)
vp_seq = gv.viewpoint_seq_from_dict(feat_dict, keys)
# vp_seq = vp_seq[:300]
num_events = len(vp_seq)

print('making markov model...')
markov = mp.make_markov_prob_dicts(xml_root, keys)

def get_subsequences(arr, m):
    n = arr.size - m + 1
    s = arr.itemsize
    return as_strided(arr, shape=(m,n), strides=(s,s))

def get_similarity(sequence_a, sequence_b, vp_seq):

    vp_size = len(vp_seq[0])

    a = [vp_seq[x] for x in sequence_a]
    b = [vp_seq[x] for x in sequence_b]

    score = nw.get_alignment_score(a, b, match_weights, gap_penalties)
    # score = -1 * (score * 2 / (len(a) + len(b)))
    score = -1 * score

    return score

sqs = [list(get_subsequences(np.arange(num_events), num_events - (i - 1))) for i in cardinalities]
sqs = [item for sublist in sqs for item in sublist]
sqs = sorted(sqs, key=lambda x: tuple(x))

# make list of pairs of sequences that will be compared.
print('getting list of candidate pairs of sequences...')
pairs_to_compare = []
for i, sqa in enumerate(sqs):
    for j, sqb in enumerate(sqs[i:]):
        # do not evaluate overlapping sequences
        # if not (set(sqa).intersection(set(sqb)) == set()):
        #     continue
        # do not evaluate sequences whose lengths are too different
        # if np.abs(len(sqa) - len(sqb)) > max_length_difference:
        #     continue
        # evaluate only sequences that are close together
        dist_1 = np.abs(max(sqa) - min(sqb))
        dist_2 = np.abs(max(sqb) - min(sqa))
        if not (dist_1 < seq_compare_dist_threshold or dist_2 < seq_compare_dist_threshold):
            continue
        pairs_to_compare.append((i, j + i))

# compute all similarity scores for selected pairs of sequences
print('computing similarity scores for all pairs of sequences...')
scores = []
dist_matrix = np.zeros((len(sqs), len(sqs))) + 10e5
np.fill_diagonal(dist_matrix, 0)
# dist_matrix = lil_matrix((len(sqs), len(sqs)))
for n, sq_pair in enumerate(pairs_to_compare):
    i, j = sq_pair
    similarity = get_similarity(sqs[i], sqs[j], vp_seq)
    scores.append(similarity)
    dist_matrix[i, j] = similarity
    dist_matrix[j, i] = similarity

    if not n % (len(pairs_to_compare) // 10):
        print(f'   {n} / {len(pairs_to_compare)} scores calculated...')

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

# get thresholds
sorted_scores = sorted(scores, reverse=True)
# dist_thresh = sorted_scores[int(prop_edges_to_keep * len(scores))]

clustering = DBSCAN(eps=50, min_samples=3, metric='precomputed')
clustering.fit(dist_matrix)
# print((clustering.labels_), max(clustering.labels_))

num_clusters = max(clustering.labels_)

cluster_plateaus = []
for c in range(num_clusters):
    sqs_inds_in_cluster = np.nonzero(clustering.labels_ == c)[0]
    core_seq_num = clustering.core_sample_indices_[c]
    core_seq_indices = sqs[core_seq_num]
    core_seq = [vp_seq[x] for x in core_seq_indices]
    core_seq_prob = mp.get_prob_of_sequence(core_seq, markov)
    # print(f'pat {c}, cardinality {len(sqs_inds_in_cluster)}, core_length {len(core_seq)}, prob {core_seq_prob}')
    # print(core_seq_indices)

    cnt = Counter()
    for x in sqs_inds_in_cluster:
        cnt.update(sqs[x])
    print(cnt)

    x = np.zeros(num_events)
    for k in cnt.keys():
        x[k] = cnt[k]
    cluster_plateaus.append(x)
cluster_plateaus = np.stack(cluster_plateaus, 0).T

# plt.plot(cluster_plateaus)
# plt.show()

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

# # print(f"exporting top motifs")
# # for i, path in enumerate(patterns):
# #     viz_seqs = [list(sqs[i]) for i in path[1]]
# #     occs = len(viz_seqs)
# #     cardinality = len(viz_seqs[0])
# #     fname = f'./exports/{tune_name}_developing_motif-{i} freq-{occs} card-{cardinality}'
# #     viz_score = vm.vis_developing_motif(viz_seqs, mus_xml)
# #     viz_score.write('musicxml.pdf', fp=fname)

# #     with open(f"{fname} description.txt", "a") as f:
# #         # f.write(f'Path score = {path[0]:.3f}\n')
# #         for j, seq in enumerate(viz_seqs):
# #             f.write(f'Occurrence {j}: notes {str(seq)}\n')
# #             for k, idx in enumerate(seq):
# #                 vps = str(main_feat_seq[idx]).replace('\'', r'').replace('),', ')').replace(',', ':')
# #                 f.write(f'    Note {idx}: {vps} \n')


# def plot_xy_graph(g):
#     from graph.random import random_xy_graph
#     from graph.hash import graph_hash
#     from graph.visuals import plot_2d

#     node_count = len(g.nodes())
#     xygraph = Graph()

#     for i, node in enumerate(g.nodes()):
#         x = (i * 12)
#         y = np.random.randint(0, node_count * 2)
#         xygraph.add_node((x, y))
        
#     mapping = {b:a for a,b in zip(xygraph.nodes(), g.nodes())}
#     for edge in g.edges():
#         start, end, distance = edge
#         xygraph.add_edge(mapping[start], mapping[end], distance)
#     plt = plot_2d(xygraph)
#     plt.show()

# # mus_xml.write('musicxml.pdf', fp='./exports/testexport')


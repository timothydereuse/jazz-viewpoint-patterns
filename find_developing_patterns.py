import music21 as m21
import numpy as np
from collections import Counter, defaultdict
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
from graph import Graph
from fastdtw import fastdtw
from numpy.lib.stride_tricks import as_strided

import visualize_motifs as vm
import generate_viewpoints as gv
import pattern
from importlib import reload
reload(vm)
reload(gv)
reload(pattern)

cardinalities = [4, 5, 6, 7]
min_occurrences = 4
prop_edges_to_keep = 0.05
num_paths_to_check = 200
seq_compare_dist_threshold = 8
tune_name = 'meshigene'

us = m21.environment.UserSettings()
# us['musescoreDirectPNGPath'] = Path(r"C:\Program Files\MuseScore 3\bin\MuseScore3.exe")

# fname = r'Meshigene - transcription sax solo.musicxml'
fname = r'Falling Grace solo.musicxml'

mus_xml = m21.converter.parse(fname)
x = list(mus_xml.recurse().getElementsByClass('PageLayout'))
mus_xml.remove(x, recurse=True)

print('pre-processing score...')
mus_xml_copy = gv.collapse_tied_notes(mus_xml)
main_feat_seq = gv.get_viewpoints(mus_xml_copy)
feats_arr, feat_probs = gv.viewpoint_seq_to_array(main_feat_seq, prob_exponent=0.5)

num_events = len(main_feat_seq)
note_self_similarity = np.matmul(feats_arr, feats_arr.T)
note_self_similarity[np.identity(num_events) == 1] = 0


def get_subsequences(arr, m):
    n = arr.size - m + 1
    s = arr.itemsize
    return as_strided(arr, shape=(m,n), strides=(s,s))


def get_similarity(sequence_a, sequence_b, feats_arr):
    sq_a = feats_arr[sequence_a]
    sq_b = feats_arr[sequence_b]
    distance, path = fastdtw(sq_a, sq_b)
    distance = distance / (min(len(sq_a), len(sq_b)))
    return distance


sqs = [list(get_subsequences(np.arange(num_events), num_events - (i - 1))) for i in cardinalities]
sqs = [item for sublist in sqs for item in sublist]
sqs = sorted(sqs, key=lambda x: tuple(x))


# make list of pairs of sequences that will be compared.
print('getting list of candidate pairs of sequences...')
pairs_to_compare = []
for i, sqa in enumerate(sqs):
    for j, sqb in enumerate(sqs[i:]):
        # do not evaluate overlapping sequences
        if not (set(sqa).intersection(set(sqb)) == set()):
            continue
        # evaluate only sequences that are close together
        dist_1 = np.abs(max(sqa) - min(sqb))
        dist_2 = np.abs(max(sqb) - min(sqa))
        if not (dist_1 < seq_compare_dist_threshold or dist_2 < seq_compare_dist_threshold):
            continue
        pairs_to_compare.append((i, j + i))



# compute all similarity scores for selected pairs of sequences
print('computing similarity scores for all pairs of sequences...')
scores = []
for n, sq_pair in enumerate(pairs_to_compare):
    i, j = sq_pair
    similarity = get_similarity(sqs[i], sqs[j], feats_arr)
    scores.append(similarity)

    if not n % (len(pairs_to_compare) // 10):
        print(f'   {n} / {len(pairs_to_compare)} scores calculated...')

# get threshold
dist_thresh = sorted(scores)[int(prop_edges_to_keep * len(scores))]

print('building graph...')
g = Graph()
for i, sq in enumerate(sqs):
    g.add_node(i)

for i, sq_pair in enumerate(pairs_to_compare):
    score = scores[i]
    if score < dist_thresh:
        g.add_edge(sq_pair[0], sq_pair[1], score)
        # g.add_edge(j, i, score)

    if not i % (len(pairs_to_compare) // 10):
        print(f'   {i} / {len(pairs_to_compare)} nodes... \n'
              f'   {len(g.edges())} total edges added')

orphaned_nodes = set(g.nodes(out_degree=0)).intersection(g.nodes(in_degree=0))
print(f'removing {len(orphaned_nodes)} orphaned nodes from graph.')
for x in orphaned_nodes:
    g.del_node(x)

print('finding all paths..')
all_paths = g.all_pairs_shortest_paths()

duples = {}
for x in all_paths.keys():
    for y in all_paths[x].keys():
        if x == y or all_paths[x][y] == np.inf:
            continue
        duples[(x, y)] = all_paths[x][y]

x = sorted(duples.keys(), key=lambda x: duples[x])

good_paths = []
for i in range(num_paths_to_check):
    p = g.shortest_path(x[i][0], x[i][1], memoize=True)

    print(p)

    if len(p[1]) < min_occurrences:
        continue

    # check to make sure this isn't too similar to any other path
    similar_paths = [
        g for g in good_paths
        if len(set(g[1]).intersection(set(p[1]))) > len(p[1]) * 0.5
        ]
    if not similar_paths:
        good_paths.append(p)
        continue

print(f"exporting top motifs")
for i, path in enumerate(good_paths):
    viz_seqs = [sqs[i] for i in path[1]]
    occs = len(viz_seqs)
    cardinality = len(viz_seqs[0])
    fname = f'./exports/{tune_name}_developing_motif-{i} freq-{occs} card-{cardinality}'
    viz_score = vm.vis_developing_motif(viz_seqs, mus_xml)
    viz_score.write('musicxml.pdf', fp=fname)

    with open(f"{fname} description.txt", "a") as f:
        f.write(f'Path score = {path[0]:.3f}\n')
        for j, seq in enumerate(viz_seqs):
            f.write(f'Occurrence {j}: notes {str(seq)}\n')
            for k, idx in enumerate(seq):
                vps = str(main_feat_seq[idx]).replace('\'', r'').replace('),', ')').replace(',', ':')
                f.write(f'    Note {idx}: {vps} \n')


def plot_xy_graph(g):
    from graph.random import random_xy_graph
    from graph.hash import graph_hash
    from graph.visuals import plot_2d

    node_count = len(g.nodes())
    xygraph = Graph()

    for i, node in enumerate(g.nodes()):
        x = (i * 12)
        y = np.random.randint(0, node_count * 2)
        xygraph.add_node((x, y))
        

    mapping = {b:a for a,b in zip(xygraph.nodes(), g.nodes())}

    for edge in g.edges():
        start, end, distance = edge
        xygraph.add_edge(mapping[start], mapping[end], distance)
    plt = plot_2d(xygraph)
    plt.show()

# mus_xml.write('musicxml.pdf', fp='./exports/testexport')


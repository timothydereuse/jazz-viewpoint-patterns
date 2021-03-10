import music21 as m21
import numpy as np
from collections import Counter, defaultdict
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
from graph import Graph
from numpy.lib.stride_tricks import as_strided

import visualize_motifs as vm
import generate_viewpoints as gv
import pattern
from importlib import reload
reload(vm)
reload(gv)
reload(pattern)

us = m21.environment.UserSettings()
# us['musescoreDirectPNGPath'] = Path(r"C:\Program Files\MuseScore 3\bin\MuseScore3.exe")

fname = r'./Falling Grace solo.musicxml'
mus_xml = m21.converter.parse(fname)
# mus_xml = gv.collapse_rests(mus_xml)

# mus_xml = mus_xml.makeNotation()

x = list(mus_xml.recurse().getElementsByClass('PageLayout'))
mus_xml.remove(x, recurse=True)

print('pre-processing score...')
mus_xml_copy = gv.collapse_tied_notes(mus_xml)
main_feat_seq = gv.get_viewpoints(mus_xml_copy)
feats_arr, feat_probs = gv.viewpoint_seq_to_array(main_feat_seq, prob_exponent=0.5)

num_events = len(main_feat_seq)
note_self_similarity = np.matmul(feats_arr, feats_arr.T)
note_self_similarity[np.identity(num_events) == 1] = 0

similarity_thresh = np.mean(note_self_similarity) * 6

def get_subsequences(arr, m):
    n = arr.size - m + 1
    s = arr.itemsize
    return as_strided(arr, shape=(m,n), strides=(s,s))


def get_similarity(sequence_a, sequence_b, note_self_similarity):

    res = 0

    # return 0 if these subsequences overlap
    if not (set(sequence_a).intersection(set(sequence_b)) == set()):
        return 0

    # return 0 if sufficiently different in length
    if np.abs(len(sequence_a) - len(sequence_b)) > 1:
        return 0
    elif np.abs(len(sequence_a) - len(sequence_b)) == 1:
        longer_seq = max(sequence_a, sequence_b, key=len)
        shorter_seq = min(sequence_a, sequence_b, key=len)
        # exclude each element of the longer sequence in turn

        scores = []
        for i in range(len(longer_seq)):
            ss = get_similarity(shorter_seq, np.delete(longer_seq, i), note_self_similarity)
            scores.append(ss)

        correction_term = len(shorter_seq) / len(longer_seq)
        return max(scores) * correction_term

    for i in range(len(sequence_a)):
        idx_a = sequence_a[i]
        idx_b = sequence_b[i]
        res += note_self_similarity[idx_a, idx_b]
    res = res / len(sequence_a)
    return res


print('building graph...')
sqs = [list(get_subsequences(np.arange(num_events), num_events - (i - 1))) for i in (5, 6, 7)]
sqs = [item for sublist in sqs for item in sublist]
sqs = sorted(sqs, key=lambda x: tuple(x))

g = Graph()

for i, sq in enumerate(sqs):
    g.add_node(i)

    for n in g.nodes():
        if i == n:
            continue

        similarity = get_similarity(sqs[i], sqs[n], note_self_similarity)

        if similarity > similarity_thresh:
            g.add_edge(n, i, -1 * similarity)
            # print(i, n, similarity)

    if not i % (len(sqs) // 20):
        print(f'   {i} of {len(sqs)} nodes... \n'
              f'   {len(g.edges())} edges added')

orphaned_nodes = set(g.nodes(out_degree=0)).intersection(g.nodes(in_degree=0))
print(f'removing {len(orphaned_nodes)} orphaned nodes from graph.')
for x in orphaned_nodes:
    g.del_node(x)

print('finding all paths..')
all_paths = g.all_pairs_shortest_paths()

duples = {}
for x in all_paths.keys():
    for y in all_paths[x].keys():
        if all_paths[x][y] == np.inf:
            continue
        duples[(x, y)] = all_paths[x][y]

x = sorted(duples.keys(), key=lambda x: duples[x])

print(f"exporting top motifs")
for i in range(20):
    path = g.shortest_path(x[i][0], x[i][1])
    viz_seqs = [sqs[i] for i in path[1]]
    occs = len(viz_seqs)
    cardinality = len(viz_seqs[0])
    fname = f'./exports/fallinggrace_discovered_motif_{i} freq-{occs} card-{cardinality}'
    viz_score = vm.vis_developing_motif(viz_seqs, mus_xml)
    viz_score.write('musicxml.pdf', fp=fname)

# mus_xml.write('musicxml.pdf', fp='./exports/testexport')

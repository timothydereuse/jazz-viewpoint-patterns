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

fname = "Meshigene - transcription sax solo.musicxml"
mus_xml = m21.converter.parse(fname)
x = list(mus_xml.recurse().getElementsByClass('PageLayout'))
mus_xml.remove(x, recurse=True)

print('pre-processing musicxml...')
mus_xml_copy = gv.collapse_tied_notes(mus_xml)
main_feat_seq = gv.get_viewpoints(mus_xml_copy)
feats_arr, feat_probs = gv.viewpoint_seq_to_array(main_feat_seq, prob_exponent=0.5)

num_events = len(main_feat_seq)
note_self_similarity = np.matmul(feats_arr, feats_arr.T)
note_self_similarity[np.identity(num_events) == 1] = 0

similarity_thresh = np.mean(note_self_similarity) * 5

def get_subsequences(arr, m):
    n = arr.size - m + 1
    s = arr.itemsize
    return as_strided(arr, shape=(m,n), strides=(s,s))

def get_similarity(a, b, sqs, note_self_similarity):
    res = 0

    sequence_a = sqs[a]
    sequence_b = sqs[b]

    # return 0 if these subsequences overlap
    if not (set(sequence_a).intersection(set(sequence_b)) == set()):
        return 0

    for i in range(len(sqs[a])):
        idx_a = sequence_a[i]
        idx_b = sequence_b[i]
        res += note_self_similarity[idx_a, idx_b]
    res = res / max(len(sqs[a]), len(sqs[b]))
    return res


print('building graph...')
sqs = get_subsequences(np.arange(num_events), num_events - (5 - 1))
g = Graph()

for i, sq in enumerate(sqs):
    g.add_node(i)

    for n in g.nodes():
        if i == n:
            continue

        similarity = get_similarity(i, n, sqs, note_self_similarity)

        if similarity > similarity_thresh:
            g.add_edge(n, i, -1 * similarity)
            # print(i, n, similarity)

print('finding all paths..')
all_paths = g.all_pairs_shortest_paths()

duples = {}
for x in all_paths.keys():
    for y in all_paths[x].keys():
        if all_paths[x][y] == np.inf:
            continue
        duples[(x, y)] = all_paths[x][y]

x = sorted(duples.keys(), key=lambda x: duples[x])

g.shortest_path(x[0][0], x[0][1])

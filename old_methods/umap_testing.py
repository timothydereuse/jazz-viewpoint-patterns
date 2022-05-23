from matplotlib import markers
import music21 as m21
import numpy as np
import os
import markov_probs as mp
import generate_viewpoints as gv
import umap
import matplotlib.pyplot as plt
import sklearn.decomposition as decomp
import sklearn.preprocessing as preproc
from numpy.lib.stride_tricks import as_strided


cardinalities = [4, 5, 6, 7]
target_length = 50
min_occurrences = 4
score_prop_thresh = 0.006
seq_compare_dist_threshold = 300
log_lh_cutoff = 1.7
top_motifs = 15
keys = ['durs', 'pitches']
match_weights = [0, -6]
gap_penalties = [-5, -5, -4, -4]

xml_root = r'.\parker_transcriptions'
fnames = [
    # 'parker_transcriptions\\1947 02 01 Home Cookin\' 2 YouTube.mxl',
    # r'parker_transcriptions\1947 03 09 Ornithology III.mxl',
    # r'parker_transcriptions\1946 01 28 I Can t Get Started.mxl',
    # r'parker_transcriptions\1947 05 08 Donna Lee V.mxl',
    # r'parker_transcriptions\1947 05 08 Donna Lee IV.mxl',
    # r'parker_transcriptions\1947 05 08 Donna Lee III.mxl',
    r'parker_transcriptions\1947 03 07 All The Things You Are.mxl',
    # r'parker_transcriptions\1946 01 28 Oh Lady Be Good.mxl',
    # r'parker_transcriptions\1945 11 26 Koko take 2.mxl',
    # r'parker_transcriptions\1945 11 26 Koko take 1.mxl',
    ]

us = m21.environment.UserSettings()
# us['musescoreDirectPNGPath'] = Path(r"C:\Program Files\MuseScore 3\bin\MuseScore3.exe")
print('making markov model...')
markov = mp.make_markov_prob_dicts(xml_root, keys)

all_sqs = []
all_mus_xmls = []
all_sequences = []
piece_indices = []
all_sqs_probs = []
for n, fname in enumerate(fnames):
    tune_name = os.path.split(fname)[-1].split('.m')[0]
    print(f'now processing {tune_name}')

    print('pre-processing score...')
    mus_xml = m21.converter.parse(fname)
    x = list(mus_xml.recurse().getElementsByClass('PageLayout'))
    mus_xml.remove(x, recurse=True)
    mus_xml_copy = gv.collapse_tied_notes(mus_xml)
    all_mus_xmls.append(gv.clean_mus_xml(mus_xml_copy))

    main_feat_seq = gv.get_viewpoints(mus_xml_copy)
    num_events = len(main_feat_seq)
    sqs = gv.get_all_subsequences(num_events, cardinalities)
    timescaled = gv.get_timescaled_signal(main_feat_seq, sqs, target_length=target_length)
    all_sequences.extend(timescaled)
    piece_indices.extend(np.ones(num_events) * n)

    vp_seq = gv.viewpoint_seq_from_dict(main_feat_seq, keys)
    sqs_probs = np.zeros(len(sqs))
    for i, sq in enumerate(sqs):
        vps = [vp_seq[i] for i in sq]
        prob = mp.get_prob_of_sequence(vps, markov)
        sqs_probs[i] = prob
    all_sqs_probs = np.concatenate([all_sqs_probs, sqs_probs])
    all_sqs.extend([n, sq] for sq in sqs)

scaled_probs = (all_sqs_probs - min(all_sqs_probs)) ** 2

all_sequences = np.array(all_sequences).reshape(len(all_sequences), -1)
reducer = umap.UMAP(random_state=42, n_components=3, min_dist=0.01, n_neighbors=10)
reducer.fit(all_sequences)
embedding = reducer.transform(all_sequences)

# plt.scatter(embedding[:, 0], embedding[:, 1], s=4)
# # plt.plot(embedding_single[:, 0], embedding_single[:, 1], linewidth=1)
# plt.gca().set_aspect('equal', 'datalim')
# plt.show()

import plotly.graph_objects as go

x, y, z = embedding[:, 0], embedding[:, 1], embedding[:, 2]

texts = []
for n, sqs in enumerate(all_sqs):
    xml_ind, note_inds = sqs
    notes = [all_mus_xmls[xml_ind][i] for i in note_inds]

    p = ' '.join([f'{note.pitch.name}{note.pitch.octave}' for note in notes])
    d = ' '.join([note.duration.type for note in notes])
    texts.append(p + "\n" + d)

fig = go.Figure(data=[
    go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=4,
            color=scaled_probs,
            colorscale='Viridis'
        ),
        text=texts
)])

fig.show()
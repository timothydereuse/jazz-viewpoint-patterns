import music21 as m21
import numpy as np
import os
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

cardinalities = [5, 6, 7]
max_length_difference = 2
min_occurrences = 3
score_prop_thresh = 0.005
seq_compare_dist_threshold = 200
log_lh_cutoff = 1.7
top_motifs = 15
tune_name = 'Donna Lee V'
keys = ['durs', 'rough_contour']
match_weights = [0, -6]
gap_penalties = [-5, -5, -4, -4]

xml_root = r'.\parker_transcriptions'
fname = 'parker_transcriptions\\1947 02 01 Home Cookin\' 2 YouTube.mxl'
# fname = r'parker_transcriptions\1947 03 09 Ornithology III.mxl'
# fname = r'parker_transcriptions\1946 01 28 I Can t Get Started.mxl'
# fname = r'parker_transcriptions\Meshigene - transcription sax solo.musicxml'
# fname = r'parker_transcriptions\Falling Grace solo.musicxml'
fname = r'parker_transcriptions\1947 05 08 Donna Lee V.mxl'

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

    a = [np.array(vp_seq[x]) for x in sequence_a]
    b = [np.array(vp_seq[x]) for x in sequence_b]

    score = nw.get_alignment_score(a, b, match_weights, gap_penalties)
    # score = -1 * (score * 2 / (len(a) + len(b)))
    score = -1 * score

    return score

sqs = [list(get_subsequences(np.arange(num_events), num_events - (i - 1))) for i in cardinalities]
sqs = [item for sublist in sqs for item in sublist]
sqs = sorted(sqs, key=lambda x: tuple(x))

# calculating negative log likelihood of all subsequences...
sqs_probs = np.zeros(len(sqs))
for i, sq in enumerate(sqs):
    vps = [vp_seq[x] for x in sq]
    prob = mp.get_prob_of_sequence(vps, markov)
    sqs_probs[i] = prob

# make list of pairs of sequences that will be compared.
print('getting list of candidate pairs of sequences...')
pairs_to_compare = []
for i, sqa in enumerate(sqs):
    for j, sqb in enumerate(sqs[i:]):
        # do not evaluate overlapping sequences
        # if not (set(sqa).intersection(set(sqb)) == set()):
        #     continue
        # do not evaluate sequences whose lengths are too different
        if np.abs(len(sqa) - len(sqb)) > max_length_difference:
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

# get distance threshold for motif-finding
# thresh = sort_scores[int(len(sort_scores) * score_prop_thresh)]

# calculate coverage of neighborhood of every sequence
print('calculating coverage of every neighborhood...')

def calculate_coverage(dist_matrix, sqs, thresh, min_occurrences=min_occurrences):

    disqualified = np.zeros(dist_matrix.shape[0])
    neighborhoods = (dist_matrix < thresh)
    coverage = np.zeros(dist_matrix.shape[0])
    for i, n in enumerate(neighborhoods):
        positions_in_neighborhood = [sqs[x] for x in n.nonzero()[0]]
        coverage_amt = (set(np.concatenate(positions_in_neighborhood)))
        coverage[i] = len(coverage_amt)

    # select set of motifs that cover as much ground as possible using greedy approach
    # disqualify any that are too close (in distance) to existing motif
    motif_inds = []
    for x in range(100):
        new_motif_ind = np.argmax(coverage)
        motif_inds.append(new_motif_ind)
        inds_to_disqualify = dist_matrix[new_motif_ind] < (2 * thresh)
        disqualified[inds_to_disqualify] = 1
        coverage[disqualified == 1] = 0

        # finish if the highest-coverage entry left has particularly low coverage
        if max(coverage) < (np.mean(cardinalities) * min_occurrences):
            break
    
    return motif_inds

print('selecting motifs that have high-coverage neighborhoods...')
sort_scores = sorted(scores)
thresh = sort_scores[int(len(sort_scores) * score_prop_thresh)]
thresh = max(thresh, min([x for x in sort_scores if x > 0]))
motif_inds = calculate_coverage(dist_matrix, sqs, thresh)
# scores_to_test = np.linspace(score_prop_thresh / 10, score_prop_thresh, 20)
# threshes = [sort_scores[int(len(sort_scores) * x)] for x in scores_to_test]
# threshes = [x for x in threshes if x > 0]
# threshes = sorted(list(set(threshes)))
# all_motif_inds = [(x, calculate_coverage(dist_matrix, sqs, x)) for x in threshes]
# all_motif_inds_probs = [np.mean(sqs_psrobs[np.array(x[1])]) for x in all_motif_inds]
# all_motif_inds_probs = [len(x[1]) for x in all_motif_inds]
# thresh, motif_inds = all_motif_inds[np.argmax(all_motif_inds_probs)]

# mark all sequences within thresh distance of discovered motifs
print('marking and filtering crowded neighborhoods...')
motif_labels = np.zeros(dist_matrix.shape[0], dtype=int) - 1
for i, mi in enumerate(motif_inds):
    close_seqs_inds = (dist_matrix[mi] < thresh).nonzero()[0]
    inds_to_remove = []

    for j, m in enumerate(close_seqs_inds):
        for k, n in enumerate(close_seqs_inds):
            pos_m = list(sqs[m])
            pos_n = list(sqs[n])

            if not bool(set(pos_m).intersection(set(pos_n))):
                # if they don't overlap, leave them both in for now
                continue
            elif j == k:
                continue

            if len(pos_n) > len(pos_m):
                # if one of these sequences overlaps with another, and one is smaller, remove the smaller one
                inds_to_remove.append(j)
                break
            elif len(pos_n) == len(pos_m) and (dist_matrix[mi][n] < dist_matrix[mi][m]):
                # if they overlap and have the same length, choose the one that's closest to the "center"
                inds_to_remove.append(j)
                break
            elif len(pos_n) == len(pos_m) and (dist_matrix[mi][n] == dist_matrix[mi][m]) and (j < k):
                # if they overlap and have the same length and are equally close, choose the first one
                inds_to_remove.append(j)
                break

    print(f'removing {len(inds_to_remove)} motifs out of {len(close_seqs_inds)}.')
    remove_mask = np.ones(len(close_seqs_inds), dtype=bool)
    remove_mask[inds_to_remove] = False
    close_seqs_inds_filtered = close_seqs_inds[remove_mask]

    motif_labels[close_seqs_inds_filtered] = i

motif_probs = []
for c, ind in enumerate(motif_inds):
    vps = [vp_seq[x] for x in sqs[ind]]
    prob = mp.get_prob_of_sequence(vps, markov)
    sqs_inds_in_cluster = np.nonzero(motif_labels == c)[0]

    # remove anything that's dropped below requisite number of occurrences
    if len(sqs_inds_in_cluster) >= min_occurrences:
        motif_probs.append([prob, c, len(sqs_inds_in_cluster)])

motif_probs = sorted(motif_probs, key=lambda x: x[0], reverse=True)
# motifs_to_export = [x[1] for x in motif_probs if x[0] > log_lh_cutoff]
motifs_to_export = [x[1] for x in motif_probs[:top_motifs]]

num_clusters = max(motif_labels)

print(f"exporting top motifs")
folder_name = f'./exports/{tune_name}, ' + '-'.join(keys)
os.mkdir(folder_name)
for i, c in enumerate(motifs_to_export):

    sqs_inds_in_cluster = np.nonzero(motif_labels == c)[0]
    viz_seqs = [list(sqs[i]) for i in sqs_inds_in_cluster]
    occs = len(viz_seqs)
    cardinality = len(viz_seqs[0])
    fname = f'{folder_name}/{tune_name}_developing_motif-{c} freq-{occs} card-{cardinality}'
    viz_score = vm.vis_developing_motif(viz_seqs, mus_xml)
    viz_score.write('musicxml.pdf', fp=fname)

    with open(f"{fname} description.txt", "a") as f:
        f.write(f'Viewpoints: {str(keys)} \n')
        f.write(f'Sequence score = {sqs_probs[motif_inds[c]]:.3f}\n')
        for j, seq in enumerate(viz_seqs):
            f.write(f'Occurrence {j}: notes {str(seq)}\n')
            for k, idx in enumerate(seq):
                vps = str(vp_seq[idx]).replace('\'', r'').replace('),', ')')
                f.write(f'    {vps} \n')


# # mus_xml.write('musicxml.pdf', fp='./exports/testexport')


# remove all sequences not within thresh distance of the discovered motifs from dist matrix
# dist_matrix_edit = np.copy(dist_matrix)
# for ind_to_remove in (motif_labels == -1).nonzero()[0]:
#     dist_matrix_edit[ind_to_remove] = 10e5
#     dist_matrix_edit[:, ind_to_remove] = 10e5

# clustering = try_clusterings(dist_matrix_edit)
# num_clusters = max(clustering.labels_)


# def try_clusterings(dist_matrix, lo=1, hi=30, step=1):
#     eps_to_try = np.arange(lo, hi, step)

#     clusterings = []
#     for eps in eps_to_try:
#         clustering = DBSCAN(eps=eps, min_samples=4, metric='precomputed')
#         clustering.fit(dist_matrix)
#         clusterings.append(clustering)
#         # print(eps, max(clustering.labels_))

#     num_clusters = [max(x.labels_) for x in clusterings]
#     max_ind = np.argmax(num_clusters)

#     return clusterings[max_ind]


# cluster_plateaus = []
# for c in range(num_clusters):
#     # sqs_inds_in_cluster = np.nonzero(clustering.labels_ == c)[0]
#     sqs_inds_in_cluster = np.nonzero(motif_labels == c)[0]
#     core_seq_num = clustering.core_sample_indices_[c]

#     # sqs_inds_in_cluster = np.nonzero(motif_labels == c)[0]
#     # core_seq_num = motif_inds[c]

#     core_seq_indices = sqs[core_seq_num]
#     core_seq = [vp_seq[x] for x in core_seq_indices]
#     core_seq_prob = mp.get_prob_of_sequence(core_seq, markov)
#     # print(f'pat {c}, cardinality {len(sqs_inds_in_cluster)}, core_length {len(core_seq)}, prob {core_seq_prob}')
#     # print(core_seq_indices)

#     cnt = Counter()
#     for x in sqs_inds_in_cluster:
#         cnt.update(sqs[x])
#     # print(cnt)

#     x = np.zeros(num_events)
#     for k in cnt.keys():
#         x[k] = cnt[k]
#     cluster_plateaus.append(x)
# cluster_plateaus = np.stack(cluster_plateaus, 0).T

# plt.plot(cluster_plateaus)
# plt.show()
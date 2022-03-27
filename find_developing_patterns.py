import music21 as m21
import numpy as np
import os
from collections import Counter, defaultdict
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
from itertools import combinations
import affine_needleman_wunsch as nw
from sklearn.cluster import DBSCAN
from joblib import Parallel, delayed
from numpy.lib.stride_tricks import as_strided

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


def get_similarity(sequence_a, sequence_b, vp_seq):
    vp_size = len(vp_seq[0])
    a = [np.array(vp_seq[x]) for x in sequence_a]
    b = [np.array(vp_seq[x]) for x in sequence_b]
    score = nw.get_alignment_score(a, b, match_weights, gap_penalties)
    # score = -1 * (score * 2 / (len(a) + len(b)))
    score = -1 * score
    return score


def calculate_coverage(dist_matrix, sqs, thresh, min_occurrences):

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

# mark all sequences within thresh distance of discovered motifs
def filter_neighborhoods(motif_inds, thresh, dist_matrix, sqs):

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

    return motif_labels

if __name__ == '__main__':

    cardinalities = [5, 6, 7]
    max_length_difference = 5
    min_occurrences = 4
    score_prop_thresh = 0.006
    seq_compare_dist_threshold = 300
    log_lh_cutoff = 1.7
    top_motifs = 15
    keys = ['durs', 'contour', 'pitches', 'sharp_melodic_peaks']
    match_weights = [0, -6]
    gap_penalties = [-5, -5, -4, -4]

    xml_root = r'.\parker_transcriptions'
    fnames = [
        r'Konitz\Konitz - Lennie-Bird.musicxml',
        # 'parker_transcriptions\\1947 02 01 Home Cookin\' 2 YouTube.mxl',
        # r'parker_transcriptions\1947 03 09 Ornithology III.mxl',
        # r'parker_transcriptions\1946 01 28 I Can t Get Started.mxl',
        # r'parker_transcriptions\Meshigene - transcription sax solo.musicxml',
        # r'parker_transcriptions\Falling Grace solo.musicxml',
        # r'parker_transcriptions\1947 05 08 Donna Lee V.mxl',
        ]

    us = m21.environment.UserSettings()
    # us['musescoreDirectPNGPath'] = Path(r"C:\Program Files\MuseScore 3\bin\MuseScore3.exe")
    print('making markov model...')
    markov = mp.make_markov_prob_dicts(xml_root, keys)


    for fname in fnames:
        tune_name = os.path.split(fname)[-1].split('.m')[0]
        print(f'now processing {tune_name}')

        print('pre-processing score...')
        mus_xml = m21.converter.parse(fname)
        x = list(mus_xml.recurse().getElementsByClass('PageLayout'))
        mus_xml.remove(x, recurse=True)
        mus_xml_copy = gv.collapse_tied_notes(mus_xml)

        feat_dict = gv.get_viewpoints(mus_xml_copy)
        vp_seq = gv.viewpoint_seq_from_dict(feat_dict, keys)
        num_events = len(vp_seq)

        sqs_unfiltered = gv.get_all_subsequences(num_events, cardinalities)
        sqs = gv.filter_subsequences(feat_dict, sqs_unfiltered)

        # calculating negative log likelihood of all subsequences...
        # sqs_probs = np.zeros(len(sqs))
        # for i, sq in enumerate(sqs):
        #     vps = [vp_seq[x] for x in sq]
        #     prob = mp.get_prob_of_sequence(vps, markov)
        #     sqs_probs[i] = prob

        # make list of pairs of sequences that will be compared.
        print('getting list of candidate pairs of sequences...')
        pairs_to_compare = []
        for i, sqa in enumerate(sqs):
            for j, sqb in enumerate(sqs[i:]):
                # do not evaluate overlapping sequences
                if not (set(sqa).intersection(set(sqb)) == set()):
                    continue
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

        num_pairs = len(pairs_to_compare)
        def compare_pair(sq_pair):
            n, i, j = sq_pair
            similarity = get_similarity(sqs[i], sqs[j], vp_seq)
            if not n % (num_pairs // 100):
                print(f'   {n} / {num_pairs} scores calculated...')
            return similarity

        # for n, sq_pair in enumerate(pairs_to_compare):
        #     i, j = sq_pair
        #     similarity = get_similarity(sqs[i], sqs[j], vp_seq)
        #     scores.append(similarity)
        #     dist_matrix[i, j] = similarity
        #     dist_matrix[j, i] = similarity

        #     if not n % (len(pairs_to_compare) // 10):
        #         print(f'   {n} / {len(pairs_to_compare)} scores calculated...')
        # output = Parallel(n_jobs=3)(delayed(compare_pair)((i,) + n) for i, n in enumerate(pairs_to_compare))
        output = [compare_pair((i,) + n) for i, n in enumerate(pairs_to_compare)]
        for n, sq_pair in enumerate(pairs_to_compare):
            dist_matrix[sq_pair[0], sq_pair[1]] = output[n]
            dist_matrix[sq_pair[1], sq_pair[0]] = output[n]

        # calculate coverage of neighborhood of every sequence
        print('calculating coverage of every neighborhood...')

        print('selecting motifs that have high-coverage neighborhoods...')
        sort_scores = sorted(output)
        thresh = sort_scores[int(len(sort_scores) * score_prop_thresh)]
        thresh = max(thresh, min([x for x in sort_scores if x > 0]))
        motif_inds = calculate_coverage(dist_matrix, sqs, thresh, min_occurrences)

        motif_labels = filter_neighborhoods(motif_inds, thresh, dist_matrix, sqs)

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

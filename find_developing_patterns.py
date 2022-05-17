import music21 as m21
import numpy as np
import os
from collections import Counter, defaultdict
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
from itertools import combinations
import affine_needleman_wunsch as nw
from copy import copy
import visualize_motifs as vm
import generate_viewpoints as gv
import markov_probs as mp
import pattern
from importlib import reload
import csv
reload(vm)
reload(gv)
reload(mp)
reload(nw)
reload(pattern)

m21.environment.set('musescoreDirectPNGPath', r'C://Program Files//MuseScore 3//bin//MuseScore3.exe')
match_weights = [0, -2]
gap_penalties = [-2, -2, -1, -1]

def get_similarity(sequence_a, sequence_b, vp_seq, partial_matches=True):
    vp_size = len(vp_seq[0])
    a = [np.array(vp_seq[x]) for x in sequence_a]
    b = [np.array(vp_seq[x]) for x in sequence_b]
    score = nw.get_alignment_score(a, b, match_weights, gap_penalties, partial_matches)
    score = -1 * (score * 2 / (len(a) + len(b)))
    # score = -1 * score
    return score


def calculate_coverage(dist_matrix, sqs, thresh, min_coverage):

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
        if max(coverage) < min_coverage:
            break
    
    return motif_inds


def filter_neighborhoods(motif_inds, thresh, dist_matrix, sqs):
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

    return motif_labels


def process_mus_xml(fname, markov, cardinalities=None, keys=None, 
    max_length_difference=2, min_occurrences=4, score_prop_thresh=1.0, max_score=2, partial_matches=True,
    markov_prob_thresh=0.6, seq_compare_dist_threshold=500, precomputed_pairs_output=None):
    
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
    sqs = gv.filter_subsequences(feat_dict, sqs_unfiltered, weak_start_end=True)

    # calculating negative log likelihood of all subsequences...
    sqs_probs = np.zeros(len(sqs))
    for i, sq in enumerate(sqs):
        vps = [vp_seq[x] for x in sq]
        prob = mp.get_prob_of_sequence(vps, markov)
        sqs_probs[i] = prob

    prob_thresh = sorted(sqs_probs)[int((1 - markov_prob_thresh) * len(sqs_probs))]
    sqs = [sqs[i] for i in range(len(sqs)) if sqs_probs[i] > prob_thresh]

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
    dist_matrix = np.zeros((len(sqs), len(sqs))) + 10e5
    np.fill_diagonal(dist_matrix, 0)
    num_pairs = len(pairs_to_compare)

    def compare_pair(sq_pair):
        n, i, j = sq_pair
        similarity = get_similarity(sqs[i], sqs[j], vp_seq, partial_matches)
        if not n % (num_pairs // 10):
            print(f'   {n} / {num_pairs} scores calculated...')
        return similarity

    if precomputed_pairs_output is None:
        output = [compare_pair((i,) + n) for i, n in enumerate(pairs_to_compare)]
    else:
        print('using precomputed pair scores.')
        output = precomputed_pairs_output

    for n, sq_pair in enumerate(pairs_to_compare):
        dist_matrix[sq_pair[0], sq_pair[1]] = output[n]
        dist_matrix[sq_pair[1], sq_pair[0]] = output[n]

    # calculate coverage of neighborhood of every sequence
    
    sort_scores = sorted(output)
    thresh = sort_scores[int(len(sort_scores) * score_prop_thresh) - 1] 

    # scores are multiplied by this factor in sequence alignment to keep everything an integer. divide by this
    # again for showing to the user
    factor = np.abs(match_weights[1])
    thresh = min(thresh, max_score * factor)
    print(
        f'effective score threshold: {thresh / factor}\n'
        f'median score:{np.mean(output) / factor}'
        f'max: {np.max(output) / factor}'
        f'min: {np.min(output) / factor}'
    )

    print('calculating coverage of the neighborhood of every sequence...')
    motif_inds = calculate_coverage(dist_matrix, sqs, thresh, min_coverage=(np.min(cardinalities) * min_occurrences))

    print('selecting motifs from high-coverage neighborhoods...')
    motif_labels = filter_neighborhoods(motif_inds, thresh, dist_matrix, sqs)

    motif_probs = []
    all_covered_notes = set()
    for c, ind in enumerate(motif_inds):
        vps = [vp_seq[x] for x in sqs[ind]]
        sqs_inds_in_cluster = np.nonzero(motif_labels == c)[0]

        # don't add anything that's dropped below requisite number of occurrences
        if len(sqs_inds_in_cluster) < min_occurrences:
            continue

        raw_inds = [sqs[i] for i in sqs_inds_in_cluster]
        all_pairwise_distances = [dist_matrix[pair[0], pair[1]] for pair in combinations(sqs_inds_in_cluster, 2)]
        pairwise_similarity = np.median(all_pairwise_distances)
        center_similarity = np.median([dist_matrix[ind, x] for x in sqs_inds_in_cluster])
        covered_notes = set(np.concatenate(raw_inds))
        all_covered_notes.update(covered_notes)

        motif_probs.append({
            'coverage': float(len(covered_notes)) / float(len(vp_seq)),
            'pairwise_similarity': pairwise_similarity,
            'center_similarity': center_similarity,
            'prototype_ind': ind,
            'prob_score': mp.get_prob_of_sequence(vps, markov),
            'sq_inds': sqs_inds_in_cluster,
            'note_inds': raw_inds,
            'mean_cardinality': np.mean([len(x) for x in raw_inds]),
            'num_occurrences': len(sqs_inds_in_cluster)
            })

    motif_probs = sorted(motif_probs, key=lambda x: x['prob_score'], reverse=True)

    motif_summary_dict = {
        'motifs': motif_probs, 
        'num_motifs': len(motif_probs),
        'mean_num_occurences': np.mean([x['num_occurrences'] for x in motif_probs]),
        'mean_mean_cardinality': np.mean([x['mean_cardinality'] for x in motif_probs]),
        'global_coverage': len(all_covered_notes) / len(vp_seq),
        'avg_pairwise_similarity': np.mean([x['pairwise_similarity'] for x in motif_probs]),
        'avg_center_similarity': np.mean([x['center_similarity'] for x in motif_probs]),
    }

    return motif_summary_dict, vp_seq, mus_xml, output

def export_motifs_to_pdf(motifs_to_export, mus_xml, vp_seq, params, tune_name):

    folder_name = os.path.join('./exports', f'{tune_name}, ' + '-'.join(params['keys']))
    os.mkdir(folder_name)
    for i, motif in enumerate(motifs_to_export):

        viz_seqs = motif['note_inds']
        occs = motif['num_occurrences']
        score = motif['prob_score']
        cardinality = np.round(motif['mean_cardinality'], 2)
        fname = os.path.join(f'{folder_name}', f'{tune_name}-{i} freq-{occs} card-{cardinality}')
        viz_score = vm.vis_developing_motif(viz_seqs, mus_xml)
        viz_score.write('musicxml.pdf', fp=str(fname))

        with open(f"{fname} description.txt", "a") as f:
            f.write(f'Params: {str(params)} \n')
            f.write(f'Sequence score = {score:.3f}\n')
            for j, seq in enumerate(viz_seqs):
                f.write(f'Occurrence {j}: notes {str(seq)}\n')
                for k, idx in enumerate(seq):
                    vps = str(vp_seq[idx]).replace('\'', r'').replace('),', ')')
                    f.write(f'    {vps} \n')


if __name__ == '__main__':

    xml_roots = [r'.\parker_transcriptions', r'.\konitz_transcriptions', r'.\other_transcriptions']
    fnames = [
        # r'parker_transcriptions\1945 11 26 Koko Savoy Vol 1.mxl',
        # r'parker_transcriptions\1945 11 26 Koko take 1.mxl',
        # r'parker_transcriptions\1945 11 26 Koko take 2.mxl',
        # r'parker_transcriptions\1947 05 08 Donna Lee V.mxl',
        # r'parker_transcriptions\1947 05 08 Donna Lee IV.mxl',
        # r'parker_transcriptions\1947 05 08 Donna Lee III.mxl',
        # r'parker_transcriptions\1953 02 22 fine And Dandy Washington YouTube.mxl',
        # r'parker_transcriptions\Parker on What is this thing called love (jam session).musicxml',
        # r'konitz_transcriptions\Lee Konitz - Subconscious-lee.musicxml',
        # r'konitz_transcriptions\donna lee - konitz.musicxml',
        # r'konitz_transcriptions\Konitz - Lennie-Bird.musicxml',
        r'konitz_transcriptions\Lee Konitz on Marshmallow.musicxml',
        # r'konitz_transcriptions\Lee Konitz on Sax of a Kind.musicxml',
        # r'konitz_transcriptions\Lee Konitz on Star Eyes.musicxml'
        ]
    us = m21.environment.UserSettings()

    keysets = [
        ['durs', 'melodic_peaks'],
        # ['durs', 'dur_contour'],
        # ['durs', 'dur_contour', 'melodic_peaks'],
        ['durs', 'intervals_semitones'],
        ['durs', 'rough_contour'],
        # ['durs', 'skips'],
        # ['durs', 'sharp_melodic_peaks'],
        # ['durs', 'diatonic_int_size'],
        # ['durs', 'pitches'],
        # ['durs', 'melodic_peaks'],
        # ['durs', 'interval_class'],
        # ['durs', 'melodic_peaks', 'dur_contour'],
        # ['durs', 'rough_contour', 'interval_class'],
        # ['durs', 'rough_contour', 'intervals_semitones'],
        # ['durs', 'rough_contour', 'diatonic_int_size'],
        # ['durs', 'rough_contour', 'pitches'],
        # ['pitches', 'interval_class'],
        # ['pitches', 'melodic_peaks'],
        # ['pitches', 'rough_contour'],
        # ['intervals_semitones', 'diatonic_int_size'],
        # ['intervals_semitones', 'melodic_peaks'],
        # ['intervals_semitones', 'rough_contour'],
    ]

    base_params = {
        'cardinalities': [5, 6, 7, 8],
        'max_length_difference': 2,
        'min_occurrences': 4,
        'score_prop_thresh': 1,
        'max_score': 0.5,
        'markov_prob_thresh': 0.5,
        'seq_compare_dist_threshold': 10000,
        'partial_matches': True,
        'keys': ['durs', 'melodic_peaks']
    }

    # max_scores_to_try = [1/4, 1/3, 1/2, 3/4, 1]
    max_scores_to_try = [1/2, 3/4]

    results = []

    for keyset in keysets:
        print(f'using keyset {keyset}...')
        print('making markov model...')

        params = copy(base_params)
        params['keys'] = keyset
        markov = mp.make_markov_prob_dicts(xml_roots, params['keys'])
        
        for fname in fnames:
            reuse_output = None

            for max_score in max_scores_to_try:
                params['max_score'] = max_score

                motif_summary, vp_seq, mus_xml, opt = process_mus_xml(fname, markov, precomputed_pairs_output=reuse_output, **params)
                reuse_output = opt

                tune_name = fname.split('\\')[-1]

                result = copy(motif_summary)
                for x in params.keys():
                    result[x] = params[x]
                result['tune_name'] = tune_name
                results.append(result)

    header = sorted(list(results[0].keys()))
    header.remove('motifs')

    with open('results.csv', 'w', newline='') as f:
        # create the csv writer
        writer = csv.writer(f)

        # write a row to the csv file
        writer.writerow(header)
        for result in results:
            entries = [result[x] for x in header]
            writer.writerow(entries)

    # export_motifs_to_pdf(motifs_to_export, mus_xml, vp_seq, params, tune_name)
    
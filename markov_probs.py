import music21 as m21
import numpy as np
from collections import Counter, defaultdict
import generate_viewpoints as gv
import os

def counter_to_prob(c):
    c = dict(c)
    s = sum(c.values())
    for k2 in c.keys():
        c[k2] = c[k2] / s
    return c

def make_markov_prob_dicts(xml_roots, keys):

    all_fnames = []
    for xml_root in xml_roots:
        fnames = [ os.path.join(xml_root, x) for x in os.listdir(xml_root) if '.mxl' in x or '.musicxml' in x] 
        all_fnames.extend(fnames)

    all_viewpoints = {}

    for i, full_path in enumerate(all_fnames):
        mus_xml = m21.converter.parse(full_path)

        print(f'making markov model... processing score {i} of {len(all_fnames)}')
        mus_xml_copy = gv.collapse_tied_notes(mus_xml)
        main_feat_dict = gv.get_viewpoints(mus_xml_copy)
        all_viewpoints[full_path] = main_feat_dict

    markov0 = Counter()
    markov1 = defaultdict(Counter)
    for feat_dict in all_viewpoints.values():
        vp_seq = gv.viewpoint_seq_from_dict(feat_dict, keys)
        markov0.update(vp_seq)
        for i in range(1, len(vp_seq)):
            markov1[vp_seq[i - 1]].update([vp_seq[i]])

    # add single event to all keys for smoothing
    for add in markov0.keys():
        for k2 in markov1.keys():
            markov1[k2].update([add])

    # convert all dicts into probability distributions
    markov0 = counter_to_prob(markov0)
    for k in markov1.keys():
        markov1[k] = counter_to_prob(markov1[k])

    markov1['raw'] = markov0

    return markov1

def get_prob_of_sequence(sq, markov_dict):
    cumulative_0_prob = 1
    cumulative_1_prob = 1
    cumulative_prob = 1
    for i in range(len(sq)):
        cur = sq[i]
        zeroth_order_prob = markov_dict['raw'][cur]
        if i > 0:
            prev = sq[i - 1]
            # first_order_prob = markov_dict[prev][cur]
            prob = markov_dict[prev][cur]
        else:
            prob = zeroth_order_prob = markov_dict['raw'][cur]

        # cumulative_0_prob *= zeroth_order_prob
        # cumulative_1_prob *= first_order_prob
        cumulative_prob *= prob

    # total_prob = (np.log(cumulative_0_prob) + np.log(cumulative_1_prob)) * -1
    total_prob = np.log(cumulative_prob) * -1
    # total_prob /= len(sq)

    return total_prob 

if __name__ == '__main__':
    fname = r'parker_transcriptions\\1947 03 09 Ornithology III.mxl'
    xml_root = r'.\parker_transcriptions'
    keys = ['durs']

    markov = make_markov_prob_dicts(xml_root, keys)

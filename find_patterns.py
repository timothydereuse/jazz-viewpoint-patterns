import music21 as m21
import numpy as np
from collections import Counter, defaultdict
from copy import deepcopy
from fractions import Fraction
from pathlib import Path

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

mus_xml_copy = gv.collapse_tied_notes(mus_xml)
# mus_xml_copy = deepcopy(mus_xml)
# # consolidate ties into single notes
# tied_notes = [x for x in mus_xml_copy.flat.notes if x.tie]
# last_not_tied_ind = 0
# for i, n in enumerate(tied_notes):
#     if n.tie.type == 'start':
#         n.tie = None
#         last_not_tied_ind = i
#     else:
#         mus_xml_copy.remove(n, recurse=True)
#         new_dur = m21.duration.Duration(tied_notes[last_not_tied_ind].duration.quarterLength + n.duration.quarterLength)
#         tied_notes[last_not_tied_ind].duration = new_dur

main_feat_seq = gv.get_viewpoints(mus_xml_copy)
c = Counter([item for sublist in main_feat_seq for item in sublist])
num_events = len(main_feat_seq)
feat_probs = {e: (c[e] / num_events) ** (1) for e in c}
common_features = [x for x in c if c[x] > 10]
feature_map = {
    e: np.array([(e in s) for s in main_feat_seq]) for e in c
}

# starting_feature = common_features[rng.choice(len(common_features))]
# pat = Pattern(starting_feature, feature_map[starting_feature])

pat = pattern.Pattern(set(), np.ones(num_events, dtype='bool'))

active_pats = [pat]
finished_pats = defaultdict(lambda: None)
visited_pattern_hashes = defaultdict(lambda: None)
discarded_pats = 0

num_variants = 6
min_freq = 6
min_cardinality = 5

def update_finished_pats(cur_pat, finished_pats):
    id = cur_pat.get_hashable_map()
    existing_pat = finished_pats[id]
    if (not existing_pat) or (existing_pat.interest(feat_probs) < cur_pat.interest(feat_probs)):
        finished_pats[id] = cur_pat

for rand_start in range(30):

    print(f"new start: {rand_start}")
    active_pats = [deepcopy(pat)]

    for iter in range(1000 * 50):
        try:
            first_pat = active_pats[0]
        except IndexError:
            print(f"active_pats empty at iteration {iter}")
            break

        # if you find a pattern that occurs at the exact same places as another pattern
        # then they're equivalent - retain the more interesting one
        equiv_pats = []
        for cmp_pat in active_pats:
            if all(cmp_pat.instance_map == first_pat.instance_map):
                equiv_pats.append(cmp_pat)
        cur_pat = max(equiv_pats, key=lambda x: x.interest(feat_probs))
        for cmp_pat in equiv_pats:
            active_pats.remove(cmp_pat)
        discarded_pats += (len(equiv_pats) - 1)
        # active_pats.remove(first_pat)
        # cur_pat = first_pat

        res = pattern.iterate_pattern(cur_pat, main_feat_seq, feat_probs, feature_map, num_variants, min_freq)
        if (not res) and (len(cur_pat) >= min_cardinality):
            update_finished_pats(cur_pat, finished_pats)
        elif not res:
            discarded_pats += 1
        else:
            active_pats = active_pats + res

        if not iter % 1000:
            avg_len = np.mean([len(x) for x in active_pats])
            print(f'iter {iter} | active_pats {len(active_pats)} | avg_active_length {avg_len:2.3f} | finished_pats {len(finished_pats)}  discarded_pats {discarded_pats}')

interest_pats = [(p, p.interest(feat_probs)) for p in finished_pats.values()]
interest_pats = sorted(interest_pats, key=lambda x: -1 * x[1])

print(f"exporting top motifs")
for i in range(50):
    p = interest_pats[i][0]
    occs = sum(p.instance_map)
    cardinality = len(p.pat) - 1
    fname = f'./exports/meshigene_discovered_motif_{i} freq-{occs} card-{cardinality}'
    viz_score = vm.vis_motif(p, mus_xml)
    viz_score.write('musicxml.pdf', fp=fname)

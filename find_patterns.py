import music21 as m21
import numpy as np
from collections import Counter, defaultdict
from copy import deepcopy
from fractions import Fraction

fname = "Meshigene - transcription sax solo.musicxml"
quarter_beat_multiplier = 12
mus_xml = m21.converter.parse(fname)

vps = {}
# EXTRACT VIEWPOINTS

notes = list(mus_xml.flat.notes)
pitches = [n.pitch.midi for n in notes]
durs = [n.duration.quarterLength * quarter_beat_multiplier for n in notes]
offsets = [n.offset * quarter_beat_multiplier for n in notes]
intervals = [m21.interval.Interval(notes[i], notes[i+1]) for i in range(len(notes) - 1)]

# MIDI pitch
vps['pitches'] = pitches

# pitch class
pc_int = [n % 12 for n in pitches]
vps['pc_int'] = pc_int

# duration
vps['durs'] = durs

# duration contour
dur_contour = np.sign(np.append(np.diff(durs), 0))
vps['dur_contour'] = dur_contour

# duration ratio
# dur_ratio = [durs[i+1] / durs[i] for i in range(len(durs) - 1)]
# dur_ratio = np.round(np.log2(np.array(dur_ratio, dtype='float')))
# vps['dur_ratio'] = np.append(dur_ratio, 0).astype('int')

# down-beats
# down_beats = (np.array(offsets) % (quarter_beat_multiplier * 4)) == 0
# down_beats = [(x if x else np.nan) for x in down_beats]
# vps['down_beats'] = down_beats

# off-beats
off_beats = (np.array(offsets) % (quarter_beat_multiplier * 4))
off_beats = np.gcd(off_beats, 2**10)
off_beats = [(1 if x == 2**10 else x) for x in off_beats]
vps['beat_subdivision'] = off_beats


# time till next note
next_offset_time = np.append(np.diff(offsets), 10)
# vps['next_offset_time'] = next_offset_time

# time till next note minus duration
rest_pad = next_offset_time - durs
rest_pad = [(x if x != 0.0 else np.nan) for x in rest_pad]
vps['rest_pad'] = rest_pad

# interval from prev. note
intervals_semitones = np.append(0, [x.semitones for x in intervals])
vps['intervals_semitones'] = intervals_semitones

# interval class
# interval_class = np.append(0, [x.intervalClass for x in intervals])
# vps['interval_class'] = interval_class

# contour
contour = np.sign(intervals_semitones)
vps['contour'] = contour

# non-stepwise motion
skips = np.append(0, [x.isSkip for x in intervals])
skips = [(x if x != 0.0 else np.nan) for x in skips]
vps['skips'] = skips

# diatonic interval size
diatonic_int_size = np.append(0, [int(x.name[-1]) for x in intervals])
vps['diatonic_int_size'] = diatonic_int_size

# convert to list-of-sets
main_feat_seq = np.array([set() for _ in range(len(notes))])
fkeys = sorted(list(vps.keys()))
for k in fkeys:
    for i in range(len(notes)):
        val = vps[k][i]
        # filter out NaNs
        if val != val:
            continue
        val2 = int(val)
        main_feat_seq[i].add((k, val2))

c = Counter([item for sublist in main_feat_seq for item in sublist])

num_events = len(main_feat_seq)
feature_probability = {e: c[e] / num_events for e in c}
common_features = [x for x in c if c[x] > 10]
feature_map = {
    e: np.array([(e in s) for s in main_feat_seq]) for e in c
}

class Pattern(object):
    def __init__(self, start, instance_map):

        if type(start) == tuple:
            self.pat = [set([start])]
        elif type(start) == set:
            self.pat = [start]
        elif type(start) == list:
            self.pat = start
        else:
            raise TypeError('first arg to Pattern must be tuple, or list of sets of tuples')

        self.instance_map = instance_map

    def __len__(self):
        return len(self.pat)

    def __str__(self):
        s = ''
        for x in self.pat:
            s += str(x) + '\n'
        num_occ = sum(self.instance_map)
        s += f'occurrences: {num_occ}'
        return s

    def __repr__(self):
        return f'Pattern of length {len(self.pat)} + {str(self.pat)}'

    def get_hashable_map(self):
        return tuple(np.nonzero(self.instance_map)[0])

    def prob(self):
        return np.product([feature_probability[cf] for sublist in self.pat for cf in sublist])

    def interest(self):
        return sum(self.instance_map) / self.prob()

rng = np.random.default_rng()
starting_feature = common_features[rng.choice(len(common_features))]
pat = Pattern(starting_feature, feature_map[starting_feature])

pat = Pattern(set(), np.ones(num_events, dtype='bool'))

def iterate_pattern(pat, num_variants=3, min_freq=5):
    # cur_pattern_prob = np.product([feature_probability[cf] for sublist in pat for cf in sublist])
    # cur_pattern_interest = sum(instance_map) / cur_pattern_prob

    # GET CANDIDATE I-STEPS
    # get all features at the end of the current pattern
    current_offset = len(pat) - 1
    offset_instance_map = np.roll(pat.instance_map, current_offset)
    cur_positions = main_feat_seq[offset_instance_map]

    c = Counter([item for sublist in cur_positions for item in sublist])
    candidate_features = list(set([x for x in c if c[x] > min_freq]).difference(pat.pat[-1]))

    # if candidate_features is empty AND the last entry in pat is an empty set, then we have
    # a pattern that cannot be improved upon - return false
    if not candidate_features and not bool(pat.pat[-1]):
        return False

    # calculate interest of all possible extensions
    # interest is observed (from counter) over expected
    interest = [c[cf] / (pat.prob() * feature_probability[cf]) for cf in candidate_features]

    # add possibility of S-STEP which extends pattern
    # only if the last step wasn't also an S-STEP
    if bool(pat.pat[-1]):
        interest.append(pat.interest())
        candidate_features.append(('S-STEP', 0))

    # interest = np.array(interest) ** 2
    interest_distribution = np.array(interest) / sum(interest)

    # choose possible extensions
    num_to_choose = min(num_variants, len(candidate_features))
    indices_to_add = rng.choice(len(candidate_features), num_to_choose, p=interest_distribution, replace=False)

    # make chosen extensions
    new_pats = []
    for ind in indices_to_add:
        new_pat = deepcopy(pat)
        feat_to_add = candidate_features[ind]
        if feat_to_add == ('S-STEP', 0):
            new_pat.pat.append(set())
        else:
            new_pat.pat[-1].add(feat_to_add)
            rolled_feature_map = np.roll(feature_map[feat_to_add], -1 * current_offset)
            new_pat.instance_map = new_pat.instance_map & rolled_feature_map
        new_pats.append(new_pat)

    # print(interest_distribution)
    # print(pat, sum(instance_map))

    return new_pats

active_pats = [pat]
finished_pats = defaultdict(lambda: None)
visited_pattern_hashes = defaultdict(lambda: None)
discarded_pats = 0

num_variants = 5
min_freq = 8
min_cardinality = 6

for iter in range(50000):
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
    cur_pat = max(equiv_pats, key=lambda x: x.interest())
    for cmp_pat in equiv_pats:
        active_pats.remove(cmp_pat)
    discarded_pats += (len(equiv_pats) - 1)
    # active_pats.remove(first_pat)
    # cur_pat = first_pat

    res = iterate_pattern(cur_pat, num_variants, min_freq)
    if (not res) and (len(cur_pat) >= min_cardinality):
        id = cur_pat.get_hashable_map()
        existing_pat = finished_pats[id]
        if (not existing_pat) or (existing_pat.interest() < cur_pat.interest()):
            finished_pats[id] = cur_pat
    elif not res:
        discarded_pats += 1
    else:
        active_pats = res + active_pats

    # if pattern is not dead, test returned patterns to see if we've seen them already
    # for p in res:
    #     pat_hash = p.get_hashable()
    #     if not visited_pattern_hashes[pat_hash]:
    #         active_pats = [p] + active_pats
    #         visited_pattern_hashes[pat_hash] == True
    #     else:
    #         hash_collisions += 1



    if not iter % 1000:
        avg_len = np.mean([len(x) for x in active_pats])
        print(f'iter {iter} | active_pats {len(active_pats)} | avg_active_length {avg_len:2.3f} | finished_pats {len(finished_pats)}  discarded_pats {discarded_pats}')


interest_pats = [(p, p.interest()) for p in finished_pats.values()]
interest_pats = sorted(interest_pats, key=lambda x: -1 * x[1])

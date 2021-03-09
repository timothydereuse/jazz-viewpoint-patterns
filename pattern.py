import numpy as np
from collections import Counter
from copy import deepcopy

rng = np.random.default_rng()

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
        s += f'length: {len(self.pat)}'
        locs = tuple(np.nonzero(self.instance_map))
        num_occ = len(locs)
        s += f'occurrences: {num_occ} at {locs}'
        return s

    def __repr__(self):
        return f'Pattern of length {len(self.pat)}: {str(self.pat)}'

    def get_hashable_map(self):
        return tuple(np.nonzero(self.instance_map)[0])

    def prob(self, feat_probs):
        return np.product([feat_probs[cf] for sublist in self.pat for cf in sublist])

    def interest(self, feat_probs):
        return sum(self.instance_map) / self.prob(feat_probs)


def iterate_pattern(pat, main_feat_seq, feat_probs, feature_map, num_variants=3, min_freq=5):
    # cur_pattern_prob = np.product([feat_probs[cf] for sublist in pat for cf in sublist])
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
    interest = [c[cf] / (pat.prob(feat_probs) * feat_probs[cf]) for cf in candidate_features]

    # add possibility of S-STEP which extends pattern
    # only if the last step wasn't also an S-STEP
    if bool(pat.pat[-1]):
        interest.append(pat.interest(feat_probs))
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

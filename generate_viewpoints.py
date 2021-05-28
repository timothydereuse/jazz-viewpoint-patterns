import music21 as m21
import numpy as np
from copy import deepcopy
from collections import Counter

quarter_beat_multiplier = 12

def collapse_tied_notes(mus_xml):
    mus_xml_copy = deepcopy(mus_xml)
    # consolidate ties into single notes
    tied_notes = [x for x in mus_xml_copy.flat.notes if x.tie]
    last_not_tied_ind = 0
    for i, n in enumerate(tied_notes):
        if n.tie.type == 'start':
            n.tie = None
            last_not_tied_ind = i
        else:
            mus_xml_copy.remove(n, recurse=True)
            new_dur = m21.duration.Duration(tied_notes[last_not_tied_ind].duration.quarterLength + n.duration.quarterLength)
            tied_notes[last_not_tied_ind].duration = new_dur
    return mus_xml_copy

def collapse_rests(mus_xml, min_rest_len = 0.25):
    mus_xml_copy = deepcopy(mus_xml)
    # consolidate ties into single notes
    notes = list(mus_xml_copy.flat.notesAndRests)
    for i in range(len(notes) - 1):
        this_event = notes[i]
        next_event = notes[i + 1]

        if not(this_event.isNote and next_event.isRest):
            continue

        if next_event.duration.quarterLength >= min_rest_len:
            continue

        new_dur = m21.duration.Duration(this_event.duration.quarterLength + next_event.duration.quarterLength)
        this_event.duration = new_dur
        mus_xml_copy.remove(next_event, recurse=True)

    return mus_xml_copy



def get_viewpoints(mus_xml):
    vps = {}
    # EXTRACT VIEWPOINTS

    notes = list(mus_xml.flat.notes)
    pitches = [n.pitch.midi for n in notes]
    durs = [n.duration.quarterLength * quarter_beat_multiplier for n in notes]
    offsets = [n.offset * quarter_beat_multiplier for n in notes]
    intervals = [m21.interval.Interval(notes[i], notes[i+1]) for i in range(len(notes) - 1)]

    # MIDI pitch
    # vps['pitches'] = pitches

    # pitch class
    pc_int = [n % 12 for n in pitches]
    vps['pc_int'] = pc_int

    # duration
    vps['durs'] = durs

    # duration contour
    dur_contour = np.sign(np.insert(np.diff(durs), 0, 0))
    dur_contour = [(x if x != 0 else np.nan) for x in dur_contour]
    vps['dur_contour'] = dur_contour

    # duration ratio
    # dur_ratio = [durs[i+1] / durs[i] for i in range(len(durs) - 1)]
    # dur_ratio = np.round(np.log2(np.array(dur_ratio, dtype='float')))
    # vps['dur_ratio'] = np.append(dur_ratio, 0).astype('int')

    # down-beats
    down_beats = (np.array(offsets) % (quarter_beat_multiplier * 4)) == 0
    down_beats = [(x if x else np.nan) for x in down_beats]
    vps['down_beats'] = down_beats

    # off-beats
    # off_beats = (np.array(offsets) % (quarter_beat_multiplier * 4))
    # off_beats = np.gcd(off_beats, 2**10)
    # off_beats = [(1 if x == 2**10 else x) for x in off_beats]
    # vps['beat_subdivision'] = off_beats

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

    # peaks and troughs
    pt = [0] + pitches + [0]
    peaks = [pt[i] > pt[i+1] and pt[i] > pt[i-1]
        for i in range(1, len(pt) - 1)]
    troughs = [pt[i] < pt[i+1] and pt[i] < pt[i-1]
        for i in range(1, len(pt) - 1)]
    pnt = [peaks[i] - troughs[i] for i in range(len(peaks))]
    vps['melodic_peaks'] = pnt

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

    return main_feat_seq


def viewpoint_seq_to_array(main_feat_seq, prob_exponent=1):
    c = Counter([item for sublist in main_feat_seq for item in sublist])
    all_viewpoint_keys = sorted(list(c.keys()))
    key_to_index = { all_viewpoint_keys[i]: i for i in range(len(all_viewpoint_keys))}
    num_events = len(main_feat_seq)

    arr = np.zeros([num_events, len(all_viewpoint_keys)])
    for i, vps in enumerate(main_feat_seq):
        idxs = [key_to_index[vp] for vp in vps]
        arr[i, idxs] = 1

    feat_probs = {e: ((c[e] + 1) / num_events) ** (prob_exponent) for e in all_viewpoint_keys}
    feat_probs_vector = [ 1 / feat_probs[k] for k in all_viewpoint_keys]

    arr = arr * np.array(feat_probs_vector)

    return arr, feat_probs

if __name__ == '__main__':

    fname = "Meshigene - transcription sax solo.musicxml"
    mus_xml = m21.converter.parse(fname)
    x = list(mus_xml.recurse().getElementsByClass('PageLayout'))
    mus_xml.remove(x, recurse=True)

    mus_xml_copy = collapse_tied_notes(mus_xml)

    main_feat_seq = get_viewpoints(mus_xml_copy)

    arr, feat_probs = viewpoint_seq_to_array(main_feat_seq, prob_exponent=0.5)

    x = np.matmul(arr, arr.T)

    x[np.identity(x.shape[0]) == 1] = 0

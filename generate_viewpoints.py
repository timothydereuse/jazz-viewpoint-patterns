from argparse import ArgumentError
import music21 as m21
import numpy as np
from copy import deepcopy
from collections import Counter
from numpy.lib.stride_tricks import as_strided
from pip import main

quarter_beat_multiplier = 12

def get_subsequences(arr, m):
    n = arr.size - m + 1
    s = arr.itemsize
    return as_strided(arr, shape=(m,n), strides=(s,s))

def get_all_subsequences(num_events, cardinalities):
    sqs = [list(get_subsequences(np.arange(num_events), num_events - (i - 1))) for i in cardinalities]
    sqs = [item for sublist in sqs for item in sublist]
    sqs = sorted(sqs, key=lambda x: tuple(x))
    return sqs

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


def clean_mus_xml(score):
    flat_notes = list(score.flat.notes)
    flat_notes = [x for x in flat_notes if (not x.tie) or (x.tie.type == 'start')]
    flat_notes = [n for n in flat_notes if not type(n) is m21.harmony.ChordSymbol]
    return flat_notes

def get_viewpoints(mus_xml):
    vps = {}
    # EXTRACT VIEWPOINTS

    notes = list(mus_xml.flat.notes)
    notes = [n for n in notes if not type(n) is m21.harmony.ChordSymbol]
    notes = [n if not n.isChord else n.notes[-1] for n in notes]
    pitches = [n.pitch.midi if not n.isChord else n.pitches[-1].midi for n in notes]
    durs = [n.duration.quarterLength * quarter_beat_multiplier for n in notes]
    offsets = [n.offset * quarter_beat_multiplier for n in notes]
    intervals = [m21.interval.Interval(notes[i], notes[i+1]) for i in range(len(notes) - 1)]

    # MIDI pitch
    vps['pitches'] = pitches

    # pitch class
    pc_int = [n % 12 for n in pitches]
    vps['pitch_class'] = pc_int

    # duration
    vps['durs'] = durs

    # duration contour
    dur_contour = np.sign(np.insert(np.diff(durs), 0, 0))
    vps['dur_contour'] = dur_contour

    # duration ratio
    # dur_ratio = [durs[i+1] / durs[i] for i in range(len(durs) - 1)]
    # dur_ratio = np.round(np.log2(np.array(dur_ratio, dtype='float')))
    # vps['dur_ratio'] = np.append(dur_ratio, 0).astype('int')

    # down-beats
    down_beats = (np.array(offsets) % (quarter_beat_multiplier * 4)) == 0
    down_beats = [(x if x else 0) for x in down_beats]
    vps['down_beats'] = down_beats

    # off-beats
    # off_beats = (np.array(offsets) % (quarter_beat_multiplier * 4))
    # off_beats = np.gcd(off_beats, 2**10)
    # off_beats = [(1 if x == 2**10 else x) for x in off_beats]
    # vps['beat_subdivision'] = off_beats

    # time till next note
    next_offset_time = np.append(np.diff(offsets), 10)
    vps['next_offset_time'] = next_offset_time

    # notes greater than median duration
    dur_thresh = np.median(next_offset_time)
    long_dur = [(0 if x <= dur_thresh else 1) for x in next_offset_time]
    vps['long_durs'] = long_dur

    # pitches greater than median pitch
    pitch_thresh = np.median(pitches)
    hi_pitch = [(0 if x <= pitch_thresh else 1) for x in pitches]
    vps['high_pitches'] = hi_pitch

    # time till next note minus duration
    rest_pad = next_offset_time - durs
    rest_pad = [(x if x != 0.0 else 0) for x in rest_pad]
    vps['rest_pad'] = rest_pad

    # interval from prev. note
    intervals_semitones = np.append(0, [x.semitones for x in intervals])
    vps['intervals_semitones'] = intervals_semitones

    # interval class
    interval_class = np.append(0, [x.intervalClass for x in intervals])
    vps['interval_class'] = interval_class

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
    skips = np.append(0, [x.isSkip * np.sign(x.semitones) for x in intervals])
    vps['skips'] = skips

    rough_contour = [skips[i] + contour[i] for i in range(len(contour))]
    vps['rough_contour'] = rough_contour

    sharp_peaks = [(skips[i] or skips[i + 1]) and pnt[i] for i in range(len(skips) - 1)]
    sharp_peaks.append(0)
    vps['sharp_melodic_peaks'] = sharp_peaks

    # diatonic interval size
    try:
        diatonic_int_size = np.append(0, [int(x.name[-1]) for x in intervals])
        vps['diatonic_int_size'] = diatonic_int_size
    except IndexError:
        ind = [x for x in range(len(intervals)) if intervals[x].name == '']
        print(notes[ind[0]], notes[ind[1]])
        raise

    # convert to list-of-sets
    main_feat_seq = np.array([set() for _ in range(len(notes))])
    main_feat_dict = np.array([{} for _ in range(len(notes))])
    fkeys = sorted(list(vps.keys()))
    for k in fkeys:
        for i in range(len(notes)):
            val = vps[k][i]
            # filter out NaNs
            if val is not None:
                val = int(val)

            main_feat_seq[i].add((k, val))
            main_feat_dict[i][k] = val

    return main_feat_dict


def viewpoint_seq_from_dict(main_feat_dict, key_list):
    res = []
    for i, e in enumerate(main_feat_dict):
        vp = tuple([e[k] for k in key_list])
        res.append(vp)
    return res


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



def get_timescaled_signal(main_feat_seq, sqs, target_length=120):

    vps = viewpoint_seq_from_dict(main_feat_seq, ['durs', 'pitches'])

    skeleton = np.array(vps)

    stretched_arrays = np.zeros([len(sqs), target_length, 2])

    for n, seq in enumerate(sqs):
        skel_elements = skeleton[seq]
        intervals = np.concatenate([[1], np.diff(skel_elements[:, 1])])
        cum_sums = np.concatenate([[0], np.cumsum(skel_elements[:, 0])])
        scale_up = target_length / np.sum(skel_elements[:, 0])
        scaled_cum_sums = np.round(cum_sums * scale_up).astype(int)
        out_arr = np.zeros((target_length, 2))
        for i in range(len(scaled_cum_sums) - 1):
            start, end = scaled_cum_sums[i], scaled_cum_sums[i+1]
            out_arr[start:end, 0] = np.linspace(3, 0, end - start) ** 2
            out_arr[start:end, 1] = intervals[i]
        stretched_arrays[n] = out_arr

    return stretched_arrays

if __name__ == '__main__':


    cardinalities = [5, 6, 7]


    fname = "Meshigene - transcription sax solo.musicxml"
    mus_xml = m21.converter.parse(fname)
    x = list(mus_xml.recurse().getElementsByClass('PageLayout'))
    mus_xml.remove(x, recurse=True)
    mus_xml_copy = collapse_tied_notes(mus_xml)
    main_feat_seq = get_viewpoints(mus_xml_copy)

    num_events = len(main_feat_seq)
    sqs = get_all_subsequences(num_events, cardinalities)
    timescaled = get_timescaled_signal(main_feat_seq, sqs, target_length=100)


    # arr, feat_probs = viewpoint_seq_to_array(main_feat_seq, prob_exponent=0.5)

    # entry = timescaled[423]
    # plt.clf()
    # plt.plot(entry[:, 0])
    # plt.plot(entry[:, 1])
    # plt.show()
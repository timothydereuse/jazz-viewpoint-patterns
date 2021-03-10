import music21 as m21
import numpy as np
from copy import deepcopy

quarter_beat_multiplier = 24

def vis_motif(pat, mus_xml):

    score = deepcopy(mus_xml)

    starts = np.nonzero(pat.instance_map)[0]
    flat_notes = list(score.flat.notes)
    flat_notes = [x for x in flat_notes if (not x.tie) or (x.tie.type == 'start')]

    pat.pat = [x for x in pat.pat if x]
    cardinality = len(pat)

    for i, st_ind in enumerate(starts):
        flat_notes[st_ind].lyric = f'#{i}:'
        for ind in np.arange(st_ind, st_ind + cardinality):
            flat_notes[ind].style.color = 'red'

    for i, ft in enumerate(pat.pat):

        string_rep = []
        for f in sorted(list(ft)):
            key = f[0]
            val = f[1]
            if key in ['durs', 'rest_pad']:
                val = np.round(val / quarter_beat_multiplier, 3)
            string_rep.append(f'{key}: {val} \n')
        string_rep = f'Note {i + 1}: ' + ', '.join(string_rep)

        tb = m21.text.TextBox(string_rep, 100, (-7000 * i))
        tb.style.fontSize = 8
        tb.page = 1
        # tb.style.justify = 'left'
        score.append(tb)

    return score


def vis_developing_motif(seqs, mus_xml):

    score = deepcopy(mus_xml)

    flat_notes = list(score.flat.notes)
    flat_notes = [x for x in flat_notes if (not x.tie) or (x.tie.type == 'start')]

    for i, seq in enumerate(seqs):
        flat_notes[seq[0]].lyric = f'#{i}:'
        for ind in seq:
            flat_notes[ind].style.color = 'red'

    # for i, ft in enumerate(pat.pat):
    #
    #     string_rep = []
    #     for f in sorted(list(ft)):
    #         key = f[0]
    #         val = f[1]
    #         if key in ['durs', 'rest_pad']:
    #             val = np.round(val / quarter_beat_multiplier, 3)
    #         string_rep.append(f'{key}: {val} \n')
    #     string_rep = f'Note {i + 1}: ' + ', '.join(string_rep)
    #
    #     tb = m21.text.TextBox(string_rep, 100, (-7000 * i))
    #     tb.style.fontSize = 8
    #     tb.page = 1
    #     # tb.style.justify = 'left'
    #     score.append(tb)

    return score

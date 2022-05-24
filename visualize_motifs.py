import music21 as m21
import numpy as np
from copy import deepcopy
import os

quarter_beat_multiplier = 24

def export_motifs_to_pdf(motifs_to_export, mus_xml, vp_seq, params, tune_name):
    keys_str = '-'.join(params['keys'])
    folder_name = os.path.join('./exports', f'{keys_str} {tune_name}')
    os.mkdir(folder_name)
    for i, motif in enumerate(motifs_to_export):

        viz_seqs = motif['note_inds']
        occs = motif['num_occurrences']
        score = motif['prob_score']
        cardinality = np.round(motif['mean_cardinality'], 2)
        fname = os.path.join(f'{folder_name}', f'{tune_name}-{i} freq-{occs} card-{cardinality}')
        viz_score = vis_developing_motif(viz_seqs, mus_xml)
        viz_score.write('musicxml.pdf', fp=str(fname))

        flat_notes = list(mus_xml.flat.notes)
        flat_notes = [x for x in flat_notes if (not x.tie) or (x.tie.type == 'start')]
        flat_notes = [n for n in flat_notes if not type(n) is m21.harmony.ChordSymbol]
        flat_notes = [n if not n.isChord else n.notes[-1] for n in flat_notes]

        with open(f"{fname} description.txt", "a") as f:
            f.write(f'Params: {str(params)} \n')
            f.write(f'Sequence score = {score:.3f}\n')
            for j, seq in enumerate(viz_seqs):
                f.write(f'Occurrence {j}: notes {str(seq)}\n')
                seq_records = ''
                note_records = ''
                for k, idx in enumerate(seq):

                    next_offset = flat_notes[idx + 1].offset - flat_notes[idx].offset if (idx + 2) < len(flat_notes) else 1000
                    rest_amt = next_offset - flat_notes[idx].duration.quarterLength
                    rest_string = f'+Rest{rest_amt}' if rest_amt > 0.01 else ''
                    try:
                        note_record = f'{flat_notes[idx].pitch.name}{flat_notes[idx].pitch.octave}-' \
                                    f'{flat_notes[idx].duration.type}{rest_string} '
                    except AttributeError:
                        note_record = 'ERR '
                    vps = str(vp_seq[idx]).replace('\'', r'').replace('),', ')')
                    seq_records = f'{seq_records} {vps}'
                    note_records += note_record
                f.write(f'{seq_records}\n{note_records}\n\n')


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
    flat_notes = [n for n in flat_notes if not type(n) is m21.harmony.ChordSymbol]

    for i, seq in enumerate(seqs):
        flat_notes[seq[0]].lyric = f'#{i}:'
        for ind in seq:
            flat_notes[ind].style.color = 'red'

    return score

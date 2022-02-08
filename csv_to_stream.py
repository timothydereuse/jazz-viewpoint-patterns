import numpy as np
import music21 as m21
import csv
from fractions import Fraction

beat_multiplier = 6

fname = "D:\Documents\datasets\jazzsolos\LeeKonitz_I'llRememberApril_Solo"

with open(f'{fname}.csv') as f:
    reader = csv.reader(f)
    next(reader, None)
    notes = [(float(row[0]), float(row[1]), int(float(row[2]))) for row in reader]

with open(f'{fname}_beats.csv') as f:
    reader = csv.reader(f)
    next(reader, None)
    beats = [(int(row[0]), int(row[1]), float(row[2])) for row in reader]

xp = np.array([x[2] for x in beats])
fp = np.array([x[0] for x in beats]) * 4
note_onsets = np.array([x[1] for x in notes])
note_durations = np.array([x[0] for x in notes])
note_offsets = note_onsets + note_durations

y = np.interp(note_onsets, xp, fp)
# y = y * 4

off = np.interp(note_onsets + note_durations, xp, fp)
# off = off * 4

durations = off - y

s = m21.stream.Stream()

for i in range(len(note_onsets) - 1):
    dur_float = Fraction((y[i + 1] - y[i]) * 2).limit_denominator(4)
    dur_float = dur_float / 2
    print(dur_float)

    duration = m21.duration.Duration(dur_float)
    n = m21.note.Note(notes[i][2], duration=duration)
    s.append(n)

s.write('musicxml.pdf', fp='./exports/testexport')

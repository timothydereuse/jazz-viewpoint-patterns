import music21 as m21
import numpy as np

fname = "Meshigene - transcription sax solo.musicxml"
quarter_beat_multiplier = 12
c = m21.converter.parse(fname)

# for n in c.flat.notesAndRests:
#     print(n, n.duration)

vps = []
# EXTRACT VIEWPOINTS

notes = list(c.flat.notes)
pitches = [n.pitch.midi for n in notes]
durs = [n.duration.quarterLength * quarter_beat_multiplier for n in notes]
offsets = [n.offset * quarter_beat_multiplier for n in notes]
intervals = [m21.interval.Interval(notes[i], notes[i+1]) for i in range(len(notes) - 1)]


# MIDI pitch
vps.append(pitches)

# pitch class
pc_int = [n % 12 for n in pitches]
vps.append(pc_int)

# duration
vps.append(durs)

# duration contour
dur_contour = np.sign(np.append(np.diff(durs), 0))
vps.append(dur_contour)

# duration ratio
dur_ratio = [durs[i+1] / durs[i] for i in range(len(durs) - 1)]
dur_ratio = (np.append(dur_ratio, 0).astype('float') * quarter_beat_multiplier).round()
vps.append(dur_ratio)

# time till next note
next_offset_time = np.append(np.diff(offsets), 10)
vps.append(next_offset_time)

# time till next note minus duration
rest_pad = next_offset_time - durs
vps.append(rest_pad)

# interval from prev. note
intervals_semitones = np.append(0, [x.semitones for x in intervals])
vps.append(intervals_semitones)

# interval class
interval_class = np.append(0, [x.intervalClass for x in intervals])
vps.append(interval_class)

# contour
contour = np.sign(intervals_semitones)
vps.append(contour)

# non-stepwise motion
skips = np.append(0, [x.isSkip for x in intervals])
vps.append(skips)

# diatonic interval size
diatonic_int_size = np.append(0, [int(x.name[-1]) for x in intervals])
vps.append(diatonic_int_size)

vps = np.stack(vps, 1)

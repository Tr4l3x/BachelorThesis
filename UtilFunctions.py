# Helpfull functions for the whole project
import math
import dataset
import pretty_midi
import torch

# Pitch must be an int in [0.127]
def pitch_to_note(pitch):
    if pitch not in range(0,128):
        raise AssertionError("Es wurde ein ungültiger Pitch übergeben")
    else:
        if pitch % 12 == 0:
            return "C"+str(math.floor(pitch//12)-1)
        elif pitch % 12 == 1:
            return "C#/Db"+str(math.floor(pitch//12)-1)
        elif pitch % 12 == 2:
            return "D"+str(math.floor(pitch//12)-1)
        elif pitch % 12 == 3:
            return "Eb/D#"+str(math.floor(pitch//12)-1)
        elif pitch % 12 == 4:
            return "E"+str(math.floor(pitch//12)-1)
        elif pitch % 12 == 5:
            return "F"+str(math.floor(pitch//12)-1)
        elif pitch % 12 == 6:
            return "F#/Gb"+str(math.floor(pitch//12)-1)
        elif pitch % 12 == 7:
            return "G"+str(math.floor(pitch//12)-1)
        elif pitch % 12 == 8:
            return "Ab/G#"+str(math.floor(pitch//12)-1)
        elif pitch % 12 == 9:
            return "A"+str(math.floor(pitch//12)-1)
        elif pitch % 12 == 10:
            return "Bb/A#"+str(math.floor(pitch//12)-1)
        elif pitch % 12 == 11:
            return "B"+str(math.floor(pitch//12)-1)


# TODO: note_to_pitch Implementation?
# Note notation must be in "NoteOctaveNumber"
def note_to_pitch(note):
    if isinstance(note, str):
        match note:
            case "C":
               print("test")
    else:
        raise AssertionError("Als Note muss ein String übergeben werden")



"""
   Function for converting piano rolls into a midi data 
   
   Arg:
      tpq(int): Ticks per quarter
"""
def roll_to_midi(rolls:list, spt):
    midi = pretty_midi.PrettyMIDI(initial_tempo=218, resolution=12)
    piano = pretty_midi.Instrument(program=0)
    time_bias = 0
    for bar in rolls:
        notes_of_bar = []
        for pitch in range(bar.shape[0]):
            onsets = sum(torch.nonzero(bar[pitch]).tolist(), [])
            if len(onsets) != 0:
                notes = ascending_patition(onsets)
                for note in notes:
                    if type(note) != list:
                        print(notes)
                        print(onsets)
                        raise AssertionError("Wieso ist note keine liste?")
                    onset_tick = note[0]
                    offset_tick = note[-1]
                    if onset_tick == offset_tick:
                        offset_tick = offset_tick + 1
                    midi_note = pretty_midi.Note(velocity=100, pitch=pitch + 36,
                                                 start=time_bias + (onset_tick*spt), end=time_bias + (offset_tick*spt))
                    notes_of_bar.append(midi_note)
        notes_of_bar = sorted(notes_of_bar, key=lambda x: x.start)
        for note in notes_of_bar:
            piano.notes.append(note)
        time_bias = time_bias + (48*spt)
    midi.instruments.append(piano)
    midi.write('test.mid')

def ascending_patition(onsets):
    # Edgecase if there is only one onset
    if len(onsets) == 1:
        res = []
        res.append(onsets)
        return res
    result = []
    seq = []
    for i in range(len(onsets)):
        if (len(seq) == 0 or (onsets[i] == onsets[i-1]+1)) and i != len(onsets)-1:
            seq.append(onsets[i])
            continue
        elif i == len(onsets) - 1:
            if onsets[i] == (onsets[i-1]+1):
                seq.append(onsets[i])
            else:
                result.append(seq)
                seq = [onsets[i]]
        result.append(seq)
        seq = [onsets[i]]
    return result

def get_spt(bpm, num_bars):
    num_of_beats = num_bars * 4
    seconds_per_beat = 60 / bpm
    duration = num_of_beats * seconds_per_beat
    sec_per_tick = duration / (num_bars * 48)
    return sec_per_tick

test = dataset.get_dataset()
test_sample = test.piano_rolls[-1]
spt = get_spt(218, 54)
for sample in test.piano_rolls:
    roll_to_midi(sample, spt)










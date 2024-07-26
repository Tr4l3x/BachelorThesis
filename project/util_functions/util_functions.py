import math
import project.dataset as dataset
import pretty_midi
import torch

"""
    Converting a given list of rolls to a midi data

    For every pitch in every bar of the sample, the onsets are computed. Those are divided into 
    sequences of consecutive following ticks. A consecutive sequence is assumed to be one note 
    played for len(sequence) ticks. Those notes are 

    Args:
        rolls (list): List with all piano rolls of a sample
        spt (int): Means seconds per tick, depending on used avg tempo
"""
def roll_to_midi(rolls:list, spt, sample_name):
    # Init. the new MIDI data object
    midi = pretty_midi.PrettyMIDI(initial_tempo=120, resolution=12)
    piano = pretty_midi.Instrument(program=0)
    # Time bias for calculating "start" time of each bar
    time_bias = 0
    for bar in rolls:
        notes_of_bar = []
        for pitch in range(bar.shape[0]):
            # For every pitch all ticks where the pitch is 1 are computed
            onsets = sum(torch.nonzero(bar[pitch]).tolist(), [])
            # If the pitch has at least one tick where it is played
            if len(onsets) != 0:
                # Divides all onsets into interconnected sequences of consecutive following ticks
                notes = ascending_patition(onsets)
                # Every note is added to the midi data
                for note in notes:
                    onset_tick = note[0]
                    offset_tick = note[-1]
                    # If a note is played for one tick, this might lead to problems otherwise
                    if onset_tick == offset_tick:
                        offset_tick = offset_tick + 1
                    # Creating and adding current note to the notes_of_bar list
                    midi_note = pretty_midi.Note(velocity=100, pitch=pitch + 36,
                                                 start=time_bias + (onset_tick*spt), end=time_bias + (offset_tick*spt))
                    notes_of_bar.append(midi_note)
        # Sorting all notes of a bar by their starting point (tick)
        notes_of_bar = sorted(notes_of_bar, key=lambda x: x.start)
        # Adding them to the midi data
        for note in notes_of_bar:
            piano.notes.append(note)
        # Moving time bias for next bar
        time_bias = time_bias + (48*spt)
    midi.instruments.append(piano)
    midi.write(sample_name)


"""
    Computes all sequences of consecutive following numbers. ([1,2,4,5] into [1,2],[4,5]).
    It's similar to a partition with ascending values through all partitions

    Args:
        onsets(list): List with integers (ticks) that has to be partitioned into consecutive following nums
    Returns:
        result(2D list): Holds a list for every sequences of consecutive following from the input
"""
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

"""
    Computes the seconds per tick for later time mapping in MIDI data.

    Args:
        bpm (int): Avg. bpm for the sample for that the seconds per tick are computed
        num_bars (int): Number of bars of the sample
"""
def get_spt(bpm, num_bars):
    # Number of beats in the whole sample
    num_of_beats = num_bars * 4
    # Seconds per beat depending on the current bpm (by using reciprocal of bpm/sec)
    seconds_per_beat = 60 / bpm
    # Quantized duration of the whole sample
    duration = num_of_beats * seconds_per_beat
    # Seconds per tick is equal to the duration in secs divided by number of ticks of the whole sample
    sec_per_tick = duration / (num_bars * 48)
    return sec_per_tick



#### Test case #####
data = dataset.get_dataset()
test_sample = data.piano_rolls[1]
spt = get_spt(120, len(test_sample))
roll_to_midi(test_sample, spt, 'AnderesTempo.mid')



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

# Note notation must be in "NoteOctaveNumber"
def note_to_pitch(note):
    if isinstance(note, str):
        match note:
            case "C":
               print("test")
    else:
        raise AssertionError("Als Note muss ein String übergeben werden")











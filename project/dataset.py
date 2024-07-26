import numpy as np
import math
import csv
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from collections import Counter

class WeimarJazzDataset():
    """
        Initialization of WeimarJazzDataset for the thesis

        Reads the csv by given path_to_csv. While reading the csv every note tuple (sample) is filtered by
        valid signature and format. Every note-tuple and name of a valid sample is filtered and added to the attributes.
        With this tuples the piano_rolls and some statistic attributes are computed.

        Attributes:
            csv_path (str): Path of the csv data with extracted note features.
            sample_names (list): Names of valid samples filtered while reading csv
            note_tuples (2D list): Note-tuple information of every valid note from the csv
            notes_of_each_sample (2D list): List with all notes of sample i at index [i]
            global_min_pitch (int): Starting pitch of the lowest octave used by any solo
            global_max_pitch (int): Ending pitch of the highest octave used by any solo
            piano_rolls (2D list): List with all piano-roll representations of sample i at index [i]
            num_of_rolls (int): Total number of piano rolls (bars) after preprocessing the dataset
    """
    def __init__(self, path_to_csv, device):
        self.device = device
        # Saving the source path for possible lookups
        self.csv_path = path_to_csv
        # Saving all valid samples names and their note tuples
        self.sample_names, self.note_tuples = self.reading_notes_from_csv(path_to_csv)
        # Saving the mapping from notes of the same samples for possible tuple information look ups
        self.notes_of_each_sample = self.get_notes_of_all_samples(self.note_tuples)
        # Saving global min and max pitch for later reducing pitch_size of piano rolls
        self.global_min_pitch, self.global_max_pitch = self.get_min_max_pitches(self.note_tuples)
        # Computing the PianoRolls of each sample
        init_piano_rolls = []
        for notes in self.notes_of_each_sample:
            init_piano_rolls.append(self.get_piano_roll_of_sample(notes))
        # Reducing the pitch size of every piano_roll
        self.piano_rolls = self.reduce_piano_rolls(init_piano_rolls)

        # Meta Information after processing
        num_of_rolls = 0
        for roll in self.piano_rolls:
            num_of_rolls = num_of_rolls + len(roll)
        self.num_of_rolls = num_of_rolls

    def reading_notes_from_csv(self, path):
        """
           Reads the csv data at the given path and prepares it for further processing.

           Reads the csv by given path. Every row of the csv will be checked for correct format.
           If so the csv row is appended to the dataset-list of all notes. Every note of a sample
           that has a signature different to 4/4 or incorrect format is not added to the dataset list.

           Args:
               path (str): Path to the csv data (which format is describe in the thesis)

           Returns:
               sample_names (list): Names of every sample that meets all the criteria
               notes (list): Every extracted features of each note in the WeimarerJazz Dataset
        """
        # Reading the csv data by given path
        with open(path) as csvdatei:
            csv_reader_object = csv.reader(csvdatei)
            # Skipping the first row since there are the labels of the meta information
            next(csv_reader_object
                 )
            sample_names = []
            notes = []
            for row in csv_reader_object:
                # Checking for correct format with 13 features (columns)
                if len(row) != 13:
                   continue
                note_metric = row[2]
                sample_signature = int(note_metric[0])
                # Checking for correct signature which should be 4
                if sample_signature != 4:
                    continue
                notes.append(row)
                if row[0] not in sample_names:
                    sample_names.append(row[0])
            return sample_names, notes

    def get_notes_of_all_samples(self, csv_list):
        """
            Assigning all note information of same samples in the same list.

            By looking for matching names every note is assigned to list in samples. If samples[i] is empty,
            the current note is from an unseen sample. If not, we look for matching names. In case that is true,
            note is added at current sample_index, otherwise current_index is incremented.

            Args:
                csv_list (list): List with all information extracted from source csv data.

            Returns:
                samples (list): 2D list with all notes of sample i in the list at index samples[i].
            """
        # Number of tracks to convert into piano rolls is empirically fixed
        num_of_samples = len(self.sample_names)
        samples = [[] for _ in range(num_of_samples)]
        sample_index = 0
        for note in csv_list:
            # Case for we arrived a new "track" in the csv_list of notes
            if len(samples[sample_index]) != 0 and samples[sample_index][0][0] != note[0]:
                sample_index = sample_index + 1
            samples[sample_index].append(note)
        return samples

    def get_piano_roll_of_sample(self, notes_of_sample):
        """
            Returns a list with length=num_of bar. Every entry holds the piano roll of the bar.

            For later piano_roll representation the number of total ticks of each sample is needed to map note-duration.
            This is computed by the number of bars(extracted from last node of sample) and TicksPerBar Parameter (=48).

            Every note of the notes_of_sample list is assigned to a list with all notes from the same bar.
            Afterward for every bar in sub-list in notes_of_every_bar, the piano-roll representation is computed
            and appended to the return list.

            Args:
                notes_of_sample (list): List with every note of one unique sample

            Returns:
                piano_rolls_of_every_bar (list, np.array): 2D-List with the piano_roll of bar i at index [i]
        """
        # Since notes_of_sample[-1] holds the last note, we can recieve the ammount of bars by this note metric position
        last_node = notes_of_sample[-1]

        numeric_list_of_node = list(last_node[2].split("."))
        # In metric position *.*.a.b.c the third last value is the bar number
        num_bars = int(numeric_list_of_node[-3])

        # For mapping their duration in the piano roll, we need the avg ticks per sec of a sample
        # Since we fixed tpb to 48, this is the amount of ticks of the current sample
        total_ticks_of_sample = num_bars * 48

        notes_of_every_bar = [[] for _ in range(num_bars)]
        for note in notes_of_sample:
            numeric_list_of_node = list(note[2].split("."))
            # Since in metric position the third last number is the bar of the node
            bar_of_node = int(numeric_list_of_node[-3])

            # Some samples have bars that are not counted (bar<=0). We assume to skip these
            if bar_of_node <= 0:
                continue
            else:
                notes_of_every_bar[bar_of_node - 1].append(note)
        # notes_of_every_bar[i] is a list of all notes of bar i

        # Then we create the piano roll of every bar
        piano_rolls_of_every_bar = []
        for bar in notes_of_every_bar:
            piano_rolls_of_every_bar.append(self.get_piano_roll_of_bar(bar, total_ticks_of_sample))

        return piano_rolls_of_every_bar

    def get_piano_roll_of_bar(self, notes_of_bar, total_ticks_of_sample):
        """
            Mapps all notes in notes_of_bar into a piano-roll representation 128x48

            Every bar will be represented in a piano roll with shape (Pitch=128,TicksPerBar=48).
            With the metric position information, for every note their onset-tick and offset-tick is computed.
            With the ticks_of_sample their duration is also mapped to a tick representation.
            (Further information to metric-position notation via MeloSPY Documentation)

            Args:
                notes_of_bar (list): List with every note of one unique bar.
                total_ticks_of_sample (int): Calculated with num_bars of a track and the tpb parameter.

            Returns:
                piano_roll (tensor): Piano_Roll representation of shape (128,48) from given bar notes.
        """
        piano_roll = np.zeros((128, 48))
        for note in notes_of_bar:
            # Extracting important note information
            note_duration = float(note[1])
            note_metric = note[2]
            note_pitch = int(note[7])
            sample_duration = float(note[9])

            # Splitting the metric notation
            note_metric = list(note_metric.split("."))
            period = int(note_metric[0])
            num_tatum = int(note_metric[1])
            beat_pos = int(note_metric[3])
            tatum_pos = int(note_metric[4])
            # Period means the signature of the current sample. There should be no signatures different to 4!
            if period != 4:
                raise AssertionError("The Dataset includes samples with different signature as 4/4 !")
            # Otherwise map their metric pos into a width of 48 ticks per bar (12 ticks per beat)
            beat_mapping = (beat_pos - 1) * 12
            # num_tatums is the number of tatums of the current bar, so we divide the num of ticks by num_of_tatums
            # to map the tatumPos to an intervall of [0,11]
            tatum_mapping = int((12 / num_tatum) * (tatum_pos - 1))
            note_onset = beat_mapping + tatum_mapping

            # Map their duration to the current beat of the bar
            note_duration_in_ticks = max(int((total_ticks_of_sample * note_duration) / sample_duration), 1)
            note_offset = note_onset + note_duration_in_ticks

            # Marking the current note in the piano roll
            piano_roll[note_pitch, note_onset:note_offset] = note_pitch
        piano_roll_tensor = torch.tensor(piano_roll).to(self.device)
        return piano_roll_tensor

    def get_min_max_pitches(self, notes):
        """
            Computes starting of the lowest octave and ending pitch of the highest octave used by any solo

            Filters the global min-/max-pitches of every note in the argument list. With this information
            the lowest/highest octave used by any note is computed and can be used for later reducing the pitch-range
            represented in the piano rolls.

            Args:
                notes (list): List of note tuple information

            Returns:
                global_min_pitch (int): Starting pitch of the lowest octave used by any solo
                global_max_pitch (int): Ending pitch of the highest octave used by any solo
        """
        min_pitch = 128
        max_pitch = 0
        # Filtering min and max pitch of the whole dataset
        for note in notes:
            note_pitch = int(note[7])
            min_pitch = min(note_pitch, min_pitch)
            max_pitch = max(note_pitch, max_pitch)
            # But we might want to have whole octaves represented
        global_min_pitch = math.floor(min_pitch / 12) * 12
        global_max_pitch = (math.floor(max_pitch / 12) + 1) * 12 - 1
        return global_min_pitch, global_max_pitch

    def reduce_piano_rolls(self, rolls):
        """
            Reduces all piano rolls to the shape of (72=highest_pitch-lowest_pitch, 48=ticks

            Using attributes global_min_pitch/global_max_pitch, we reduce the shape of every piano roll.

            Args:
                rolls (2D list): List with every piano roll from each sample

            Returns:
                reduces_piano_rolls (2D list): Reduced Piano_Roll representation of shape (72,48) from given bar notes.
        """
        reduced_piano_rolls = []
        for sample in rolls:
            current_sample = []
            for roll in sample:
                current_sample.append(roll[self.global_min_pitch:self.global_max_pitch+1, :])
            reduced_piano_rolls.append(current_sample)
        return reduced_piano_rolls

def get_dataset(device="cpu"):
    dataset = WeimarJazzDataset(path_to_csv='csv\weimarjazz.csv', device=device)
    return dataset

" Converting a list of given samples into a tensor"
def get_samples_as_tensor(dataset):
    samples = []
    for sample in dataset.piano_rolls:
        for roll in sample:
            samples.append(roll)
    stacked_samples = torch.stack(samples)
    print(stacked_samples.shape)
    return stacked_samples

" Preparing Dataset as Dataloader"
class TensorDataset(Dataset):
    def __init__(self, data_tensors, transform=None):
        if len(data_tensors.shape) == 3:
            self.data_tensors = data_tensors.unsqueeze(1)
        elif len(data_tensors.shape) == 4:
            self.data_tensors = data_tensors
        self.transform = transform

    def __len__(self):
        return len(self.data_tensors)

    def __getitem__(self, idx):
        sample = self.data_tensors[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample

" Returns a dataloader object with the WeimarJazzDataset as data"
def get_dataloader():
    dataset = get_dataset()
    data_tensors = get_samples_as_tensor(dataset)

    transform = transforms.Compose([
        #transforms.Normalize(mean=[0.5], std=[0.5])  # Beispielhafte Normalisierung für einen Kanal
    ])

    #dataset = TensorDataset(data_tensors, transform=transform)
    dataset = TensorDataset(data_tensors)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
    return dataloader

def partition_by_initial(strings):
    """
        Erstellt eine Partition der Liste 'strings', wobei jede Partitionsmenge
        mit demselben Anfangsbuchstaben beginnt.

        param:
         strings: Liste von Strings, die partitioniert werden sollen
        return:
         Ein Wörterbuch, bei dem die Schlüssel die Anfangsbuchstaben sind und die Werte Listen von Strings
    """
    partitions = {}
    for s in strings:
        if s:  # Ignoriere leere Strings
            initial = s[0]
            if initial not in partitions:
                partitions[initial] = []
            partitions[initial].append(s)
    return partitions










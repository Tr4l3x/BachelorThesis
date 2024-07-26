import csv
from collections import Counter

path = "all_existing_chords.csv"
chords = []
with open(path) as csvdatei:
    csv_reader_object = csv.reader(csvdatei)
    # Skipping the first row since there are the labels of the meta information
    next(csv_reader_object)

    for row in csv_reader_object:
        chords.append(row[0])
print("Gesamte Chord Anzahl: "+str(len(chords)))

# Grundton Arten
accidental_base = ["Cb, C#, Db, D#, Eb, E#, Fb, F#, Gb, G#, Ab, A#, Bb, B#"]
base = ["C", "D", "E", "F", "G", "A", "B"]
qualities = []

for chord in chords:
    # Grundtöne mit accidental
    if chord[0:2] in accidental_base:
        qualities.append(chord[2:])
        continue
    # Grundtöne ohne accidental
    elif chord[0] in base:
        qualities.append(chord[1:])
        continue
    # Hier wird nur der "NC" Chord der Liste hinzugefügt
    else:
        qualities.append(chord)

histogram = Counter(qualities)
with open("quality_histogram.csv", "w", newline='') as datei:
    writer = csv.writer(datei)

    writer.writerow(["Quality", "Häufigkeit"])
    for quality, frequency in histogram.items():
        writer.writerow([quality, frequency])

print("Anzahl unterschiedlicher Qualitäten: "+str(len(histogram)))
print("Anzahl unterschiedlicher Chords (Grundtöne berücksichtigt): "+str(len(set(chords))))

exit()

histogram = Counter(sorted_list)
unique_chords = set(all_chords)
print("Anzahl unterschiedlicher Chords: "+str(len(unique_chords)))

print(histogram)

with open("chords_frequency.csv", "w", newline='') as datei:
    writer = csv.writer(datei)

    writer.writerow(["Chord", "Häufigkeit"])
    for chord, frequency in histogram.items():
        writer.writerow([chord, frequency])

partitioned = partition_by_initial(unique_chords)
qualities = set()
for initial, words in partitioned.items():
    for word in words:
        qualities.add(word[1:-1])
    print(f"{initial}: {len(words)}")

qualities = list(qualities)
print(len(qualities))
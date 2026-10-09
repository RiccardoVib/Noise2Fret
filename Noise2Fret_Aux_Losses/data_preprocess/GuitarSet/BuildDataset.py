"""
Code adapted from https://github.com/KimSehun725/Tab-estimator/tree/master
"""

import os
from pathlib import Path
from Noise2Fret_Aux_Losses.src.utils import find_folder_upward
import jams_to_midi
import midi_to_numpy
import yaml
from multiprocessing import Pool
import glob
from itertools import repeat


note_resolution = 16 # th note
down_sampling_rate = 22050
bins_per_octave = 24
n_bins = 192
hop_length = 512
generated_midi_n_bars = 4
n_cores = 12

# Extract
ROOT_DIR = "../GuitarSet/"

jams_to_midi.main(note_resolution, ROOT_DIR)

kwargs = {
    "note_resolution": note_resolution,
    "down_sampling_rate": down_sampling_rate,
    "bins_per_octave": bins_per_octave,
    "n_bins": n_bins,
    "hop_length": hop_length
}

midi_dir = os.path.join("data", "midi")

# original
npz_dir = os.path.join("data", "npz", "original")
midi_file_path = os.path.join(midi_dir, "original", "*")

midi_filename_list = glob.glob(midi_file_path)
midi_filename_list.sort()
if not (os.path.exists(npz_dir)):
    os.makedirs(npz_dir)

# paralell process
p = Pool(n_cores)
p.starmap(midi_to_numpy.main, zip(midi_filename_list, repeat(kwargs)))
p.close()  # or p.terminate()
p.join()

# check for missing file
for midi_filename in midi_filename_list:
    name = os.path.split(midi_filename)[1][:-4]
    if not os.path.exists("data/npz/original/" + name + ".npz"):
        print(f"{name} does not exist!")

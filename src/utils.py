import math
import numpy as np
import re
from scipy.io.wavfile import write
import os
from pathlib import Path
import json
import torch
from torch.utils.flop_counter import FlopCounterMode
from typing import Union, Tuple


def get_flops(model, *inputs, with_backward=False):
    is_train = model.training
    model.eval()

    flop_counter = FlopCounterMode(mods=model, display=False, depth=None)
    with flop_counter:
        if with_backward:
            model(*inputs).sum().backward()
        else:
            model(*inputs)

    total_flops = flop_counter.get_total_flops()
    if is_train:
        model.train()
    return total_flops

def find_folder_upward(folder_name, start_path=None):
    """
    Search backward through parent directories until finding the requested folder.

    Args:
        folder_name: Name of the folder to find
        start_path: Starting directory (defaults to current working directory)

    Returns:
        Path object of the found folder, or None if not found
    """
    if start_path is None:
        current_path = Path.cwd()
    else:
        current_path = Path(start_path).resolve()

    # Check current directory and all parents
    for parent in [current_path] + list(current_path.parents):
        target = parent / folder_name
        if target.exists() and target.is_dir():
            return target

        # Stop at filesystem root
        if parent == parent.parent:
            break

    return None

def save_audio_files(output_audio, prediction_audio, model_path, prefix, sample_rate=48000):
    """
    Save audio files in WAV format.

    Parameters:
        output_audio (np.ndarray): Output audio data array (processed).
        prediction_audio: Predicted labels or values (could be additional info to save).
        model_path (str): The path where to save the audio files (should exist).
    """
    # Create the model path directory if it doesn't exist
    os.makedirs(model_path, exist_ok=True)

    # Saving output audio
    output_file_path = os.path.join(model_path, prefix + '_output_audio.wav')
    output_audio = np.array(output_audio.squeeze(), dtype=np.float32)
    write(output_file_path, sample_rate, output_audio)  # Scale to int16

    # Saving output audio
    output_file_path = os.path.join(model_path, prefix + '_prediction_audio.wav')
    prediction_audio = np.array(prediction_audio.squeeze(), dtype=np.float32)
    write(output_file_path, sample_rate, prediction_audio)  # Scale to int16

    print(f"Audio files saved to {model_path}")

def natural_sort_key(s):
    """
    Function to use as a key for sorting strings in natural order.
    This ensures that strings with numbers are sorted in human-expected order.
    For example: ["file1", "file10", "file2"] -> ["file1", "file2", "file10"]

    Args:
        s: The string to convert to a natural sort key

    Returns:
        A list of string and integer parts that can be used for natural sorting
    """
    # Split the string into text and numeric parts
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]

def compute_lcm(x, y):
    """Compute the least common multiple of two numbers."""
    return (x * y) // math.gcd(x, y)

# json functionalities
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj) -> json.JSONEncoder:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def write_json(data: dict, out_path: Path, jsonl: bool = True) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)  # Create parent directories
    with open(str(out_path), "w", encoding="utf-8") as outputFile:
        if not jsonl:
            json.dump(data, outputFile, cls=NumpyEncoder, indent=4)
        else:
            for item in data:
                outputFile.write(json.dumps(item) + "\n")


import os
import random
import shutil
import sys
from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from pydub import AudioSegment
from tensorflow.python.keras import backend as K
from torch.utils.data import Dataset

from audio import read_mfcc
from batcher import sample_from_mfcc
from constants import NUM_FRAMES, SAMPLE_RATE
from conv_models import DeepSpeakerModel

AUDIO_PATH = "audio"
SOURCE_DIR = "accents"  #'sherlock_holmes'
VERBOSE = False

# ### First, trim large audio segments into 1 second clips
def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)


def split_audio(input_path: str, length=1, filetype=".wav"):
    """given a directory containing several long audio files, trim every
    clip into 1 second incremenets and output to a folder

    """
    # first delete all existing audio at location
    cleaned_name = input_path.split("/")[-1]
    split_path = f"{AUDIO_PATH}/{cleaned_name}_split"
    shutil.rmtree(split_path, ignore_errors=True)
    Path(split_path).mkdir(parents=True, exist_ok=True)

    inputs = [f for f in os.listdir(input_path) if f != ".DS_Store"]

    error_count = 0
    for i, input_name in enumerate(inputs):
        path = f"{input_path}/{input_name}"
        name = input_name.replace(filetype, "")

        try:
            audio = AudioSegment.from_file(path)
        except Exception as e:
            print(f"WARN: Unable to process {input_name}/{name}")
            continue

        # measured in ms
        audio_length_s = len(audio) / 1000

        # split clips into 10s clips
        num_clips = ceil(audio_length_s / length)

        trimmed_clips = []
        prev_end = 0
        for clip_num in range(1, num_clips):
            trimmed_clip = audio[prev_end : clip_num * length * 1_000]
            trimmed_clips.append(trimmed_clip)
            prev_end = clip_num * length * 1_000

        path = f"{AUDIO_PATH}/{cleaned_name}_split/{name}"
        clean_path = path.replace(".mp3", "")
        clean_path = clean_path.replace(".wav", "")
        Path(clean_path).mkdir(parents=True, exist_ok=True)

        for clip_num, clip in enumerate(trimmed_clips):
            clip = match_target_amplitude(clip, -20.0)
            clip.export(f"{clean_path}/{clip_num}.wav", format="wav")

        # reduce verbosity
        if i % 50 == 0 or VERBOSE:
            print(
                f"INFO: Completed clip generation for speaker {input_name} [{i + 1}/{len(inputs)}]"
            )


def generate_mfcc_for_dir(split_path):
    mfcc_path = split_path.replace("split", "mfcc")
    shutil.rmtree(mfcc_path, ignore_errors=True)
    Path(mfcc_path).mkdir(parents=True, exist_ok=True)

    speakers = os.listdir(split_path)
    speakers = [s for s in speakers if s != ".DS_Store"]

    for i, speaker in enumerate(speakers):
        speaker_path = f"{split_path}/{speaker}"
        speaker_mfcc_path = f"{mfcc_path}/{speaker}"
        Path(speaker_mfcc_path).mkdir(parents=True, exist_ok=True)
        clips = [c for c in os.listdir(speaker_path) if c != ".DS_Store"]
        for clip in clips:
            # dimensions of full_mfcc are ~(120, 64), so we are not losing any data when padding
            try:
                full_mfcc = read_mfcc(f"{speaker_path}/{clip}", SAMPLE_RATE)
                sampled_mfcc = sample_from_mfcc(full_mfcc, NUM_FRAMES)

                clip_name = clip.replace(".wav", ".npy")
                np.save(f"{speaker_mfcc_path}/{clip_name}", sampled_mfcc)
            except Exception as e:
                print(f"WARN: Error generating MFCC for speaker {speaker} clip {clip}")

        if i % 50 == 0 or VERBOSE:
            print(
                f"INFO: Completed MFCC generation for speaker {speaker} [{i + 1}/{len(speakers)}]"
            )


def generate_features(mfcc_path):
    feature_path = mfcc_path.replace("mfcc", "features")
    Path(feature_path).mkdir(parents=True, exist_ok=True)

    speakers = os.listdir(mfcc_path)
    speakers = [s for s in speakers if s != ".DS_Store"]

    for i, speaker in enumerate(speakers):
        speaker_path = f"{mfcc_path}/{speaker}"
        speaker_feature_path = f"{feature_path}/{speaker}"
        Path(speaker_feature_path).mkdir(parents=True, exist_ok=True)
        mfcc_files = [m for m in os.listdir(speaker_path) if m != ".DS_Store"]

        # there may already be some features generated. make sure we don't
        # redo work!
        current_features = os.listdir(speaker_feature_path)
        if len(current_features) > 0:
            print(
                f"INFO: Speaker {speaker} already has {len(current_features)} features. Continuing..."
            )

        mfccs_to_predict = [
            mfcc_file for mfcc_file in mfcc_files if mfcc_file not in current_features
        ]
        mfccs_to_predict.sort()

        for mfcc_idx, mfcc_file in enumerate(mfccs_to_predict):
            mfcc = np.load(f"{speaker_path}/{mfcc_file}")

            # generate prediction from this mfcc (TODO: Data augmentation step)
            features = model.m.predict(np.expand_dims(mfcc, axis=0))

            # this outputs a tensor
            # features = model.m(np.expand_dims(mfcc, axis=0))

            # features comes in as (1, 512) for some reason
            features = features[0]

            # we are saving another npy file, so we can reuse the same name because
            # we're saving to a different directory
            np.save(f"{speaker_feature_path}/{mfcc_file}", features)
            # print(f'INFO: Wrote prediction {mfcc_file} for {speaker}: [{mfcc_idx + 1}/{len(mfccs_to_predict)}]')

        if i % 50 == 0 or VERBOSE:
            print(
                f"INFO: Completed feature generation for speaker {speaker} [{i + 1}/{len(speakers)}]"
            )


if __name__ == "__main__":
    split_audio(f"{AUDIO_PATH}/{SOURCE_DIR}", length=1.5)

    generate_mfcc_for_dir(f"{AUDIO_PATH}/{SOURCE_DIR}_split")

    model = DeepSpeakerModel()
    model.m.load_weights("ResCNN_triplet_training_checkpoint_265.h5", by_name=True)

    tf.executing_eagerly()

    generate_features(f"{AUDIO_PATH}/{SOURCE_DIR}_mfcc")

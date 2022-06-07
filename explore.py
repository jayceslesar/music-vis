"""Evolving Functions for EDA + Prototyping."""

import os
import time
import argparse

import numpy as np
import librosa
import vlc

from demucs import separate

from visualizers import BarVisualizer

import warnings
warnings.filterwarnings("ignore")


SAMPLING_RATE = 44_100


def pre_process(song_path: str, needs_split: bool) -> tuple:
    data, sampling_rate = librosa.load(song_path, sr=SAMPLING_RATE)
    duration = librosa.get_duration(y=data, sr=sampling_rate)

    tempo, beats = librosa.beat.beat_track(y=data, sr=sampling_rate)
    beat_timestamps = librosa.frames_to_time(beats, sr=sampling_rate)

    if needs_split:
        parser = argparse.ArgumentParser()
        separate.add_arguments(parser)
        separate_args = parser.parse_args([
            song_path,
            '--mp3',
            '-j', str(os.cpu_count())
        ])
        separate.main(separate_args)

    return data, sampling_rate, duration, tempo, beat_timestamps


def pre_process_components(splits_path: str) -> dict:
    out = {}
    for component in os.listdir(splits_path):
        component_path = os.path.join(splits_path, component)
        data, sampling_rate = librosa.load(component_path, sr=SAMPLING_RATE)
        S = librosa.magphase(librosa.stft(data, window=np.ones))[0]
        rms = librosa.feature.rms(S=S)
        times = (librosa.times_like(rms)/2)[1:]
        diff = rms[0]
        out[component] = {
            'rms_differential': np.abs(diff),
        }
        out['timestamps'] = times

    return out


def get_component_value(data: dict, key: str, component: str, index: int) -> float:
    return data[component][key][index]


def main():
    # preprocess everything
    data_dir = 'data'
    song_name = 'perfume_del-water-gap.mp3'
    song = os.path.join(data_dir, song_name)
    os.makedirs(data_dir, exist_ok=True)
    expected_splits = ['bass', 'drums', 'other', 'vocals']
    actual_split_counts = 0
    split_files = os.path.join('separated', 'mdx_extra_q', os.path.splitext(song_name)[0])
    for split_file in os.listdir(split_files):
        if os.path.splitext(split_file)[0] in expected_splits:
            actual_split_counts += 1

    needs_split = False
    if not actual_split_counts == len(expected_splits):
        print('Missing at least 1 split track.')
        needs_split = True
    data, sampling_rate, duration, tempo, beat_timestamps = pre_process(song, needs_split)

    component_data = pre_process_components(split_files)
    print('Done Preprocessing.')

    barcolors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 255)]
    viz = BarVisualizer(song, data, sampling_rate, duration, beat_timestamps, component_data, barcolors=barcolors)
    viz.play()


if __name__ == '__main__':
    main()

"""Visualizer Classes."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple


import vlc
import time
import pygame
import numpy as np


class Visualizer(ABC):
    """Abstract visualizer class."""

    def __init__(self, song_path: str, song_data: list[float], sampling_rate: int,
                 duration: float, beat_timestamps: list[float], channels_data: dict[str, list[float]]) -> None:
        """Create an instance of an abstract visualizer.

        Args:
            song_path: path to mp3 file of song
            song_data: flat list of song signals
            sampling_rate: the sample we processed this file at
            duration: length in seconds of the song
            beat_timestamps: timestamps of what librosa determined to be the beat
            channels_data: {channel_name: [channel RMS values]}
        """
        self.song_path = song_path
        self.song_data = song_data
        self.sampling_rate = sampling_rate
        self.duration = duration
        self.beat_timestamps = beat_timestamps
        self.channels_timestamps = channels_data.pop('timestamps').tolist()
        self.channels_data = channels_data
        self.current_beat_index = 0

    def play(self) -> None:
        """Handle the play logic for all visualizer classes."""
        media_player = vlc.MediaPlayer()
        media = vlc.Media(self.song_path)

        media_player.set_media(media)
        media_player.audio_set_volume(100)
        media_player.play()

        previous_timestamp_index = 0

        start_time = time.monotonic()
        while True:
            now = time.monotonic()
            elapsed = now - start_time

            if elapsed >= self.duration:
                break

            is_beat = np.isclose(elapsed, self.beat_timestamps[self.current_beat_index], rtol=0.01)

            current_timestamp = min(self.channels_timestamps, key=lambda x: abs(x - elapsed))
            current_timestamp_index = self.channels_timestamps.index(current_timestamp)
            if previous_timestamp_index != current_timestamp_index:
                component_rms_values = {component: self.get_component_value(self.channels_data, 'rms_differential', component, current_timestamp_index) for component in self.channels_data}
                self.render_channels(component_rms_values)

            if is_beat:
                self.render_beat()
                self.current_beat_index += 1

            previous_timestamp_index = current_timestamp_index

    @abstractmethod
    def render_beat(self) -> None:
        """Abstract method for derived classes to render a beat."""
        raise NotImplementedError

    @abstractmethod
    def render_channels(self, component_rms_values: dict[str, float], is_beat: bool) -> None:
        """Abstract method for derived class to render the channels while playing.

        Args:
            component_rms_values: rms values to use to visualize heights
            is_beatt: is this a beat
        """
        raise NotImplementedError

    @staticmethod
    def get_component_value(data: dict[str, list[float]], key: str, component: str, index: int) -> float:
        """Helper function to get the correct component data for a given channel.

        Args:
            data: {channel_name: [channel RMS values]}
            key: channel name
            component: component in channel
            index: index to pull

        Returns:
            float: value at the indexed keys
        """
        return data[component][key][index]

    @staticmethod
    def get_random_fill() -> tuple[int, int, int]:
        """Get a random RGB color.

        Returns:
            random RGB color
        """
        return (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))


class BarVisualizer(Visualizer):
    """Bar Visualizer class."""

    HEIGHT = 1000
    WIDTH = 1200
    MAX_BAR_HEIGHT = 700

    BAR_GAP_PROPORTION = 0.1
    NUM_BARS = 4

    def __init__(self, *args: list[Any], barcolors: list[str]) -> None:
        """Initialize a BarVisualizer.

        Args:
            barcolors (List[str]): colors for the bars
        """
        super().__init__(*args)
        if len(self.channels_data) != len(barcolors):
            raise ValueError('Number of channels and number of barcolors must match.')
        self.barcolors = barcolors

        pygame.init()
        self.display = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        self.display.fill((0, 0, 0))
        self.bar_positions = []
        self.previous_heights = [0, 0, 0, 0]
        self.channel_minmax_linspace = {}
        for channel in self.channels_data:
            self.channel_minmax_linspace[channel] = [
                np.linspace(min(self.channels_data[channel]['rms_differential']), max(self.channels_data[channel]['rms_differential']), self.MAX_BAR_HEIGHT)
            ]

        self.font = pygame.font.Font(pygame.font.get_default_font(), 30)


        bar_space_total_width = int(self.WIDTH - (self.WIDTH * self.BAR_GAP_PROPORTION))
        bar_gap_total_width = self.WIDTH - bar_space_total_width
        self.bar_width = int(bar_space_total_width / self.NUM_BARS)
        self.bar_gap_width = int(bar_gap_total_width / (self.NUM_BARS + 1))

        current_bar_gap = self.bar_gap_width
        for i in range(self.NUM_BARS):
            pygame.draw.rect(self.display, self.barcolors[i], pygame.Rect(current_bar_gap, 100, self.bar_width, 0))
            self.bar_positions.append(current_bar_gap)
            pygame.display.flip()
            current_bar_gap += self.bar_width + self.bar_gap_width

    def render_channels(self, values: dict[str, float]) -> None:
        redraw = False

        for i, channel in enumerate(values):
            linspace = self.channel_minmax_linspace[channel]

            # find the closest linearly interpolated value from the min/max RMS diff of this channel
            scaled_height = np.argmin(np.abs(linspace - values[channel]))

            diff = scaled_height - self.previous_heights[i]
            next_height = int(self.previous_heights[i] + (diff / 15))
            if scaled_height < self.previous_heights[i]:
                redraw = True

            if next_height > self.MAX_BAR_HEIGHT:
                next_height = self.MAX_BAR_HEIGHT
            if next_height < 0:
                next_height = 0

            pygame.draw.rect(self.display, self.barcolors[i], pygame.Rect(self.bar_positions[i], self.HEIGHT-next_height, self.bar_width, self.HEIGHT))
            self.previous_heights[i] = next_height

        if redraw:
            pygame.display.flip()
            self.display.fill((0, 0, 0))

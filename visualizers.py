"""Visualizer Classes."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple


import vlc
import time
import pygame
import numpy as np


class Visualizer(ABC):

    def __init__(self, song_path: str, song_data: List[float], sampling_rate: int,
                 duration: float, beat_timestamps: List[float], channels_data: Dict[str, List[float]]) -> None:
        self.song_path = song_path
        self.song_data = song_data
        self.sampling_rate = sampling_rate
        self.duration = duration
        self.beat_timestamps = beat_timestamps
        self.channels_timestamps = channels_data.pop('timestamps').tolist()
        self.channels_data = channels_data
        self.current_beat_index = 0

    def play(self) -> None:
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

            # if is_beat:
            #     self.render_beat()

            previous_timestamp_index = current_timestamp_index

    @abstractmethod
    def render_beat(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def render_channels(self, component_rms_values: Dict[str, float], is_beat: bool) -> None:
        raise NotImplementedError

    @staticmethod
    def get_component_value(data: dict, key: str, component: str, index: int) -> float:
        return data[component][key][index]

    @staticmethod
    def get_random_fill() -> Tuple[int, int, int]:
        return (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))


class BarVisualizer(Visualizer):
    HEIGHT = 600
    WIDTH = 1400
    MAX_BAR_HEIGHT = 500
    BAR_WIDTH = 150
    NUM_BARS = 4
    DEFAULT_HEIGHT = 50
    BAR_GAP = (WIDTH / NUM_BARS) / (NUM_BARS + 1)

    def __init__(self, *args: List[Any], barcolors: List[str], background: Optional[str] = None) -> None:
        super().__init__(*args)
        if len(self.channels_data) != len(barcolors):
            raise ValueError('Number of channels and number of barcolors must match.')
        self.barcolors = barcolors
        self.background = background

        pygame.init()
        self.display = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        self.display.fill((255, 255, 255))
        self.bar_positions = []
        self.previous_heights = [0, 0, 0, 0]
        self.channel_minmax_linspace = {}
        for channel in self.channels_data:
            self.channel_minmax_linspace[channel] = [
                np.linspace(min(self.channels_data[channel]['rms_differential']), max(self.channels_data[channel]['rms_differential']), self.MAX_BAR_HEIGHT)
            ]

        self.font = pygame.font.Font(pygame.font.get_default_font(), 30)

        current_bar_gap = self.BAR_GAP
        for i in range(self.NUM_BARS):
            pygame.draw.rect(self.display, self.barcolors[i], pygame.Rect(current_bar_gap, 100, self.BAR_WIDTH, self.DEFAULT_HEIGHT))
            self.bar_positions.append(current_bar_gap)
            pygame.display.flip()
            text_surface = self.font.render(list(self.channels_data.keys())[i], True, (0, 0, 0))
            self.display.blit(text_surface, dest=(current_bar_gap, 50))
            current_bar_gap += self.BAR_WIDTH + self.BAR_GAP

    def render_channels(self, values: Dict[str, float]) -> None:
        redraw = False
        for i, channel in enumerate(values):
            linspace = self.channel_minmax_linspace[channel]

            # find the closest linearly interpolated value from the min/max RMS diff of this channel
            scaled_height = np.argmin(np.abs(linspace - values[channel]))

            diff = scaled_height - self.previous_heights[i]
            next_height = int(self.previous_heights[i] + (diff / 10))
            if scaled_height < self.previous_heights[i]:
                redraw = True

            if next_height > self.MAX_BAR_HEIGHT:
                next_height = self.MAX_BAR_HEIGHT
            if next_height < 0:
                next_height = 0

            pygame.draw.rect(self.display, self.barcolors[i], pygame.Rect(self.bar_positions[i], self.HEIGHT-next_height, self.BAR_WIDTH, self.HEIGHT))
            self.previous_heights[i] = next_height

        if redraw:
            pygame.display.flip()
            self.display.fill((255, 255, 255))
            for i, channel in enumerate(self.channels_data):
                text_surface = self.font.render(channel, True, (0, 0, 0))
                self.display.blit(text_surface, dest=(self.bar_positions[i], 50))

    def render_beat(self) -> None:
        self.display.fill(self.get_random_fill())
        pygame.display.flip()
        self.current_beat_index += 1


class CircleVisualizer(Visualizer):
    pass

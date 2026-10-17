"""Small simulated optical bench for Homework 2, Task 2.

The simulator follows the lecture model

    E_out = T @ E_in
    I_out = |E_out|**2

The DMD has 16 x 16 amplitude controls and the camera has 64 x 64 pixels.
This file only implements the bench and noisy measurements. The optimization
loop that searches for a focus is added after this component is verified.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class Region:
    """A rectangular camera region using half-open array bounds."""

    row_start: int
    row_stop: int
    col_start: int
    col_stop: int

    def extract(self, image: Array) -> Array:
        return image[self.row_start:self.row_stop, self.col_start:self.col_stop]


@dataclass(frozen=True)
class BenchConfig:
    dmd_shape: tuple[int, int] = (16, 16)
    camera_shape: tuple[int, int] = (64, 64)
    target_shape: tuple[int, int] = (5, 5)
    noise_fraction: float = 0.02
    seed: int = 42


@dataclass
class Measurement:
    intensity: Array
    target: Region
    backgrounds: tuple[Region, Region, Region, Region]

    @property
    def target_mean(self) -> float:
        return float(self.target.extract(self.intensity).mean())

    @property
    def background_mean(self) -> float:
        values = [region.extract(self.intensity).mean() for region in self.backgrounds]
        return float(np.mean(values))

    @property
    def enhancement(self) -> float:
        return self.target_mean / max(self.background_mean, np.finfo(float).eps)


class SimulatedBench:
    """Deterministic optical bench with stochastic noisy measurements."""

    def __init__(self, config: BenchConfig = BenchConfig()) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self._validate_config()
        self.transmission = self._make_transmission_matrix()
        self.target, self.backgrounds = self._make_regions()

    @property
    def dmd_size(self) -> int:
        return int(np.prod(self.config.dmd_shape))

    @property
    def camera_size(self) -> int:
        return int(np.prod(self.config.camera_shape))

    def gaussian_beam(self) -> Array:
        """Return a normalized Gaussian amplitude on the DMD."""
        rows, cols = self.config.dmd_shape
        y, x = np.mgrid[0:rows, 0:cols]
        cy, cx = (np.array(self.config.dmd_shape) - 1) / 2.0
        sigma = min(rows, cols) / 3.0
        beam = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma**2))
        return (beam / beam.max()).astype(np.float64)

    def measure(self, dmd_amplitude: Array, *, noisy: bool = True) -> Measurement:
        """Propagate one DMD amplitude pattern and return a camera measurement."""
        pattern = np.asarray(dmd_amplitude, dtype=np.float64)
        if pattern.shape != self.config.dmd_shape:
            raise ValueError(f"Expected DMD shape {self.config.dmd_shape}, got {pattern.shape}")
        if not np.all(np.isfinite(pattern)) or np.any(pattern < 0) or np.any(pattern > 1):
            raise ValueError("DMD amplitude values must be finite and lie in [0, 1].")

        field = self.transmission @ pattern.reshape(-1)
        intensity = np.abs(field.reshape(self.config.camera_shape)) ** 2
        if noisy:
            sigma = self.config.noise_fraction * max(float(intensity.mean()), np.finfo(float).eps)
            intensity = intensity + self.rng.normal(0.0, sigma, intensity.shape)
            intensity = np.clip(intensity, 0.0, None)
        return Measurement(intensity=intensity, target=self.target, backgrounds=self.backgrounds)

    def _validate_config(self) -> None:
        if self.config.noise_fraction < 0:
            raise ValueError("noise_fraction must be non-negative")
        tr, tc = self.config.target_shape
        cr, cc = self.config.camera_shape
        if min(tr, tc, *self.config.dmd_shape) < 1 or 5 * max(tr, tc) > min(cr, cc):
            raise ValueError("Positive dimensions and space for adjacent backgrounds required.")
        if tr > cr or tc > cc:
            raise ValueError("Target region must fit inside the camera.")

    def _make_transmission_matrix(self) -> Array:
        """Create a fixed complex random transmission matrix for this run."""
        real = self.rng.normal(size=(self.camera_size, self.dmd_size))
        imaginary = self.rng.normal(size=(self.camera_size, self.dmd_size))
        return (real + 1j * imaginary) / np.sqrt(2.0 * self.dmd_size)

    def _make_regions(self) -> tuple[Region, tuple[Region, Region, Region, Region]]:
        """Choose a random target with four non-overlapping adjacent regions."""
        rows, cols = self.config.camera_shape
        tr, tc = self.config.target_shape
        margin = 2 * max(tr, tc)
        row = int(self.rng.integers(margin, rows - margin - tr + 1))
        col = int(self.rng.integers(margin, cols - margin - tc + 1))
        target = Region(row, row + tr, col, col + tc)
        backgrounds = (
            Region(row - tr, row, col, col + tc),
            Region(row + tr, row + 2 * tr, col, col + tc),
            Region(row, row + tr, col - tc, col),
            Region(row, row + tr, col + tc, col + 2 * tc),
        )
        return target, backgrounds

"""Synthetic seismic data support for TAPIR training."""

from tapnet.seismic.synthetic import SyntheticSeismicConfig
from tapnet.seismic.synthetic import generate_synthetic_sample
from tapnet.seismic.synthetic import iter_synthetic_samples

__all__ = [
    'SyntheticSeismicConfig',
    'generate_synthetic_sample',
    'iter_synthetic_samples',
]

"""Defaults and validation shared by streaming training and deployment."""
import math

DEFAULT_FS = 4000.
DEFAULT_MAX_SAMPLES = 7500
DEFAULT_SKIP = 1500
DEFAULT_MIN_SAMPLES = 432
DEFAULT_STRIDE = 48
DEFAULT_BUCKET_SAMPLES = 768


def sampling_rate(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError('sampling rate (--fs) must be finite and positive')
    return value


def bucket_count(max_samples, skip=DEFAULT_SKIP, min_samples=DEFAULT_MIN_SAMPLES,
                 bucket_samples=DEFAULT_BUCKET_SAMPLES):
    """Ordinary intervals before the forced raw endpoint; exclude the forced endpoint itself."""
    if int(max_samples) != max_samples or max_samples <= skip + min_samples:
        raise ValueError(f'decision endpoint (--max-samples) must exceed {skip + min_samples} samples')
    return (int(max_samples) - skip - min_samples + bucket_samples - 1)//bucket_samples

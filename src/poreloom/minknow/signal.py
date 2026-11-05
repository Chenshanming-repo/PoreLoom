"""Per-channel mapping from device ADC to the int16 ADC grid on which the live policy buffers reads.

The policy converts buffered samples to picoamperes with one fixed contract (``offset``, ``range``,
``digitisation``). Each channel's device calibration is folded into a gain and shift so that
``(adc_model + offset) * range / digitisation`` equals the device picoamperes up to half a model ADC step. With
the same calibration on every channel this is exactly the mapping of the Icarust client
(experiments/20261011_simloop/closed_loop.py).
"""
import numpy as np

INT16 = np.iinfo(np.int16)


def adc_mapping(device, model):
    """device: dict(offsets=[C], pa_ranges=[C], digitisation); model: dict(offset, range, digitisation).
    Returns (gain [C], shift [C]) with adc_model = adc_device * gain + shift."""
    offsets = np.asarray(device['offsets'], np.float64)
    ranges = np.asarray(device['pa_ranges'], np.float64)
    gain = (ranges/float(device['digitisation']))/(float(model['range'])/float(model['digitisation']))
    shift = offsets*gain - float(model['offset'])
    return gain, shift


def to_model_adc(raw, gain, shift):
    """Device int16 samples of one channel -> model-grid int16 (rounded, saturated)."""
    x = np.rint(np.asarray(raw, np.float64)*gain + shift)
    return np.clip(x, INT16.min, INT16.max).astype(np.int16)

from __future__ import annotations


def calculate_phog_bins(base_value: int, level: int) -> int:
    """Calculate Σ_{k=0..level-1} base_value * 4^k."""
    sum_series = 0
    for k in range(int(level)):
        sum_series += int(base_value) * (4**k)
    return sum_series

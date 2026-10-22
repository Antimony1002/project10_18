from __future__ import annotations

import numpy as np


def validate_timestamps(timestamps_ns: np.ndarray, label: str) -> np.ndarray:
    timestamps = np.asarray(timestamps_ns, dtype=np.int64)
    if timestamps.ndim != 1 or timestamps.size == 0:
        raise ValueError(f"{label}: timestamps must be a non-empty 1-D array")
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError(f"{label}: timestamps must be strictly increasing")
    return timestamps


def nearest_indices(
    source_timestamps_ns: np.ndarray,
    query_timestamps_ns: np.ndarray,
    max_skew_ns: int,
    label: str,
) -> np.ndarray:
    source = validate_timestamps(source_timestamps_ns, label)
    query = np.asarray(query_timestamps_ns, dtype=np.int64)
    right = np.searchsorted(source, query, side="left")
    right = np.clip(right, 0, source.size - 1)
    left = np.clip(right - 1, 0, source.size - 1)
    choose_right = np.abs(source[right] - query) < np.abs(source[left] - query)
    indices = np.where(choose_right, right, left)
    skew = np.abs(source[indices] - query)
    if np.any(skew > max_skew_ns):
        worst = float(skew.max()) / 1e6
        raise ValueError(f"{label}: nearest-sample skew {worst:.2f} ms exceeds limit")
    return indices.astype(np.int64)


def interpolate_stream(
    source_timestamps_ns: np.ndarray,
    values: np.ndarray,
    query_timestamps_ns: np.ndarray,
    max_skew_ns: int,
    label: str,
) -> np.ndarray:
    source = validate_timestamps(source_timestamps_ns, label)
    values = np.asarray(values)
    query = np.asarray(query_timestamps_ns, dtype=np.int64)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape[0] != source.size:
        raise ValueError(f"{label}: values must have shape (samples, dimensions)")
    if query.size == 0 or query.min() < source[0] or query.max() > source[-1]:
        raise ValueError(f"{label}: query timestamps fall outside the source stream")
    nearest_indices(source, query, max_skew_ns, label)
    relative_source = (source - source[0]).astype(np.float64)
    relative_query = (query - source[0]).astype(np.float64)
    result = np.column_stack(
        [np.interp(relative_query, relative_source, values[:, dim]) for dim in range(values.shape[1])]
    )
    return result.astype(np.float32)

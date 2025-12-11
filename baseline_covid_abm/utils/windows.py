from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd
from epiweeks import Week


@dataclass(frozen=True)
class WindowSpec:
    """Specification for a simulation/calibration window."""

    label: str
    offset: int
    start_week: int


def _validate_inputs(
    daily_path: Path, start_date: pd.Timestamp, end_date: pd.Timestamp, window_days: int
) -> pd.DataFrame:
    if window_days <= 0:
        raise ValueError("window_days must be positive")
    if not daily_path.exists():
        raise FileNotFoundError(f"Daily data file {daily_path} not found.")

    df = pd.read_csv(daily_path, parse_dates=["date"])
    if df.empty:
        raise ValueError(f"{daily_path} is empty.")
    df = df.sort_values("date").reset_index(drop=True)

    first_date = df.loc[0, "date"]
    last_date = df.loc[len(df) - 1, "date"]
    if start_date < first_date or end_date > last_date:
        raise ValueError(
            f"Requested range {start_date.date()}–{end_date.date()} is outside "
            f"{daily_path} coverage ({first_date.date()}–{last_date.date()})."
        )
    if start_date > end_date:
        raise ValueError("start_date must be earlier than end_date.")

    return df


def derive_windows(
    daily_path: Path,
    start_date: str,
    end_date: str,
    window_days: int,
    stride_days: int | None = None,
) -> List[WindowSpec]:
    """
    Construct sequential windows across the requested date range.

    Args:
        daily_path: CSV with columns [date,cases,...].
        start_date: Inclusive start date (YYYY-MM-DD).
        end_date: Inclusive end date (YYYY-MM-DD).
        window_days: Simulation horizon in days (typically num_steps_per_episode).
        stride_days: Optional stride between windows. Defaults to window_days but
            the tail window is always added so the final day <= end_date is covered.
    """

    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)
    df = _validate_inputs(daily_path, start_ts, end_ts, window_days)

    stride = stride_days or window_days
    if stride <= 0:
        raise ValueError("stride_days must be positive.")

    in_range_mask = (df["date"] >= start_ts) & (df["date"] <= end_ts)
    indices = df.index[in_range_mask]
    if len(indices) == 0:
        raise ValueError(
            f"No rows between {start_ts.date()} and {end_ts.date()} in {daily_path}."
        )
    range_start = indices.min()
    range_end = indices.max()

    offsets: list[int] = []
    pointer = range_start
    while pointer + window_days - 1 <= range_end:
        offsets.append(pointer)
        pointer += stride

    last_possible_start = range_end - window_days + 1
    if last_possible_start >= range_start:
        if not offsets or offsets[-1] < last_possible_start:
            offsets.append(last_possible_start)

    windows: list[WindowSpec] = []
    for offset in offsets:
        start_day = df.loc[offset, "date"]
        end_day = df.loc[offset + window_days - 1, "date"]
        start_week = int(str(Week.fromdate(start_day)))
        end_week = int(str(Week.fromdate(end_day)))
        label = f"{start_week}-{end_week}"
        windows.append(WindowSpec(label=label, offset=int(offset), start_week=start_week))

    return windows



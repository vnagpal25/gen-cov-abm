"""
Bar plots comparing pre- vs post-genomics MAE for each county and window.

Data are hard-coded from the experiment tables supplied by the user. For
each window we intersect the counties that appear in both the pre- and
post-genomics tables to avoid plotting incomplete pairs.
"""

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


CountyRecord = Dict[str, Dict[str, float]]
WindowData = Dict[str, Dict[str, CountyRecord]]

WINDOW_DATA: WindowData = {
    "202027-202031": {
        "pre": {
            "25001": {"county": "Barnstable", "MAE": 42.2857},
            "25003": {"county": "Berkshire", "MAE": 64.2571},
            "25005": {"county": "Bristol", "MAE": 89.2857},
            "25009": {"county": "Essex", "MAE": 78.1714},
            "25013": {"county": "Hampden", "MAE": 82.4000},
            "25021": {"county": "Norfolk", "MAE": 67.4571},
            "25023": {"county": "Plymouth", "MAE": 55.0857},
            "25025": {"county": "Suffolk", "MAE": 73.6000},
            "25027": {"county": "Worcester", "MAE": 66.2286},
        },
        "post": {
            "25001": {"county": "Barnstable", "MAE": 63.7429},
            "25003": {"county": "Berkshire", "MAE": 101.9143},
            "25005": {"county": "Bristol", "MAE": 99.2571},
            "25009": {"county": "Essex", "MAE": 86.0000},
            "25013": {"county": "Hampden", "MAE": 101.8571},
            "25015": {"county": "Hampshire", "MAE": 94.3429},
            "25021": {"county": "Norfolk", "MAE": 82.9429},
            "25023": {"county": "Plymouth", "MAE": 69.4286},
            "25025": {"county": "Suffolk", "MAE": 89.9714},
            "25027": {"county": "Worcester", "MAE": 72.6571},
        },
    },
    "202031-202035": {
        "pre": {
            "25001": {"county": "Barnstable", "MAE": 37.1714},
            "25003": {"county": "Berkshire", "MAE": 61.7714},
            "25005": {"county": "Bristol", "MAE": 73.6857},
            "25009": {"county": "Essex", "MAE": 87.3143},
            "25013": {"county": "Hampden", "MAE": 76.0571},
            "25021": {"county": "Norfolk", "MAE": 45.7714},
            "25023": {"county": "Plymouth", "MAE": 59.8857},
            "25025": {"county": "Suffolk", "MAE": 91.8000},
            "25027": {"county": "Worcester", "MAE": 53.0000},
        },
        "post": {
            "25001": {"county": "Barnstable", "MAE": 57.9429},
            "25003": {"county": "Berkshire", "MAE": 101.4000},
            "25005": {"county": "Bristol", "MAE": 90.2571},
            "25009": {"county": "Essex", "MAE": 95.7143},
            "25013": {"county": "Hampden", "MAE": 108.2571},
            "25015": {"county": "Hampshire", "MAE": 94.9143},
            "25021": {"county": "Norfolk", "MAE": 63.2571},
            "25023": {"county": "Plymouth", "MAE": 78.2571},
            "25025": {"county": "Suffolk", "MAE": 112.3429},
            "25027": {"county": "Worcester", "MAE": 67.4571},
        },
    },
}

OUTPUT_DIR = Path("Results/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _intersected_fips(window: str) -> List[str]:
    """Return FIPS present in both pre and post sets, sorted by FIPS."""
    pre_fips = set(WINDOW_DATA[window]["pre"].keys())
    post_fips = set(WINDOW_DATA[window]["post"].keys())
    return sorted(pre_fips & post_fips)


def plot_window(window: str) -> None:
    """Render MAE comparison bars for a single window."""
    fips_codes = _intersected_fips(window)
    pre = WINDOW_DATA[window]["pre"]
    post = WINDOW_DATA[window]["post"]

    labels = [pre[fips]["county"] for fips in fips_codes]
    pre_vals = [pre[fips]["MAE"] for fips in fips_codes]
    post_vals = [post[fips]["MAE"] for fips in fips_codes]

    x = np.arange(len(labels))
    width = 0.35

    color_map = {"pre": "#4C72B0", "post": "#DD8452"}

    fig, ax = plt.subplots(figsize=(12, 5))

    for idx, fips in enumerate(fips_codes):
        pre_val = pre[fips]["MAE"]
        post_val = post[fips]["MAE"]

        if post_val >= pre_val:
            left_key, left_val = "post", post_val
            right_key, right_val = "pre", pre_val
        else:
            left_key, left_val = "pre", pre_val
            right_key, right_val = "post", post_val

        ax.bar(
            x[idx] - width / 2,
            left_val,
            width,
            color=color_map[left_key],
        )

        ax.bar(
            x[idx] + width / 2,
            right_val,
            width,
            color=color_map[right_key],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("MAE")
    ax.set_title(f"Window {window}: County-level MAE (Pre vs. Post Genomics)")
    legend_handles = [
        Patch(color=color_map["pre"], label="Pre-genomics"),
        Patch(color=color_map["post"], label="Post-genomics"),
    ]
    ax.legend(handles=legend_handles)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    out_path = OUTPUT_DIR / f"genomics_mae_{window}.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    for window in WINDOW_DATA:
        plot_window(window)


if __name__ == "__main__":
    main()


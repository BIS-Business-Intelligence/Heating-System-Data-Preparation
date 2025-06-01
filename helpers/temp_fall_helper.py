"""
temp_fall_helper.py
~~~~~~~~~~~~~~~~~~~
Enrich a telemetry dataframe with *temperature-fall* features that help a
classifier recognize night-setback periods.

Public entry point
------------------
    add_temp_fall_features(df: pd.DataFrame, ...)

New columns added (all per-building):
    • Slope_<n>min          – rolling OLS slope  [K/min]
    • DropFromPeak_<h>h     – distance below rolling peak  [K]
    • TempFallFlag          – binary trigger  {0,1,NA}

The slope is calculated with scikit-learn’s ``LinearRegression`` exactly like
the Medium article on rolling linear regression, but wrapped in a tidy helper
so you can adjust the window size without touching the math.
"""

from __future__ import annotations

from typing import List
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


# ──────────────────────────────────────────────────────────────────────────────
#  Internal utility: rolling OLS slope
# ──────────────────────────────────────────────────────────────────────────────

def rolling_ols_slope(series: pd.Series, window: int) -> pd.Series:
    """Return a Series of OLS slopes (K/min) for an *evenly spaced* series.

    Parameters
    ----------
    series : pd.Series
        Numeric values (e.g., SupplyTemp) indexed in a regular cadence.
    window : int
        Number of consecutive samples inside each moving window.

    Notes
    -----
    • The first *window-1* positions are NaN – as expected for a rolling operator.
    • Complexity is O(n · window) – fast enough for typical HVAC data.
    """
    # fixed x-axis 0 … window-1; build once to reuse in every call
    x = np.arange(window).reshape(-1, 1)
    model = LinearRegression()

    def _fit(y: np.ndarray) -> float:
        if np.isnan(y).any():
            return np.nan
        model.fit(x, y.reshape(-1, 1))
        return model.coef_[0, 0]

    return series.rolling(window, min_periods=window).apply(
        lambda y: _fit(y.to_numpy()), raw=False
    )


# ──────────────────────────────────────────────────────────────────────────────
#  Public feature builder
# ──────────────────────────────────────────────────────────────────────────────

def add_temp_fall_features(
    df: pd.DataFrame,
    *,
    resample_freq: str = "1min",       # regular grid to avoid skew
    slope_window_min: int = 20,        # window length for OLS slope
    peak_window_h: int = 4,            # look-back for rolling peak
    slope_thresh: float = -0.02,       # sustained downward slope [K/min]
    drop_thresh: float = 4.0,          # distance below peak [K]
) -> pd.DataFrame:
    """Augment *df* with supply-temperature fall features.

    The input must contain at least `Timestamp`, `Building`, and `SupplyTemp` columns.
    Any extra columns are preserved.

    The function works per building to respect independent timelines and avoids
    accidental interpolation across different assets.
    """

    required_cols = {"Timestamp", "Building", "SupplyTemp"}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"add_temp_fall_features: missing columns: {missing}")

    enriched_chunks: List[pd.DataFrame] = []

    for bldg, grp in df.groupby("Building", sort=False):
        # Sort and drop duplicate timestamps to avoid reindex‐errors
        g = (
            grp.sort_values("Timestamp")
               .drop_duplicates(subset="Timestamp", keep="first")
               .set_index("Timestamp")
        )

        # 1) Regularize time axis and interpolate small gaps
        g = g.resample(resample_freq).asfreq()
        g["SupplyTemp"] = g["SupplyTemp"].interpolate(limit_direction="both")

        # 2) Rolling OLS slope (per-minute change)
        slope_col = f"Slope_{slope_window_min}min"
        g[slope_col] = rolling_ols_slope(g["SupplyTemp"], slope_window_min)

        # 3) Distance below recent rolling peak
        peak_col = f"Peak_{peak_window_h}h"
        g[peak_col] = g["SupplyTemp"].rolling(f"{peak_window_h}h").max()
        drop_col = f"DropFromPeak_{peak_window_h}h"
        g[drop_col] = g[peak_col] - g["SupplyTemp"]

        # 4) Binary flag: sustained slope *and* deep enough drop
        falling_mean = g[slope_col].rolling(30).mean()  # 30-min smoothing
        g["TempFallFlag"] = (
            (falling_mean < slope_thresh) & (g[drop_col] >= drop_thresh)
        ).astype("Int8")  # Int8 keeps {0,1,NA} compact

        # 5) Reshape for merge back to the original granularity
        g = g.reset_index()
        enriched_chunks.append(
            g[["Timestamp", "Building", slope_col, drop_col, "TempFallFlag"]]
        )

    # Combine all buildings, then merge back onto the *original* df
    addon = pd.concat(enriched_chunks, ignore_index=True)

    # <-- NOTE: we drop `validate` so pandas does a standard left‐merge and
    #     does not error out when the raw df has multiple rows per Timestamp+Building.
    enriched_df = df.merge(
        addon,
        on=["Timestamp", "Building"],
        how="left"
    )

    return enriched_df

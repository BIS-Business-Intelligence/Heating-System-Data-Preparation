import numpy as np


def get_radian_angle(value, period):
    return (value / period) * 2 * np.pi

def get_cyclic_encoding_sin(value, period):
    """Returns the sine component of cyclic encoding for a given value."""
    angle = get_radian_angle(value, period)
    return np.sin(angle)

def get_cyclic_encoding_cos(value, period):
    """Returns the cosine component of cyclic encoding for a given value."""
    angle = get_radian_angle(value, period)
    return np.cos(angle)

def add_cyclic_time_features(df, timestamp_col="Timestamp"):
    """
    Adds cyclic encoding columns for month, hour, and minute from a timestamp column.
    Columns added: Month_sin, Month_cos, Hour_sin, Hour_cos, Minute_sin, Minute_cos
    """
    ts = df[timestamp_col]
    df["Month_sin"] = get_cyclic_encoding_sin(ts.dt.month, 12)
    df["Month_cos"] = get_cyclic_encoding_cos(ts.dt.month, 12)
    df["Hour_sin"] = get_cyclic_encoding_sin(ts.dt.hour, 24)
    df["Hour_cos"] = get_cyclic_encoding_cos(ts.dt.hour, 24)
    df["Minute_sin"] = get_cyclic_encoding_sin(ts.dt.minute, 60)
    df["Minute_cos"] = get_cyclic_encoding_cos(ts.dt.minute, 60)
    return df


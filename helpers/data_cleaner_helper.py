import pandas as pd
import csv


# ───────────────────────────────────────────────
#  Generic CSV loader for one measurement column
# ───────────────────────────────────────────────
def load_measurement(file_path: str, new_value_col: str) -> pd.DataFrame:
    """
    Read a CSV produced by Influx|Timescale: columns `_time`, `_value`.
    Rename them and convert both to sane dtypes.
    """

    # Auto-detect delimiter (comma, semicolon, tab)
    with open(file_path, "r") as f:
        sample = f.read(1024)
        f.seek(0)
        delimiter = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t"]).delimiter

    df = pd.read_csv(file_path, sep=delimiter)

    df.rename(columns={"_time": "Timestamp", "_value": new_value_col}, inplace=True)

    # (1) Robust timestamp parsing (mixed ISO strings, always force UTC)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True, format="mixed")

    # (2) Force numeric – non-numeric blobs turn into NaN, which is what we want
    df[new_value_col] = pd.to_numeric(df[new_value_col], errors="coerce")

    return df[["Timestamp", new_value_col]]


# ───────────────────────────────────────────────
#  Load & merge all channels for one building
# ───────────────────────────────────────────────
def load_building_data(
    building: str, input_dir_name: str, file_prefix: str
) -> pd.DataFrame:
    """
    Reads four CSVs (supply, return, outside, ground-truth) for *one* building,
    outer-joins them on Timestamp, and tags the frame with the Building name.
    """

    supply = load_measurement(
        f"{input_dir_name}{file_prefix} supply temperature.csv", "SupplyTemp"
    )
    ret = load_measurement(
        f"{input_dir_name}{file_prefix} return temperature.csv", "ReturnTemp"
    )
    outside = load_measurement(
        f"{input_dir_name}{file_prefix} outside temperature.csv", "OutsideTemp"
    )
    ground_truth = load_measurement(
        f"{input_dir_name}{file_prefix} ground truth.csv", "SetbackActive"
    )

    merged = (
        supply.merge(ret, on="Timestamp", how="outer")
        .merge(outside, on="Timestamp", how="outer")
        .merge(ground_truth, on="Timestamp", how="outer")
    )

    merged["Building"] = building
    return merged


# ───────────────────────────────────────────────
#  Build a model-ready event/telemetry frame
# ───────────────────────────────────────────────
def build_training_frame(
    df: pd.DataFrame,
    *,
    tolerance: str | pd.Timedelta = "90s",
    direction: str = "backward",
) -> pd.DataFrame:
    """
    Each output row corresponds to a SetbackActive event and contains:

        Timestamp, Building,
        SupplyTemp, ReturnTemp,
        SupplyTempReturnTempDerivation, OutsideTemp,
        SetbackActive

    Telemetry values are copied from the nearest (or previous) telemetry sample
    within *tolerance* seconds, **inside the same building**.
    """

    # Process each building separately to avoid timestamp sorting issues across buildings
    buildings = df["Building"].unique()
    result_frames = []
    
    for building in buildings:
        building_df = df[df["Building"] == building].copy()
        
        # 1) Split once
        events = building_df[building_df["SetbackActive"].notna()].copy()
        telemetry = building_df[building_df["SetbackActive"].isna()].copy()
        
        # Drop SetbackActive from telemetry - it's all NaN anyway
        telemetry = telemetry.drop(columns=["SetbackActive"])
        
        # Ensure proper sorting by timestamp within this building
        events = events.sort_values("Timestamp")
        telemetry = telemetry.sort_values("Timestamp")
        
        base_cols = ["SupplyTemp", "ReturnTemp", "OutsideTemp", "Slope_15min"]
        
        # Skip buildings with no events
        if len(events) == 0:
            continue
            
        # 2) Join in a single pass (suffix *_tel for copied telemetry cols)
        joined = pd.merge_asof(
            events, telemetry,
            on="Timestamp",
            direction=direction,
            tolerance=pd.Timedelta(tolerance),
            suffixes=("", "_tel"),
        )
        
        # 3) Fill missing event columns from their telemetry counterparts, then drop *_tel
        for col in base_cols:
            if f"{col}_tel" in joined.columns:
                joined[col] = joined[col].fillna(joined.pop(f"{col}_tel"))
        
        # 4) Keep only rows where the three base features are now present
        joined = joined.dropna(subset=base_cols)
        
        # 5) Compute the derivation AFTER filling
        joined["SupplyTempReturnTempDerivation"] = (
            joined["SupplyTemp"] - joined["ReturnTemp"]
        ).abs()
        
        result_frames.append(joined)
    
    # Combine results from all buildings
    if not result_frames:
        return pd.DataFrame(columns=[
            "Timestamp", "Building", "SupplyTemp", "ReturnTemp",
            "SupplyTempReturnTempDerivation", "OutsideTemp", "SetbackActive", "Slope_15min"
        ])
    
    # 6) Combine and sort final result
    final_result = pd.concat(result_frames)
    
    # 7) Final tidy order
    col_order = [
        "Timestamp",
        "Building",
        "SupplyTemp",
        "ReturnTemp", 
        "SupplyTempReturnTempDerivation",
        "OutsideTemp",
        "Slope_15min",
        "SetbackActive",
    ]
    return final_result[col_order].sort_values(["Building", "Timestamp"]).reset_index(drop=True)

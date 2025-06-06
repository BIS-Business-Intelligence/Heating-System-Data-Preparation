import pandas as pd
from helpers import (
    data_cleaner_helper as dc_helper,
    star_schema_helper as star_helper,   # (left as-is even if unused)
    directory_helper as dir_helper,
    temp_fall_helper as tf_helper,        # ← our feature helper
    cyclic_encoding_helper as ce_helper
)

# ── I/O paths ─────────────────────────────────────────────────────────
input_dir_name  = "data/input/"
output_dir_name = "data/output/"

# ── Load + merge per building ─────────────────────────────────────────
b3_data = dc_helper.load_building_data("Building 3", input_dir_name, "Building 3")
b3_data["SupplyTempReturnTempDerivation"] = (b3_data["SupplyTemp"] - b3_data["ReturnTemp"]).abs()

b6_data = dc_helper.load_building_data("Building 6", input_dir_name, "Building 6")
b6_data["SupplyTempReturnTempDerivation"] = (b6_data["SupplyTemp"] - b6_data["ReturnTemp"]).abs()

b8_data = dc_helper.load_building_data("Building 8", input_dir_name, "Building 8")
b8_data["SupplyTempReturnTempDerivation"] = (b8_data["SupplyTemp"] - b8_data["ReturnTemp"]).abs()

all_buildings_data = pd.concat([b3_data, b6_data, b8_data], ignore_index=True)

# ── Normalise timestamps to whole seconds, then sort ─────────────────
all_buildings_data["Timestamp"] = all_buildings_data["Timestamp"].dt.floor("min")  ### CHANGED
all_buildings_data = (
    all_buildings_data
    .sort_values(["Building", "Timestamp"])
    .reset_index(drop=True)
)

# ── ***Add temperature-fall features on the FULL timeline*** ───────── ### NEW
all_buildings_data = tf_helper.add_temp_fall_features(all_buildings_data)      ### NEW
print(all_buildings_data.filter(regex="Slope_|DropFromPeak_|TempFallFlag").head())  ### NEW

# ── Ensure output dir exists ─────────────────────────────────────────
dir_helper.create_directory(output_dir_name)

# ── Write merged CSVs (per-building + all) ───────────────────────────
b3_data.to_csv(f"{output_dir_name}building_3_merged.csv", index=False)
b6_data.to_csv(f"{output_dir_name}building_6_merged.csv", index=False)
b8_data.to_csv(f"{output_dir_name}building_8_merged_test_data_set.csv", index=False)
all_buildings_data.to_csv(f"{output_dir_name}all_buildings_enriched.csv", index=False)  ### CHANGED (file name)

print("Enriched dataset exported as 'all_buildings_enriched.csv'.")

# ── Build model-ready dataset (inherits the new columns) ───────────── ### CHANGED comment
training_df = dc_helper.build_training_frame(
    all_buildings_data,
    tolerance="90s",
    direction="backward"
)

# Add cyclic encoding for month, hour, and minute, then drop Timestamp
training_df = ce_helper.add_cyclic_time_features(training_df, timestamp_col="Timestamp")
if "Timestamp" in training_df.columns:
    training_df = training_df.drop(columns=["Timestamp"])

print(
    f"Events with label: {all_buildings_data['SetbackActive'].notna().sum()} | "
    f"rows kept: {len(training_df)}"
)
print(training_df.head(5))

training_df.to_csv(f"{output_dir_name}training_dataset.csv", index=False)
print("Training dataset exported as 'training_dataset.csv'.")

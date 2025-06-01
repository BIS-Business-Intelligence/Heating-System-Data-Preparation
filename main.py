import pandas as pd
from helpers import (
    data_cleaner_helper as dc_helper,
    star_schema_helper as star_helper,   # (left as-is even if unused)
    directory_helper as dir_helper
)

# ── I/O paths ─────────────────────────────────────────────────────────
input_dir_name  = "data/input/"
output_dir_name = "data/output/"

# ── Load + merge per building ─────────────────────────────────────────
b3_data = dc_helper.load_building_data("Building 3", input_dir_name, "Building 3")
b3_data["SupplyTempReturnTempDerivation"] = abs(b3_data["SupplyTemp"] - b3_data["ReturnTemp"])

b6_data = dc_helper.load_building_data("Building 6", input_dir_name, "Building 6")
b6_data["SupplyTempReturnTempDerivation"] = abs(b6_data["SupplyTemp"] - b6_data["ReturnTemp"])

b8_data = dc_helper.load_building_data("Building 8", input_dir_name, "Building 8")
b8_data["SupplyTempReturnTempDerivation"] = abs(b8_data["SupplyTemp"] - b8_data["ReturnTemp"])

all_buildings_data = pd.concat([b3_data, b6_data, b8_data], ignore_index=True)

# ── Normalise timestamps to whole seconds, then sort ─────────────────
all_buildings_data["Timestamp"] = all_buildings_data["Timestamp"].dt.floor("s")
all_buildings_data = (
    all_buildings_data
    .sort_values(["Building", "Timestamp"])
    .reset_index(drop=True)
)

# ── Ensure output dir exists ─────────────────────────────────────────
dir_helper.create_directory(output_dir_name)

# ── Write merged CSVs (per-building + all) ───────────────────────────
b3_data.to_csv(f"{output_dir_name}building_3_merged.csv", index=False)
b6_data.to_csv(f"{output_dir_name}building_6_merged.csv", index=False)
b8_data.to_csv(f"{output_dir_name}building_8_merged_test_data_set.csv", index=False)
all_buildings_data.to_csv(f"{output_dir_name}all_buildings_merged.csv", index=False)
print("Merged dataset exported as 'all_buildings_merged.csv'.")

# ── Build model-ready dataset ────────────────────────────────────────
training_df = dc_helper.build_training_frame(
    all_buildings_data,
    tolerance="90s",        # widen to '120s' or '3min' if needed
    direction="backward"
)

print(
    f"Events with label: {all_buildings_data['SetbackActive'].notna().sum()} | "
    f"rows kept: {len(training_df)}"
)
print(training_df.head(5))   # quick visual sanity check

training_path = f"{output_dir_name}training_dataset.csv"
training_df.to_csv(training_path, index=False)
print(f"Training dataset exported as '{training_path}'.")

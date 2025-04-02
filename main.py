import pandas as pd
from helpers import (data_cleaner_helper as dc_helper,
                     star_schema_helper as star_helper,
                    directory_helper as dir_helper
)

# Reading and writing directories
input_dir_name = "data/input/"
output_dir_name = "data/output/"

# -------------------------
# Data Cleaning and Merging
# -------------------------

b2_data = dc_helper.load_building_data("Building 2", input_dir_name, "Building 2")
b3_data = dc_helper.load_building_data("Building 3", input_dir_name, "Building 3")

all_buildings = pd.concat([b2_data, b3_data], ignore_index=True)

dir_helper.create_directory(output_dir_name)

print("Export of Merging dataset to 'all_buildings_merged.csv' has started.")
all_buildings.to_csv(f"{output_dir_name}all_buildings_merged.csv", index=False)
print("Merged dataset exported as 'all_buildings_merged.csv'.")

# -------------------------
# Star Schema Creation
# -------------------------
dim_building = star_helper.create_dim_building(all_buildings)
dim_time = star_helper.create_dim_time(all_buildings)
fact_measurements = star_helper.create_fact_measurements(all_buildings, dim_building, dim_time)

print("Star schema CSV tables export has started.")
dim_building.to_csv(f"{output_dir_name}dim_building.csv", index=False)
dim_time.to_csv(f"{output_dir_name}dim_time.csv", index=False)
fact_measurements.to_csv(f"{output_dir_name}fact_measurements.csv", index=False)
print("Star schema CSV tables exported.")

print("Start schema Excel export has started.")
dim_time["Timestamp"] = dim_time["Timestamp"].dt.tz_localize(None)
excel_filename = f"{output_dir_name}star_schema.xlsx"
with pd.ExcelWriter(excel_filename) as writer:
    dim_building.to_excel(writer, sheet_name="DIM_BUILDING", index=False)
    dim_time.to_excel(writer, sheet_name="DIM_TIME", index=False)
    fact_measurements.to_excel(writer, sheet_name="FACT_MEASUREMENTS", index=False)
print(f"Star schema Excel file exported as '{excel_filename}'.")

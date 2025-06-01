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
b3_data = dc_helper.load_building_data("Building 3", input_dir_name, "Building 3")
b3_data["SupplyTempReturnTempDerivation"] = abs(b3_data["SupplyTemp"] - b3_data["ReturnTemp"])

b6_data = dc_helper.load_building_data("Building 6", input_dir_name, "Building 6")
b6_data["SupplyTempReturnTempDerivation"] = abs(b6_data["SupplyTemp"] - b6_data["ReturnTemp"])

b8_data = dc_helper.load_building_data("Building 8", input_dir_name, "Building 8")
b8_data["SupplyTempReturnTempDerivation"] = abs(b8_data["SupplyTemp"] - b8_data["ReturnTemp"])

# Merging datasets
all_buildings_data = pd.concat([b3_data, b6_data, b8_data], ignore_index=True)

# -------------------------
# Wrting data to CSV files
# -------------------------
print("Export of Merging dataset to 'all_buildings_merged.csv' has started.")
dir_helper.create_directory(output_dir_name)
b3_data.to_csv(f"{output_dir_name}building_3_merged.csv", index=False)
b6_data.to_csv(f"{output_dir_name}building_6_merged.csv", index=False)
b8_data.to_csv(f"{output_dir_name}building_8_merged.csv", index=False)
all_buildings_data.to_csv(f"{output_dir_name}all_buildings_merged.csv", index=False)
print("Merged dataset exported as 'all_buildings_merged.csv'.")


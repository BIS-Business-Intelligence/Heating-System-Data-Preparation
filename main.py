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

dir_helper.create_directory(output_dir_name)

print("Export of Merging dataset to 'all_buildings_merged.csv' has started.")
b3_data.to_csv(f"{output_dir_name}all_buildings_merged.csv", index=False)
print("Merged dataset exported as 'all_buildings_merged.csv'.")

# -------------------------
# Data Validation
# -------------------------

# Load the original source files for validation
print("\nPerforming validation checks on merged data...")
supply_temp = dc_helper.load_measurement(f"{input_dir_name}Building 3 supply temperature.csv", "SupplyTemp")
return_temp = dc_helper.load_measurement(f"{input_dir_name}Building 3 return temperature.csv", "ReturnTemp")
outside_temp = dc_helper.load_measurement(f"{input_dir_name}Building 3 outside temperature.csv", "OutsideTemp")
ground_truth = dc_helper.load_measurement(f"{input_dir_name}Building 3 ground truth.csv", "SetbackActive")

# Check data counts
print(f"Rows in original supply temperature dataset: {len(supply_temp)}")
print(f"Rows in original return temperature dataset: {len(return_temp)}")
print(f"Rows in original outside temperature dataset: {len(outside_temp)}")
print(f"Rows in original ground truth dataset: {len(ground_truth)}")
print(f"Rows in merged dataset: {len(b3_data)}")

# Validate presence of data from original datasets in merged dataset
supply_sample = supply_temp.sample(min(20, len(supply_temp)))
return_sample = return_temp.sample(min(20, len(return_temp)))
outside_sample = outside_temp.sample(min(20, len(outside_temp)))
ground_sample = ground_truth.sample(min(20, len(ground_truth)))

print("\nValidating supply temperature sample...")
for _, row in supply_sample.iterrows():
    timestamp = row['Timestamp']
    value = row['SupplyTemp']
    merged_value = b3_data.loc[b3_data['Timestamp'] == timestamp, 'SupplyTemp'].values
    if len(merged_value) == 0:
        print(f"ERROR: Timestamp {timestamp} from supply dataset not found in merged data")
    elif merged_value[0] != value:
        print(f"ERROR: Value mismatch at {timestamp}. Original: {value}, Merged: {merged_value[0]}")
    
print("\nValidating return temperature sample...")
for _, row in return_sample.iterrows():
    timestamp = row['Timestamp']
    value = row['ReturnTemp']
    merged_value = b3_data.loc[b3_data['Timestamp'] == timestamp, 'ReturnTemp'].values
    if len(merged_value) == 0:
        print(f"ERROR: Timestamp {timestamp} from return dataset not found in merged data")
    elif merged_value[0] != value:
        print(f"ERROR: Value mismatch at {timestamp}. Original: {value}, Merged: {merged_value[0]}")

print("\nValidating outside temperature sample...")
for _, row in outside_sample.iterrows():
    timestamp = row['Timestamp']
    value = row['OutsideTemp']
    merged_value = b3_data.loc[b3_data['Timestamp'] == timestamp, 'OutsideTemp'].values
    if len(merged_value) == 0:
        print(f"ERROR: Timestamp {timestamp} from outside temp dataset not found in merged data")
    elif merged_value[0] != value:
        print(f"ERROR: Value mismatch at {timestamp}. Original: {value}, Merged: {merged_value[0]}")

print("\nValidating ground truth sample...")
for _, row in ground_sample.iterrows():
    timestamp = row['Timestamp']
    value = row['SetbackActive']
    merged_value = b3_data.loc[b3_data['Timestamp'] == timestamp, 'SetbackActive'].values
    if len(merged_value) == 0:
        print(f"ERROR: Timestamp {timestamp} from ground truth dataset not found in merged data")
    elif merged_value[0] != value:
        print(f"ERROR: Value mismatch at {timestamp}. Original: {value}, Merged: {merged_value[0]}")

# Validate derivation calculation on sample
print("\nValidating temperature derivation calculation...")
sample_with_temps = b3_data.dropna(subset=['SupplyTemp', 'ReturnTemp']).sample(min(20, len(b3_data)))
for _, row in sample_with_temps.iterrows():
    supply = row['SupplyTemp']
    return_temp = row['ReturnTemp']
    derivation = row['SupplyTempReturnTempDerivation']
    expected = abs(supply - return_temp)
    if abs(derivation - expected) > 0.0001:  # Using tolerance for float comparison
        print(f"ERROR: Derivation calculation incorrect at {row['Timestamp']}. Expected: {expected}, Got: {derivation}")

print("\nValidation complete!")


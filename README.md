# Heating Systems BI Project Documentation

## Overview

This project is part of the BI Projects 2025 Heating System. The goal is to analyze the performance of heating systems by exploring key metrics such as supply temperature, return temperature, outside temperature, and the activation of nighttime setback. The project is divided into two major parts:

1. **Data Exploration & Transformation (Part 2):**  
   - Understand the data provided by the sponsor.
   - Transform the raw CSV data into a structured format.
   - Develop a multidimensional model and star schema to support both descriptive and predictive analyses.

2. **Predictive Analysis (Part 3):**  
   - Leverage the prepared data and star schema to perform forecasting and detect anomalies in system performance.

## Data Transformation and Renaming Process

The raw data was provided in multiple CSV files (for both Building 2 and Building 3) with non-intuitive column names such as `_time` and `_value`. To standardize and ease further analysis, the following transformations were performed:

- **Timestamp Conversion:**  
  - The `_time` column was renamed to `Timestamp` and converted to a Python datetime object.
  
- **Measurement Renaming:**  
  - The `_value` column was renamed based on the measurement type:
    - For supply temperature files, `_value` became `SupplyTemp`.
    - For return temperature files, `_value` became `ReturnTemp`.
    - For outside temperature files, `_value` became `OutsideTemp`.
    - For ground truth files, `_value` became `SetbackActive`.

- **Data Merging:**  
  - For each building, individual dataframes (supply, return, outside, ground truth) were merged on the common `Timestamp` column.
  - A new column, `Building`, was added to differentiate between Building 2 and Building 3.

## Star Schema Design

The cleaned and merged dataset was transformed into a star schema to support efficient analytical queries. The design includes:

### DIM_BUILDING

- **Purpose:**  
  Provide a dimension for building information.
  
- **Key Columns:**  
  - `buildingID` (Surrogate Key)
  - `buildingName` (E.g., "Building 2", "Building 3")

### DIM_TIME

- **Purpose:**  
  Offer a detailed time dimension for time-based aggregations.
  
- **Key Columns:**  
  - `timeID` (Surrogate Key)
  - `Timestamp` (Original datetime, converted to be timezone-naïve for Excel export)
  - `Year`, `Month`, `Day`, `Hour`, `Minute`, `Second`

### FACT_MEASUREMENTS

- **Purpose:**  
  Store the core measurement events for each building at each timestamp.
  
- **Key Columns:**  
  - `measurementID` (Surrogate Key)
  - `buildingID` (Foreign Key from DIM_BUILDING)
  - `timeID` (Foreign Key from DIM_TIME)
  - `SupplyTemp`, `ReturnTemp`, `OutsideTemp`, `SetbackActive` (Measures)

## Python Script Structure

The Python script is divided into several sections:

1. **Data Loading and Cleaning:**  
   Uses helper functions to read each CSV file, rename columns, convert timestamps, and filter relevant columns.

2. **Merging Data for Each Building:**  
   Merges supply, return, outside, and ground truth data based on `Timestamp` for each building separately.

3. **Combining Building Data:**  
   Concatenates the merged datasets for both Building 2 and Building 3.

4. **Star Schema Creation:**  
   - **DIM_BUILDING:** Extracts unique building names and assigns surrogate keys.
   - **DIM_TIME:** Extracts unique timestamps and computes additional date/time attributes.
   - **FACT_MEASUREMENTS:** Joins the merged dataset with the dimensions to map foreign keys.

5. **Exporting Results:**  
   The final star schema tables are exported as CSV files and as an Excel file with separate sheets for each table.

## How to Run the Code

1. **Set Up the Environment:**  
   - Ensure Python 3.12 is installed.
   - Ensure that the `input/` and `output/` directory exists.
   - Place all input CSV files in the `input/` directory.

### Deployment Instructions

- Setup a Virtual Python environment

#### On Windows based Operating Systems
```bash
python -m venv venv
venv\Scripts\activate
```
#### On MacOS or Linux based Operating Systems
```bash
python3 -m venv venv
source venv/bin/activate
```

- Install the Dependencies
```bash
pip install -r requirements.txt
```

- **Run the Script:**  
   Execute the Python script using your preferred IDE or command line:
```bash
python main.py
```
## Review the Outputs
   - The merged dataset is saved as all_buildings_merged.csv.
   - The star schema tables are saved as dim_building.csv, dim_time.csv, and fact_measurements.csv.
   - An Excel file (star_schema.xlsx) is generated with separate sheets for each table.

## Conclusion
This project successfully transforms raw CSV data into a structured star schema that supports self-service BI and predictive analysis. The documented renaming and transformation process ensures that the data is consistent and ready for visualization in tools such as Tableau.
#!/usr/bin/env python3
import json
import pandas as pd
import os
import glob


def concatenate_json_files(file_pattern, output_file):
    all_data = []
    field_names = None
    field_cardinality = {}

    # Get list of all JSON files
    json_files = glob.glob(file_pattern)
    print(f"Found {len(json_files)} JSON files to process")

    for file_path in json_files:
        print(f"Processing: {file_path}")

        try:
            with open(file_path, "r") as f:
                json_data = json.load(f)

            # Extract field information from the first file
            if field_names is None and "fields" in json_data:
                field_names = []
                for field in json_data["fields"]:
                    field_names.append(field["name"])
                    field_cardinality[field["name"]] = field["cardinality"]

                # Sort field names by their cardinality
                field_names = sorted(field_names, key=lambda x: field_cardinality[x])
                print(f"Field names (in cardinality order): {field_names}")

            # Collect data from the current file
            if "data" in json_data:
                for row in json_data["data"]:
                    # Convert the row to a dictionary using field names
                    if len(row) == len(field_names):
                        row_dict = {
                            field_names[i]: row[i] for i in range(len(field_names))
                        }
                        all_data.append(row_dict)
                    else:
                        print(
                            f"Warning: Row has {len(row)} values but there are {len(field_names)} fields. Skipping."
                        )

        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")

    if all_data:
        # Create DataFrame from all data
        df = pd.DataFrame(all_data)

        # Save concatenated data to CSV
        df.to_csv(output_file, index=False)
        print(f"Successfully saved {len(all_data)} rows to {output_file}")
        return df
    else:
        print("No data found in the JSON files.")
        return None


if __name__ == "__main__":
    # For the filtered directory
    filtered_output = "concatenated_filtered_data.csv"
    df_filtered = concatenate_json_files("data/api/filtered/*.json", filtered_output)

    # For the all directory - this may be very large and require more memory
    all_output = "concatenated_all_data.csv"
    df_all = concatenate_json_files("data/api/all/*.json", all_output)

    # Choose one SettlementPointName value, and filter the data for that value; then save
    # that to a "filtered_all_data.csv" file
    if df_all is not None and "settlementPoint" in df_all.columns:
        # Get unique settlement points
        settlement_points = df_all["settlementPoint"].unique()
        print(f"Found {len(settlement_points)} unique settlement points")

        if len(settlement_points) > 0:
            # Select the first settlement point (or you can specify a particular one)
            # You could also modify this to use ALP_BESS_RN as mentioned in final_notebook2.py
            selected_point = (
                "ALP_BESS_RN"  # Can be changed to any other point of interest
            )

            # Check if the selected point exists in the data
            if selected_point in settlement_points:
                print(f"Filtering data for settlement point: {selected_point}")
            else:
                print(
                    f"{selected_point} not found. Using first available point: {settlement_points[0]}"
                )
                selected_point = settlement_points[0]

            # Filter the data
            filtered_df = df_all[df_all["settlementPoint"] == selected_point]

            # Save to a new file
            filtered_output_file = "filtered_all_data.csv"
            filtered_df.to_csv(filtered_output_file, index=False)
            print(
                f"Successfully saved {len(filtered_df)} rows to {filtered_output_file}"
            )
        else:
            print("No settlement points found in the data")
    else:
        print("Cannot filter by settlementPoint - data not available or column missing")

    print("Concatenation complete!")

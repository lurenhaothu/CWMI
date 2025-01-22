import os
import pandas as pd

# Define the path to the results folder
results_folder = "model/results/"

# Initialize an empty list to store all raw data
all_data = []

# Loop through each subfolder in the results folder
for experiment in os.listdir(results_folder):
    experiment_path = os.path.join(results_folder, experiment)

    # Check if the path is a folder
    if os.path.isdir(experiment_path):
        result_file = os.path.join(experiment_path, "_test_result_new.csv")

        # Check if the result file exists
        if os.path.isfile(result_file):
            # Read the CSV file
            df = pd.read_csv(result_file)

            # Add a column to indicate the experiment name
            df["Experiment"] = experiment

            # Append the DataFrame to the list
            all_data.append(df)

# Concatenate all the DataFrames
all_data_df = pd.concat(all_data, ignore_index=True)

# Save the concatenated data to a CSV file
raw_data_csv_path = results_folder + "all_raw_data_GlaS.csv"
all_data_df.to_csv(raw_data_csv_path, index=False)

print(f"All raw data saved to {raw_data_csv_path}")

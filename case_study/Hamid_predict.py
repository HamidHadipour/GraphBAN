
import os
from tqdm import tqdm

# Base command template
command_template = (
    "python predict.py --test_path zinc_data2/split_zinc_{index}.csv "
    "--folder_path result_biosnap12_zinc2{index} --save_dir test_zinc_new{index}_preds.csv"
  
)

# Loop through indexes 1 to 25 with tqdm progress bar
for i in tqdm(range(1, 26), desc="Running model for zinc files"):
    # Format the command with the current index
    command = command_template.format(index=i)
    
    # Execute the command
    os.system(command)

import os
import pandas as pd

# List to store dataframes
dataframes = []

# Define the first and last indexes


# Loop through the specified indexes (first and last)
for i in range(1,26):
    # Construct the folder name
    folder_name = f"result_biosnap12_zinc2{i}"
    
    # Construct the file path for the CSV
    csv_file_path = os.path.join(folder_name, f"test_zinc_new{i}_preds.csv")
    
    # Check if the file exists
    if os.path.exists(csv_file_path):
        # Read the CSV file
        df = pd.read_csv(csv_file_path)
        
        # Append the dataframe to the list
        dataframes.append(df)
    else:
        print(f"File {csv_file_path} does not exist.")
        
# Concatenate all dataframes vertically
stacked_df = pd.concat(dataframes, ignore_index=True)

# Optionally save the stacked dataframe to a CSV file
stacked_df.to_csv("stacked_results_oct28.csv", index=False)

print("All files have been stacked and saved to 'stacked_results_new.csv'.")

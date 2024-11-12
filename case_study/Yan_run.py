import os
from tqdm import tqdm

# Base command template
command_template = (
    "python run_model.py --train_path kiba_train_data/source_train_kiba12.csv "
    "--val_path zinc_data2/split_zinc_{index}.csv --test_path zinc_data2/split_zinc_{index}.csv "
    "--seed {index} --mode inductive --teacher_path kiba_train_data/kiba12_inductive_teacher_emb.parquet "
    "--result_path result_kiba12_zinc2{index}/"
)

# Loop through indexes 1 to 25 with tqdm progress bar
for i in tqdm(range(1, 26), desc="Running model for zinc files"):
    # Format the command with the current index
    command = command_template.format(index=i)
    
    # Execute the command
    os.system(command)

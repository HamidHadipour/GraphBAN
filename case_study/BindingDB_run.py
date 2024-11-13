import os
from tqdm import tqdm

# Base command template
command_template = (
    "python case_study/run_model.py --train_path case_study/bindingdb_train_data/source_train_bindingdb12.csv "
    "--val_path case_study/zinc_data/split_zinc_{index}.csv --test_path case_study/zinc_data/split_zinc_{index}.csv "
    "--seed {index} --mode inductive --teacher_path case_study/bindingdb_train_data/bindingdb12_inductive_teacher_emb.parquet "
    "--result_path case_study/result_bindingdb12_zinc2{index}/"
)

# Loop through indexes 1 to 25 with tqdm progress bar
for i in tqdm(range(1, 26), desc="Running model for zinc files"):
    # Format the command with the current index
    command = command_template.format(index=i)
    
    # Execute the command
    os.system(command)

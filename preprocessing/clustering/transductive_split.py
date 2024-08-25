from tqdm import tqdm
import re
from sklearn.metrics import pairwise_distances
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import pandas as pd
import numpy as np
import sys


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sys

seed = 20

def split_csv(file_path, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Split into train and temp (remaining part)
    train_df, temp_df = train_test_split(df, test_size=1-train_ratio, random_state=seed)
    
    # Calculate the validation and test size from the temp set
    val_ratio_temp = val_ratio / (val_ratio + test_ratio)
    
    # Split the temp set into validation and test sets
    val_df, test_df = train_test_split(temp_df, test_size=1-val_ratio_temp, random_state=seed)
    
    # Save the splits to new CSV files
    train_df.to_csv('train_pdb20.csv', index=False)
    val_df.to_csv('val_pdb20.csv', index=False)
    test_df.to_csv('test_pdb20.csv', index=False)
    
    print(f"Train set: {len(train_df)} samples")
    print(f"Validation set: {len(val_df)} samples")
    print(f"Test set: {len(test_df)} samples")

# Usage example
file_path = 'balanced_pdb_dataset12.csv'  # Replace with your CSV file path
split_csv(file_path)

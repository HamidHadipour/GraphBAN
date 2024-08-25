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

AALetter = [
    "A",
    "R",
    "N",
    "D",
    "C",
    "E",
    "Q",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
]


def get_ecfp_encoding(smiles, radius=2, nBits=1024):
    ecfp_lst = []
    for smile in smiles:
        mol = Chem.MolFromSmiles(smile)
        if not mol:
            print(f"Unable to compile SMILES: {smile}")
            sys.exit()
        features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
        features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(features_vec, features)
        ecfp_lst.append(features)
    ecfp_lst = np.array(ecfp_lst)
    return ecfp_lst


def get_3mers():
    kmers = list()
    for i in AALetter:
        for j in AALetter:
            for k in AALetter:
                kmers.append(i + j + k)
    return kmers


def get_kmers_encoding(aa_sequences, kmers):
    kmers_encoding_lst = []
    for aa_sequence in tqdm(aa_sequences):
        kmers_encoding = []
        for i in kmers:
            kmers_encoding.append(len(re.findall(i, aa_sequence)))
        kmers_encoding_lst.append(kmers_encoding)
    return np.array(kmers_encoding_lst)



df = pd.read_csv('train_smiles.csv')
labels = df['Y'].tolist()

#extract SMILES features

smile_lst = df['SMILES'].tolist()
drug_feature_lst = get_ecfp_encoding(smile_lst)
df_mol_features = pd.DataFrame(drug_feature_lst)
print(df_mol_features.shape)

#extract protein features
#aas_lst = df['Protein'].tolist()
#kmers = get_3mers()
#target_feature_lst = get_kmers_encoding(aas_lst, kmers)
#df_protein_features = pd.DataFrame(target_feature_lst)

#print(df_protein_features.shape)

#result = pd.concat([df_mol_features, df_protein_features], axis=1)
#result = pd.concat([df_mol_features, df_protein_features], axis=1)
df_mol_features['Y'] = labels
print(df_mol_features.shape)
df_mol_features.to_csv("ecoli_train_features.csv", index = False)

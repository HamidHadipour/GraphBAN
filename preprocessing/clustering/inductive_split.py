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


def hi_cluster_split(df, drug_threshold=0.5, target_threshold=0.5):
    smile_lst = df['SMILES'].unique().tolist()
    drug_feature_lst = get_ecfp_encoding(smile_lst)

    aas_lst = df['Protein'].unique().tolist()
    kmers = get_3mers()
    target_feature_lst = get_kmers_encoding(aas_lst, kmers)

    # drug cluster
    smile_cluster_dict = {}
    distance_matrix = pairwise_distances(X=drug_feature_lst, metric="jaccard")
    cond_distance_matrix = squareform(distance_matrix)
    Z = linkage(cond_distance_matrix, method="single")
    cluster_labels = fcluster(Z, t=drug_threshold, criterion="distance")
    for smile, cluster_ in zip(smile_lst, cluster_labels):
        smile_cluster_dict[smile] = cluster_
    df["drug_cluster"] = df["SMILES"].map(smile_cluster_dict)

    # protein cluster
    target_cluster_dict = {}
    distance_matrix = pairwise_distances(X=target_feature_lst, metric="cosine")
    cond_distance_matrix = squareform(distance_matrix)
    Z = linkage(cond_distance_matrix, method="single")
    cluster_labels = fcluster(Z, t=target_threshold, criterion="distance")
    for aas, cluster_ in zip(aas_lst, cluster_labels):
        target_cluster_dict[aas] = cluster_
    df["target_cluster"] = df["Protein"].map(target_cluster_dict)
    return df




def unseen_cluster_pair_test_split(df, test_size=0.4, seed=20):
    np.random.seed(seed)  # Seed for reproducibility in cluster selection

    drug_clusters = df.drug_cluster.unique()
    target_clusters = df.target_cluster.unique()
    test_num_drug_clusters = int(len(drug_clusters) * test_size)
    test_num_target_clusters = int(len(target_clusters) * test_size)

    # Select clusters randomly with seeding
    test_drug_clusters = np.random.choice(drug_clusters, test_num_drug_clusters, replace=False)
    test_target_clusters = np.random.choice(target_clusters, test_num_target_clusters, replace=False)

    target_df = df[df['drug_cluster'].isin(test_drug_clusters) & df['target_cluster'].isin(test_target_clusters)]
    source_df = df[~(df['drug_cluster'].isin(test_drug_clusters) | df['target_cluster'].isin(test_target_clusters))]

    # Sampling from DataFrame with seed for reproducibility
    target_train_df = target_df.sample(frac=0.8, random_state=seed)
    target_test_df = target_df[~(target_df.index.isin(target_train_df.index))]

    print(f"Source Training size: {len(source_df)}")
    print(f"Target Training size: {len(target_train_df)}, test size: {len(target_test_df)}")
    return source_df, target_train_df, target_test_df



if __name__ == "__main__":
    df = pd.read_csv("balanced_pdb_dataset12.csv")
    df = hi_cluster_split(df, drug_threshold=0.5, target_threshold=0.5)
    source_train_df, target_train_df, target_test_df = unseen_cluster_pair_test_split(df, test_size=0.4)
    source_train_df.to_csv("source_train_pdb20.csv", index = False)
    target_train_df.to_csv("target_train_pdb20.csv", index = False)
    target_test_df.to_csv("target_test_pdb20.csv", index = False)

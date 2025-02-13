import pandas as pd

import numpy as np

from sklearn.manifold import TSNE

binding_train = pd.read_csv("bindingdb_trans_12_train_features.csv")
binding_test = pd.read_csv("bindingdb_trans_12_test_features.csv")

print(binding_train.shape)

# Apply t-SNE to both datasets
tsne = TSNE(n_components=2, random_state=42)
bindingdb_train_tsne = tsne.fit_transform(binding_train)
bindingdb_test_tsne = tsne.fit_transform(binding_test)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(bindingdb_train_tsne[:, 0], bindingdb_train_tsne[:, 1], color='red', alpha=0.5, label='ST Dataset', s=10)
plt.scatter(bindingdb_test_tsne[:, 0], bindingdb_test_tsne[:, 1], color='blue', alpha=0.5, label='TT Dataset', s=10)
#plt.title('tsne Comparison of bindingDB12_st and bindingDB12_tt')
plt.xlabel('tsne 1')
plt.ylabel('tsne 2')
plt.legend()
plt.savefig("bindingdb_transductive_tsne_domains.png", bbox_inches='tight', dpi = 600)
plt.show()
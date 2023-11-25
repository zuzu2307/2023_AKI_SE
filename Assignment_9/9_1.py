import mglearn
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF
from sklearn.manifold import TSNE

digits = load_digits()

#----------------------------- PCA ----------------------------- 

scaler = StandardScaler()
scaler.fit(digits.data)
X_scaled = scaler.transform(digits.data)
pca = PCA(n_components=2)


#----------------------------- NMF ----------------------------- 

nmf = NMF(n_components=16, init='random', random_state=0)


#----------------------------- t-SNE ----------------------------- 

tsne = TSNE(random_state=42)


# transform data onto the first two principal components
digits_pca = pca.fit_transform(X_scaled)
digits_nmf = nmf.fit_transform(digits.data)
digits_tsne = tsne.fit_transform(digits.data)

colors = ["#476A2A", "#7851B8", "#BD3430", "#4A2D4E", "#875525", "#A83683", "#4E655E", "#853541", "#3A3120", "#535D8E"]

selected_item = digits_tsne

plt.figure(figsize=(10, 10))
plt.xlim(selected_item[:, 0].min(), selected_item[:, 0].max() + 1)
plt.ylim(selected_item[:, 1].min(), selected_item[:, 1].max() + 1)

for i in range(len(digits.data)):
    # actually plot the digits as text instead of using scatter
    plt.text(selected_item[i, 0], selected_item[i, 1], str(digits.target[i]),
    color = colors[digits.target[i]],
    fontdict={'weight': 'bold', 'size': 9})

plt.xlabel("First principal component")
plt.ylabel("Second principal component")
plt.show()
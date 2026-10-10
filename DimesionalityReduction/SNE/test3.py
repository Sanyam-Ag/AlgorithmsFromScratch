from sne import SNE
from sklearn.datasets import load_iris
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

iris = load_iris()
X = iris.data
y = iris.target

# Your vanilla SNE
s = SNE(
    perplexity=10,
    seed=42,
    lr=0.1,
    trackloss=True
)

Y_sne = s.fit(X, newdims=2, iters=1000)

# If fit() currently returns None, make it return Y at the end.
if Y_sne is None:
    Y_sne = s.embedding

# Reference: scikit-learn t-SNE (not vanilla SNE)
reference = TSNE(
    n_components=2,
    perplexity=10,
    learning_rate="auto",
    init="random",
    max_iter=1000,
    random_state=42
)

Y_tsne = reference.fit_transform(X)

# Plot both embeddings
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

names = iris.target_names
colors = ["red", "green", "blue"]

for ax, Y, title in [
    (axes[0], Y_sne, "My vanilla SNE"),
    (axes[1], Y_tsne, "Scikit-learn t-SNE"),
]:
    for i, name in enumerate(names):
        mask = y == i
        ax.scatter(
            Y[mask, 0],
            Y[mask, 1],
            color=colors[i],
            label=name,
            s=35,
            alpha=0.8
        )

    ax.set_title(title)
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.legend()
    ax.grid(alpha=0.2)

plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split

from sne import SNE


print("Loading MNIST...")
mnist = fetch_openml("mnist_784", version=1, as_frame=False)

X = mnist.data.astype(np.float32) / 255.0
y = mnist.target.astype(int)

X_train, _, y_train, _ = train_test_split(
    X,
    y,
    train_size=3000,
    random_state=42,
    stratify=y,
)

rng = np.random.default_rng(42)
indices = np.arange(len(y))
selected, _ = train_test_split(
    indices,
    train_size=3000,
    random_state=42,
    stratify=y,
)
X_sample = X[selected]
y_sample = y[selected]

print("Dataset shape:", X_sample.shape)
print("Digits:", np.bincount(y_sample))


print("Running vanilla SNE...")
sne = SNE(
    perplexity=30,
    seed=42,
    lr=0.01,
    trackloss=True,
)

Y_sne = sne.fit(X_sample, newdims=2, iters=1000)

if Y_sne is None:
    Y_sne = sne.embedding


print("Running scikit-learn t-SNE...")
tsne = TSNE(
    n_components=2,
    perplexity=30,
    init="random",
    learning_rate="auto",
    max_iter=1000,
    random_state=42,
)

Y_tsne = tsne.fit_transform(X_sample)


fig, axes = plt.subplots(1, 2, figsize=(16, 7))

for ax, embedding, title in [
    (axes[0], Y_sne, "My vanilla SNE"),
    (axes[1], Y_tsne, "Scikit-learn t-SNE"),
]:
    points = ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=y_sample,
        cmap="tab10",
        s=8,
        alpha=0.75,
    )
    ax.set_title(title)
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.grid(alpha=0.2)

fig.colorbar(
    points,
    ax=axes,
    ticks=range(10),
    label="Digit label",
    shrink=0.8,
)

plt.tight_layout()
plt.show()

if getattr(sne, "lossArr", None):
    plt.figure(figsize=(8, 4))
    plt.plot(sne.lossArr)
    plt.title("Vanilla SNE training loss")
    plt.xlabel("Recorded iteration")
    plt.ylabel("Cross-entropy loss")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
from sne import SNE

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# MNIST: 28x28 images flattened into 784 features.
N_SAMPLES =3000
RANDOM_STATE = 42

mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
X = mnist.data.astype(np.float64)[:N_SAMPLES] / 255.0
y = mnist.target.astype(np.int64)[:N_SAMPLES]

# Optional speed-up: reduce 784 pixel features to 50 PCA features first.
# This keeps most useful variation while making pairwise distance calculations cheaper.
X = PCA(n_components=50, random_state=RANDOM_STATE).fit_transform(X)

s = SNE(
    perplexity=80,
    seed=RANDOM_STATE,
    lr=0.2,
    trackloss=True,
)

Y = s.fit(X, newdims=2, iters=1000)

if Y is None:
    Y = s.embedding

Y = np.asarray(Y)

fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(
    Y[:, 0],
    Y[:, 1],
    c=y,
    cmap="tab10",
    s=8,
    alpha=0.75,
    linewidths=0,
)
legend = ax.legend(
    *scatter.legend_elements(),
    title="Digit",
    loc="best",
    markerscale=2,
)
ax.add_artist(legend)
ax.set_title("MNIST embedded in 2D using my SNE implementation")
ax.set_xlabel("SNE dimension 1")
ax.set_ylabel("SNE dimension 2")
ax.grid(alpha=0.2)
plt.tight_layout()
plt.show()

if getattr(s, "lossArr", None):
    plt.figure(figsize=(8, 4))
    plt.plot(s.lossArr)
    plt.title("SNE training loss")
    plt.xlabel("Iteration")
    plt.ylabel("Cross-entropy")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()

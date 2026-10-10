"""
    t-SNE is a nonlinear dimensionality-reduction technique primarily used to visualize high-dimensional data in a 
    low-dimensional space, typically 2D or 3D.

    Unlike PCA, which performs a linear projection by finding directions that maximize variance, 
    t-SNE focuses primarily on preserving local neighborhood relationships.

    It converts pairwise similarities between high-dimensional data points into probabilities. 
    It then constructs a low-dimensional representation in which points that are close or similar in the original space tend to 
    remain close together.

    One important distinction is that t-SNE should not be interpreted as preserving the global structure of the data. 
    In particular, the distances between well-separated clusters, the relative sizes of clusters, and even the positions 
    of clusters can be misleading. Therefore, t-SNE is better suited for understanding local neighborhoods and identifying visually 
    separated groups than for interpreting global distances.

    t-SNE is an extension of SNE. A key difference is that SNE uses Gaussian distributions in the low-dimensional space, 
    whereas t-SNE uses a heavy-tailed Student's t-distribution. The heavy tails help address the crowding problem encountered by 
    SNE, giving moderately dissimilar points more room in the low-dimensional representation and generally making separated clusters 
    more visually distinct.

    Therefore, t-SNE is particularly useful for visualizing complex, nonlinear structure in high-dimensional datasets, but its 
    2D/3D embedding should not be treated as a faithful representation of all pairwise distances or global relationships.

"""

import numpy as np

class tSNE:
    def __init__(self):
        pass


    def __call__(self):
        pass
import numpy as np
from sklearn.neighbors import NearestNeighbors


class SMOTEOverSampler:
    def __init__(self, k_neighbors=5, random_state=None):
        self.k_neighbors = k_neighbors
        self.random_state = random_state

    def fit_resample(self, X, y,W=None):
        X = np.array(X)
        y = np.array(y)

        classes, counts = np.unique(y, return_counts=True)
        max_samples = np.max(counts)

        rng = np.random.default_rng(self.random_state)

        new_X, new_y = [], []
        for class_label in classes:
            class_indices = np.where(y == class_label)[0]
            class_samples = X[class_indices]
            if W is not None:
                W=np.array(W)
                w=W[class_indices]

            if len(class_samples) < self.k_neighbors:
                raise ValueError(f"Class {class_label} has fewer samples than k_neighbors.")

            if len(class_samples) < max_samples:
                nbrs = NearestNeighbors(n_neighbors=self.k_neighbors).fit(class_samples)
                distances, indices = nbrs.kneighbors(class_samples)

                num_synthetic_samples = max_samples - len(class_samples)
                for i in range(num_synthetic_samples):
                    cen=i % len(class_samples)
                    sample = class_samples[cen]
                    rr=rng.integers(1, self.k_neighbors)
                    nn = class_samples[indices[cen][rr]]
                    diff = nn - sample

                    if W is not None:
                        nn_w=w[indices[cen][rr]]
                        sample_w=w[cen]
                        gap = nn_w/(sample_w+nn_w) #rng.random()
                    else:
                        gap = rng.random()
                    synthetic_sample = sample + gap * diff
                    new_X.append(synthetic_sample)
                    new_y.append(class_label)
            new_X.extend(class_samples)
            new_y.extend([class_label] * len(class_samples))

        return np.array(new_X), np.array(new_y)

import numpy as np
from sklearn.neighbors import NearestNeighbors


class ADASYNOverSampler:
    def __init__(self, k_neighbors=100, random_state=None):
        self.k_neighbors = k_neighbors
        self.random_state = random_state

    def fit_resample(self, X, y,W=None):
        X = np.array(X)
        y = np.array(y)

        classes, counts = np.unique(y, return_counts=True)
        max_samples = np.max(counts)
        minority_class = classes[np.argmin(counts)]

        rng = np.random.default_rng(self.random_state)

        new_X, new_y = [], []

        class_indices = np.where(y == minority_class)[0]
        class_samples = X[class_indices]

        if W is not None:
            w = np.array(W)
            # w = W[class_indices]

        if len(class_samples) < self.k_neighbors:
            raise ValueError(f"Class {minority_class} has fewer samples than k_neighbors.")

        if len(class_samples) < max_samples:
            nbrs = NearestNeighbors(n_neighbors=self.k_neighbors).fit(X)
            distances, indices = nbrs.kneighbors(class_samples)

            # Calculate the ratio of samples to generate for each sample in the minority class
            ki = np.sum(y[indices] != minority_class, axis=1)  # Count of majority samples among k neighbors
            ri = ki / self.k_neighbors  # Ratio of majority neighbors
            ri = ri / np.sum(ri)  # Normalize to make sum = 1
            num_synthetic_samples = max_samples - len(class_samples)
            num_samples_per_instance = (ri * num_synthetic_samples+0.5).astype(int)# Number of samples to generate for each instance
            # print(sum(num_samples_per_instance))

            for i, sample in enumerate(class_samples):
                for _ in range(num_samples_per_instance[i]):
                    rr = rng.integers(1, self.k_neighbors)
                    nn_idx=indices[i][rr]
                    nn = X[nn_idx]
                    while y[nn_idx] != minority_class:
                        nn_idx = indices[i][rng.integers(1, self.k_neighbors)]
                    diff = nn - sample
                    if W is not None:
                        nn_w=w[nn_idx]
                        sample_w=w[class_indices[i]]
                        gap = nn_w/(sample_w+nn_w) #rng.random()
                    else:
                        gap = rng.random()

                    synthetic_sample = sample + gap * diff
                    new_X.append(synthetic_sample)
                    new_y.append(minority_class)

        new_X.extend(class_samples)
        new_y.extend([minority_class] * len(class_samples))

        # Append majority class samples as well
        majority_indices = np.where(y != minority_class)[0]
        new_X.extend(X[majority_indices])
        new_y.extend(y[majority_indices])

        return np.array(new_X), np.array(new_y)



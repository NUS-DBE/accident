import numpy as np


class RandomOverSampler:
    def __init__(self, random_state=None):
        self.random_state = random_state

    def fit_resample(self, X, y,W=None):
        # Ensure data is numpy array
        X = np.array(X)
        y = np.array(y)

        # Get unique classes and their counts
        classes, counts = np.unique(y, return_counts=True)

        # Find the class with maximum samples
        max_samples = np.max(counts)

        # Seed the random number generator
        rng = np.random.default_rng(self.random_state)


        # Oversampling
        new_X, new_y = [], []
        for class_label in classes:
            class_indices = np.where(y == class_label)[0]
            if W is not None:
                weight=W[class_indices].tolist()
                # print(sum(weight))
                if sum(weight)!=0:
                    oversample_indices = rng.choice(class_indices, max_samples,p=weight, replace=True)
                else: oversample_indices = rng.choice(class_indices, max_samples, replace=True)   #p = [0.1, 0, 0.3, 0.6, 0]
            else:
                oversample_indices = rng.choice(class_indices, max_samples, replace=True)
            new_X.extend(X[oversample_indices])
            new_y.extend([class_label] * max_samples)

        return np.array(new_X), np.array(new_y)

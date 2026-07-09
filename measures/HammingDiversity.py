import numpy as np


class HammingDiversity:
    def __init__(self):
        self.history = []

    @staticmethod
    def pairwise_distances(population):
        pop = np.asarray(population)
        m = pop.shape[0]
        if m < 2:
            return np.array([])
        dist_matrix = (pop[:, None, :] != pop[None, :, :]).sum(axis=2)
        iu = np.triu_indices(m, k=1)
        return dist_matrix[iu]

    @classmethod
    def summary(cls, population):
        pop = np.asarray(population)
        n = pop.shape[1]
        distances = cls.pairwise_distances(pop)
        if distances.size == 0:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "mean_normalized": 0.0}
        return {
            "mean":            float(np.mean(distances)),
            "std":             float(np.std(distances)),
            "min":             float(np.min(distances)),
            "max":             float(np.max(distances)),
            "mean_normalized": float(np.mean(distances) / n),
        }

    def record(self, generation, population):
        entry = {"generation": generation, **self.summary(population)}
        self.history.append(entry)
        return entry

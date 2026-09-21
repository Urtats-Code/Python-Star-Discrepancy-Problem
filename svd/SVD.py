
import numpy as np


# In this file we provide a class for receiving a DSM decomposing it with Singular Value Decomposition (SVD) and reconstruct it with noise injection 

class SVD_Noise_Injection(): 

    def __init__(self, external_DSM):
        self.DSM = external_DSM.copy()
        self.U = None
        self.V = None
        self.Sigma = None

    def decompose(self):
        self.U, self.Sigma, self.V = np.linalg.svd(self.DSM, full_matrices=False)

    def reconstruct(self):
        result = self.U @ np.diag(self.Sigma) @ self.V
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(result, a_min=0.0, a_max=None)

    def noise_injection(self, diagonal, theta=0.4):
        noise = np.random.normal(0, theta, size=diagonal.shape)
        return np.clip(diagonal + noise, a_min=0, a_max=None)

    def inject(self, theta=0.4):
        self.decompose()
        self.Sigma = self.noise_injection(self.Sigma, theta=theta)
        return self.reconstruct()

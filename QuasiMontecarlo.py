import numpy as np
from scipy.stats import qmc
import math

class QuasiMontecarlo:
    def __init__(self, n: int):
        self.n = n

    def calculate_area(self) -> float:
        sampler = qmc.Sobol(d=2, scramble=False)
        sample = sampler.random(n=self.n)
        dist_sq = np.sum(sample**2, axis=1)
        points_inside = np.sum(dist_sq <= 1.0)
        estimated_area = 4 * (points_inside / self.n)
        
        return estimated_area

if __name__ == "__main__":
    n_points = 50 
    qmc_calculator = QuasiMontecarlo(n_points)
    
    area = qmc_calculator.calculate_area()
    true_area = math.pi
    error = abs(area - true_area)
    
    print(f"Estimation of circle area using N = {n_points} Sobol points:")
    print(f"Estimated Area: {area:.8f}")
    print(f"True Area (Pi): {true_area:.8f}")
    print(f"Absolute Error: {error:.8e}")

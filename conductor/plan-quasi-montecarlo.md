# Implementation Plan: QuasiMontecarlo Class

## Objective
Create a `QuasiMontecarlo` class in a new file that utilizes Sobol sequences to estimate the area of a circle. The class will take the number of points `N` as a parameter.

## Proposed Solution
We will implement the class in `QuasiMontecarlo.py` using `scipy.stats.qmc.Sobol` for generating low-discrepancy sequences. The estimation will be done by generating 2D points in the unit square `[0, 1) x [0, 1)`, checking how many fall within the quarter circle (`x^2 + y^2 <= 1`), and multiplying the proportion by 4.

## Implementation Steps
1.  **Create File:** Create `QuasiMontecarlo.py` in the root directory.
2.  **Define Class:** Implement the `QuasiMontecarlo` class with an `__init__(self, n)` method.
3.  **Implement Area Calculation:**
    *   Add a method `calculate_area(self)`.
    *   Initialize `scipy.stats.qmc.Sobol(d=2, scramble=False)`.
    *   Generate `N` points. Note that `scipy`'s Sobol generator requires `N` to be a power of 2 for optimal uniformity, but we will generate exactly the requested `N` points using `random(n)`.
    *   Calculate the distance from the origin for each point.
    *   Count points where `x**2 + y**2 <= 1`.
    *   Return `4 * (points_inside / N)`.
4.  **Add Validation:** Include an `if __name__ == "__main__":` block to instantiate the class with a sample `N` (e.g., $2^{12} = 4096$), print the estimated area, and compare it to `math.pi`.

## Verification & Testing
-   Run `python3 QuasiMontecarlo.py`.
-   Verify that the script executes without errors.
-   Check that the printed area is a close approximation of Pi (~3.14159).
import numpy as np
import gurobipy as gp
from gurobipy import GRB

class SDPProblem:
    def __init__(self, problem_size: int, max_evaluations: int, epsilon: float):
        """
        Initializes the SDP problem with a persistent Gurobi environment.
        """
        self.n = problem_size
        self.max_evaluations = max_evaluations
        self.epsilon = epsilon
        self.evaluations = 0
        
        # Reuse environment to avoid re-licensing overhead in EDA loops
        self.env = gp.Env(empty=True)
        self.env.setParam("OutputFlag", 0)  # Silent mode
        self.env.start()

    def transform_into_binary_matrix(self, solution: np.ndarray) -> np.ndarray:
        """
        Converts a permutation into an (n+1)x(n+1) matrix using vectorized NumPy indexing.
        """
        matrix = np.zeros((self.n + 1, self.n + 1), dtype=int)
        # Fast vectorized assignment
        rows = np.arange(self.n)
        cols = solution.astype(int) - 1
        matrix[rows, cols] = 1
        
        # Add dummy points as defined in the original C++ SDP::TransformIntoBinaryMatrix
        matrix[0, self.n] = 1
        matrix[self.n, 0] = 1
        return matrix

    def solve_optimization(self, a_matrix: np.ndarray) -> float:
        """
        Builds and solves the non-convex MIQCP problem using Gurobi.
        Optimized with O(1) prefix-sum lookups for constraint parameters.
        """
        model = gp.Model("SDP_Optimization", env=self.env)
        
        # Configuration for Global Optimization
        model.Params.NonConvex = 2 
        model.Params.Threads = 1
        model.Params.MIPGap = 0.0  
        model.Params.TimeLimit = 600.0

        # --- Variables ---
        f = model.addVar(lb=1.0/self.n, name="f")
        x = model.addVars(self.n + 1, lb=0.0, ub=1.0, name="x")
        y = model.addVars(self.n + 1, lb=0.0, ub=1.0, name="y")
        model.setObjective(f, GRB.MINIMIZE)

        # --- Constraints ---
        # 2c: Boundary points
        model.addConstr(x[self.n] == 1)
        model.addConstr(y[self.n] == 1)

        # 2d & 2e: Ordering constraints
        for i in range(self.n - 1):
            model.addConstr(x[i+1] - x[i] >= self.epsilon)
            model.addConstr(y[i+1] - y[i] >= self.epsilon)

        # --- Efficient Parameter Pre-calculation ---
        # sum_auv[i, j] stores Sum_{u=0 to i} Sum_{v=0 to j} a_uv
        sum_auv = np.cumsum(np.cumsum(a_matrix, axis=0), axis=1)
        # Vectors for quick O(1) calculation of weird_parameters
        col_sums = np.cumsum(a_matrix, axis=0)
        row_sums = np.cumsum(a_matrix, axis=1)

        # Constraint 2a: (1/n)*Sum(a_uv) - x_i*y_j <= f + (2 - Sum(a_uj) - Sum(a_iv))
        for i in range(self.n):
            for j in range(self.n):
                w_param = 2 - col_sums[i, j] - row_sums[i, j]
                model.addQConstr((1.0/self.n) * sum_auv[i, j] - x[i] * y[j] <= f + w_param)

        # Constraint 2b: -1/n * Sum(a_uv) + x_i*y_j <= f + weird_parameter_2b
        for i in range(self.n + 1):
            for j in range(self.n + 1):
                # Handles index shift u < i, v < j using 0 for boundary cases
                s_val = 0 if (i == 0 or j == 0) else sum_auv[i-1, j-1]
                # weird_parameter_2b: 2 - Sum_{u=0 to i-1} a_uj - Sum_{v=0 to j-1} a_iv
                # We use i-1 and j-1 to match the 'u < i' logic
                sum1 = col_sums[i-1, j] if i > 0 else 0
                sum2 = row_sums[i, j-1] if j > 0 else 0
                w_param_2b = 2 - sum1 - sum2
                
                model.addQConstr(-(1.0/self.n) * s_val + x[i] * y[j] <= f + w_param_2b)

        model.optimize()

        if model.Status == GRB.OPTIMAL:
            return model.ObjVal
        return 0.0

    def evaluate(self, permutation: np.ndarray) -> float:
        """Standard fitness evaluation for the EDA."""
        self.evaluations += 1
        binary_mat = self.transform_into_binary_matrix(permutation)
        return self.solve_optimization(binary_mat)

    def parse_and_validate(self, line: str, tolerance: float = 1e-6):
        parts = np.fromstring(line.replace('\t', ' '), sep=' ')
        if len(parts) < 9:
            return False, 0, 0
            
        permutation = parts[:8].astype(int)
        expected_sdp = parts[8]

        calculated_sdp = self.evaluate(permutation)
        diff = abs(calculated_sdp - expected_sdp)
        
        return (diff < tolerance), expected_sdp, calculated_sdp

# --- Main Test Loop ---
if __name__ == "__main__":
    sdp = SDPProblem(problem_size=8, max_evaluations=1000, epsilon=0.0001)
    fail_count = 0
    
#    with open("test-small.txt", "r") as f:
#        for idx, line in enumerate(f):
#            passed, expected, actual = sdp.parse_and_validate(line)
#            diff = abs(expected - actual)
#            status = "PASS" if passed else "FAIL"
#            if not passed: 
#                fail_count += 1
#            print(f"Line {idx+1}: {status} | Diff: {diff:.2e} | Exp: {expected:.8f} | Act: {actual:.8f}")

    minimum = 1000

    with open("test-small.txt", "r") as f:
        for line in f: 
            parts = np.fromstring(line.replace('\t', ' '), sep=' ')
            val = parts[8]
            minimum = min(val, minimum)

    print("=" * 30 )
    print("Fail count: ", fail_count)
    print("=" * 30 )
    
    print("=" * 30 )
    print("Min", minimum)
    print("=" * 30 )

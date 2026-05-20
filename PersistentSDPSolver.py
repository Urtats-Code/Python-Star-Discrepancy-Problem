
import numpy as np
from SDPProblem import SDPProblem  # Assuming your solver is in SDPProblem.py

class PersistentSDPSolver:

    def __init__(self, n, epsilon):
        self.n = n
        self.epsilon = epsilon
        self.env = gp.Env(empty=True)
        self.env.setParam("OutputFlag", 0)
        self.env.start()
        
        # Build the static parts of the model once
        self.model = gp.Model("SDP_Persistent", env=self.env)
        self.model.Params.NonConvex = 2
        self.model.Params.MIPGap = 1e-4  # Match C++ SDP.cpp logic
        
        self.f = self.model.addVar(lb=1.0/n, name="f")
        self.x = self.model.addVars(n + 1, lb=0.0, ub=1.0, name="x")
        self.y = self.model.addVars(n + 1, lb=0.0, ub=1.0, name="y")
        self.model.setObjective(self.f, GRB.MINIMIZE)
        
        # Add static constraints (2c, 2d, 2e)
        self.model.addConstr(self.x[n] == 1)
        self.model.addConstr(self.y[n] == 1)
        for i in range(n - 1):
            self.model.addConstr(self.x[i+1] - self.x[i] >= epsilon)
            self.model.addConstr(self.y[i+1] - self.y[i] >= epsilon)
        
        # Placeholders for dynamic constraints (2a, 2b)
        self.dynamic_constrs = []

    def solve(self, a_matrix):
        """Updates and solves the model without rebuilding it."""
        # Remove old dynamic constraints
        if self.dynamic_constrs:
            self.model.remove(self.dynamic_constrs)
            self.dynamic_constrs = []
            
        # Efficient parameter lookup (O(1) using prefix sums)
        sum_auv = np.cumsum(np.cumsum(a_matrix, axis=0), axis=1)
        col_sums = np.cumsum(a_matrix, axis=0)
        row_sums = np.cumsum(a_matrix, axis=1)

        # Re-add only the constraints that depend on a_matrix
        for i in range(self.n):
            for j in range(self.n):
                w_param = 2 - col_sums[i, j] - row_sums[i, j]
                c = self.model.addQConstr((1.0/self.n)*sum_auv[i,j] - self.x[i]*self.y[j] <= self.f + w_param)
                self.dynamic_constrs.append(c)
        
        self.model.optimize()
        return self.model.ObjVal


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

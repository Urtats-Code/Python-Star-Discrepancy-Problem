
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from SDPProblem import SDPProblem  # Assuming your solver is in SDPProblem.py

class PersistentSDPProblem:
    def __init__(self, n, epsilon):
        self.n = n
        self.epsilon = epsilon
        
        # Start isolated environment
        self.env = gp.Env(empty=True)
        self.env.setParam("OutputFlag", 0)
        self.env.start()
        
        # Build persistent model
        self.model = gp.Model("SDP_Persistent", env=self.env)
        self.model.Params.NonConvex = 2
        self.model.Params.MIPGap = 1e-4
        self.model.Params.Threads = 1  # Limit internal threads since we parallelize externally
        
        # Static variables
        self.f = self.model.addVar(lb=1.0/n, name="f")
        self.x = self.model.addVars(n + 1, lb=0.0, ub=1.0, name="x")
        self.y = self.model.addVars(n + 1, lb=0.0, ub=1.0, name="y")
        self.model.setObjective(self.f, GRB.MINIMIZE)
        
        # Static constraints (2c, 2d, 2e)
        self.model.addConstr(self.x[n] == 1)
        self.model.addConstr(self.y[n] == 1)
        for i in range(n - 1):
            self.model.addConstr(self.x[i+1] - self.x[i] >= epsilon)
            self.model.addConstr(self.y[i+1] - self.y[i] >= epsilon)
            
        self.dynamic_constrs = []

    def transform_into_binary_matrix(self, solution):
        matrix = np.zeros((self.n + 1, self.n + 1), dtype=int)
        rows = np.arange(self.n)
        cols = solution.astype(int) - 1
        matrix[rows, cols] = 1
        matrix[0, self.n] = 1
        matrix[self.n, 0] = 1
        return matrix

    def evaluate(self, permutation):
        a_matrix = self.transform_into_binary_matrix(permutation)

        # 1. Remove old dynamic constraints
        if self.dynamic_constrs:
            self.model.remove(self.dynamic_constrs)
            self.dynamic_constrs.clear()

        # 2. O(1) Prefix sums for parameter lookup
        sum_auv = np.cumsum(np.cumsum(a_matrix, axis=0), axis=1)
        col_sums = np.cumsum(a_matrix, axis=0)
        row_sums = np.cumsum(a_matrix, axis=1)

        # 3. Add dynamic constraints (2a)
        for i in range(self.n):
            for j in range(self.n):
                w_param = 2 - col_sums[i, j] - row_sums[i, j]
                c = self.model.addQConstr((1.0/self.n)*sum_auv[i,j] - self.x[i]*self.y[j] <= self.f + w_param)
                self.dynamic_constrs.append(c)

        # 4. Add dynamic constraints (2b)
        for i in range(self.n + 1):
            for j in range(self.n + 1):
                s_val = 0 if (i == 0 or j == 0) else sum_auv[i-1, j-1]
                sum1 = col_sums[i-1, j] if i > 0 else 0
                sum2 = row_sums[i, j-1] if j > 0 else 0
                w_param_2b = 2 - sum1 - sum2
                
                c = self.model.addQConstr(-(1.0/self.n) * s_val + self.x[i]*self.y[j] <= self.f + w_param_2b)
                self.dynamic_constrs.append(c)

        # 5. Solve and return
        self.model.optimize()
        if self.model.Status == GRB.OPTIMAL:
            return self.model.ObjVal
        return float('inf')

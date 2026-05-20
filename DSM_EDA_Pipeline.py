import numpy as np
import time
import gurobipy as gp
from gurobipy import GRB
import concurrent.futures
import multiprocessing
from enum import Enum

# ==========================================
# 0. Enums for Configuration
# ==========================================
class LearningStrategy(Enum):
    BIRKHOFF = "birkhoff"
    PBIL = "pbil"

# ==========================================
# 1. Persistent SDP Solver
# ==========================================
class PersistentSDPProblem:
    def __init__(self, n, epsilon):
        self.n = n
        self.epsilon = epsilon
        
        self.env = gp.Env(empty=True)
        self.env.setParam("OutputFlag", 0)
        self.env.start()
        
        self.model = gp.Model("SDP_Persistent", env=self.env)
        self.model.Params.NonConvex = 2
        self.model.Params.MIPGap = 1e-4
        self.model.Params.Threads = 1
        
        self.f = self.model.addVar(lb=1.0/n, name="f")
        self.x = self.model.addVars(n + 1, lb=0.0, ub=1.0, name="x")
        self.y = self.model.addVars(n + 1, lb=0.0, ub=1.0, name="y")
        self.model.setObjective(self.f, GRB.MINIMIZE)
        
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

        if self.dynamic_constrs:
            self.model.remove(self.dynamic_constrs)
            self.dynamic_constrs.clear()

        sum_auv = np.cumsum(np.cumsum(a_matrix, axis=0), axis=1)
        col_sums = np.cumsum(a_matrix, axis=0)
        row_sums = np.cumsum(a_matrix, axis=1)

        for i in range(self.n):
            for j in range(self.n):
                w_param = 2 - col_sums[i, j] - row_sums[i, j]
                c = self.model.addQConstr((1.0/self.n)*sum_auv[i,j] - self.x[i]*self.y[j] <= self.f + w_param)
                self.dynamic_constrs.append(c)

        for i in range(self.n + 1):
            for j in range(self.n + 1):
                s_val = 0 if (i == 0 or j == 0) else sum_auv[i-1, j-1]
                sum1 = col_sums[i-1, j] if i > 0 else 0
                sum2 = row_sums[i, j-1] if j > 0 else 0
                w_param_2b = 2 - sum1 - sum2
                
                c = self.model.addQConstr(-(1.0/self.n) * s_val + self.x[i]*self.y[j] <= self.f + w_param_2b)
                self.dynamic_constrs.append(c)

        self.model.optimize()
        if self.model.Status == GRB.OPTIMAL:
            return self.model.ObjVal
        return float('inf')

# ==========================================
# 2. Worker Initialization
# ==========================================
global_solver = None

def init_worker(n, epsilon):
    global global_solver
    global_solver = PersistentSDPProblem(n, epsilon)

def worker_evaluate(permutation):
    return global_solver.evaluate(permutation)

# ==========================================
# 3. Cache-Optimized Pipeline with Strategies
# ==========================================
class DSM_EDA_Pipeline:
    def __init__(self, n, population_size, selection_size, strategy: LearningStrategy):
        self.n = n
        self.population_size = population_size
        self.selection_size = selection_size
        self.strategy = strategy
        self.dsm = np.full((n, n), 1.0 / n) 
        self.best_fitness = float('inf')
        self.best_solution = None
        
        # Initialize Cache and Tracking Metrics
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

    def evaluate_population_parallel(self, population, executor):
        fitness_scores = [None] * len(population)
        uncached_permutations = []
        uncached_indices = []

        # 1. Check Cache
        for i, p in enumerate(population):
            p_tuple = tuple(p)  
            if p_tuple in self.cache:
                fitness_scores[i] = self.cache[p_tuple]
                self.cache_hits += 1
            else:
                uncached_permutations.append(p)
                uncached_indices.append(i)
                self.cache_misses += 1

        # 2. Evaluate Uncached Permutations
        if uncached_permutations:
            new_scores = list(executor.map(worker_evaluate, uncached_permutations))
            
            for idx, p, score in zip(uncached_indices, uncached_permutations, new_scores):
                fitness_scores[idx] = score
                self.cache[tuple(p)] = score
        
        # 3. Sort and Return
        sorted_indices = np.argsort(fitness_scores)
        return [population[i] for i in sorted_indices], [fitness_scores[i] for i in sorted_indices]

    def sinkhorn_knopp(self, matrix, iterations=10):
        mat = matrix.copy()
        for _ in range(iterations):
            mat /= (mat.sum(axis=1, keepdims=True) + 1e-15)
            mat /= (mat.sum(axis=0, keepdims=True) + 1e-15)
        return mat

    def learn(self, selected_permutations, best_permutation, learning_rate=0.1):
        """Dispatches the learning update to the chosen strategy."""
        if self.strategy == LearningStrategy.BIRKHOFF:
            self._learn_birkhoff(selected_permutations, learning_rate)
        elif self.strategy == LearningStrategy.PBIL:
            self._learn_pbil(best_permutation, learning_rate)
        else:
            raise ValueError(f"Unknown Learning Strategy: {self.strategy}")

    def _learn_birkhoff(self, selected_permutations, alpha):
        """Birkhoff Learning: Updates DSM based on a population of top selected permutations."""
        uniform_val = alpha / self.n
        new_info = np.zeros((self.n, self.n))
        
        num_selected = len(selected_permutations)
        rows = np.tile(np.arange(self.n), num_selected)
        cols = np.array(selected_permutations).flatten() - 1  # 0-indexing
        
        np.add.at(new_info, (rows, cols), 1.0)
        
        weight = (1.0 - alpha) / num_selected
        new_info = (new_info * weight) + uniform_val
        
        updated_dsm = (1 - alpha) * self.dsm + new_info
        self.dsm = self.sinkhorn_knopp(updated_dsm)

    def _learn_pbil(self, best_permutation, learning_rate):
        """PBIL Learning: Shifts the DSM towards the absolute best solution found so far."""
        # Convert the best permutation into a binary indicator matrix
        best_dsm_matrix = np.zeros((self.n, self.n))
        for i, val in enumerate(best_permutation):
            best_dsm_matrix[i, int(val) - 1] = 1.0

        # PBIL Update: (1 - LR) * old_dsm + LR * best_dsm
        updated_dsm = ((1.0 - learning_rate) * self.dsm) + (learning_rate * best_dsm_matrix)
        
        self.dsm = self.sinkhorn_knopp(updated_dsm)

    def sample_permutation(self):
        perm = np.zeros(self.n, dtype=int)
        available_cols = list(range(self.n))
        for i in range(self.n):
            probs = self.dsm[i, available_cols]
            probs /= (probs.sum() + 1e-15)
            choice_idx = np.random.choice(len(available_cols), p=probs)
            perm[i] = available_cols.pop(choice_idx) + 1
        return perm

    def run(self, epsilon=0.0001, generations=50, num_workers=4, learning_rate=0.1):
        population = [np.random.permutation(self.n) + 1 for _ in range(self.population_size)]

        start_time = time.time()
        
        print(f"Starting EDA with {num_workers} workers using {self.strategy.name} strategy...")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=init_worker, 
            initargs=(self.n, epsilon)
        ) as executor:
            
            for gen in range(generations):
                # Parallel Evaluation with Cache
                sorted_pop, sorted_fitness = self.evaluate_population_parallel(population, executor)
                
                # Track Best
                if sorted_fitness[0] < self.best_fitness:
                    self.best_fitness = sorted_fitness[0]
                    self.best_solution = sorted_pop[0]
                
                # Learning & Sampling
                selected = sorted_pop[:self.selection_size]
                
                self.learn(selected, self.best_solution, learning_rate)
                
                population = [self.sample_permutation() for _ in range(self.population_size)]
                
                # Calculate current hit rate
                total_evals = self.cache_hits + self.cache_misses
                hit_rate = (self.cache_hits / total_evals * 100) if total_evals > 0 else 0
                
                print(f"Gen {gen:03d} | Best Fitness: {self.best_fitness:.8f} | Hit Rate: {hit_rate:.1f}%")

        end_time = time.time()
        elapsed_time = end_time - start_time       

        return self.best_solution, self.best_fitness, elapsed_time, self.cache_misses

if __name__ == "__main__":
    N = 20  
    POP_SIZE = 200
    SELECTION_SIZE = 20
    GENERATIONS = 500
    EPSILON = 0.0001
    LEARNING_RATE = 0.1
    
    WORKERS = max(1, multiprocessing.cpu_count() - 1)
    
    chosen_strategy = LearningStrategy.BIRKHOFF
    
    pipeline = DSM_EDA_Pipeline(
        n=N, 
        population_size=POP_SIZE, 
        selection_size=SELECTION_SIZE,
        strategy=chosen_strategy
    )
    
    best_p, best_f, total_time, gurobi_calls = pipeline.run(
        epsilon=EPSILON, 
        generations=GENERATIONS, 
        num_workers=WORKERS, 
        learning_rate=LEARNING_RATE
    )

    print("\n" + "="*40)
    print("--- Final EDA Results ---")
    print("="*40)
    print(f"Best Permutation:   {best_p}")
    print(f"Best Fitness:       {best_f:.8f}")
    print("-" * 40)
    print(f"Total Time Elapsed: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Total Gurobi Calls: {gurobi_calls} (Actual solver evaluations)")
    print(f"Total Cache Hits:   {pipeline.cache_hits} (Saved solver evaluations)")
    
    total_requested_evals = pipeline.cache_hits + pipeline.cache_misses
    if total_requested_evals > 0:
        overall_saved = (pipeline.cache_hits / total_requested_evals) * 100
        print(f"Overall Cache Savings: {overall_saved:.1f}%")

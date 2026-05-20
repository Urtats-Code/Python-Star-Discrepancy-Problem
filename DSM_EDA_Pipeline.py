import numpy as np
import gurobipy as gp
from gurobipy import GRB
import concurrent.futures
from PersistentSDPSolver import PersistentSDPProblem

# ==========================================
# 2. Multiprocessing Worker Setup
# ==========================================
global_solver = None

def init_worker(n, epsilon):
    """Initializes one Persistent Gurobi model per CPU Core."""
    global global_solver
    global_solver = PersistentSDPProblem(n, epsilon)

def worker_evaluate(permutation):
    """Worker function that uses its local solver."""
    return global_solver.evaluate(permutation)

# ==========================================
# 3. The EDA Pipeline
# ==========================================
class DSM_EDA_Pipeline:
    def __init__(self, n, population_size, selection_size):
        self.n = n
        self.population_size = population_size
        self.selection_size = selection_size
        self.dsm = np.full((n, n), 1.0 / n) 
        self.best_fitness = float('inf')
        self.best_solution = None

    def evaluate_population_parallel(self, population, executor):
        """Uses the multiprocessing pool to evaluate the population."""
        # Map population to workers
        fitness_scores = list(executor.map(worker_evaluate, population))
        
        # Sort population by fitness
        sorted_indices = np.argsort(fitness_scores)
        return [population[i] for i in sorted_indices], [fitness_scores[i] for i in sorted_indices]

    def sinkhorn_knopp(self, matrix, iterations=10):
        mat = matrix.copy()
        for _ in range(iterations):
            mat /= (mat.sum(axis=1, keepdims=True) + 1e-15)
            mat /= (mat.sum(axis=0, keepdims=True) + 1e-15)
        return mat

    def learn(self, selected_permutations, alpha=0.1):
        """Vectorized Birkhoff Learning."""
        uniform_val = alpha / self.n
        new_info = np.zeros((self.n, self.n))
        
        # Vectorized accumulation
        num_selected = len(selected_permutations)
        rows = np.tile(np.arange(self.n), num_selected)
        cols = np.array(selected_permutations).flatten() - 1  # 0-indexing
        
        np.add.at(new_info, (rows, cols), 1.0)
        
        # Calculate weighted info
        weight = (1.0 - alpha) / num_selected
        new_info = (new_info * weight) + uniform_val
        
        # Stabilize via Sinkhorn
        updated_dsm = (1 - alpha) * self.dsm + new_info
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

    def run(self, epsilon=0.0001, generations=50, num_workers=4):
        # Initialize random population
        population = [np.random.permutation(self.n) + 1 for _ in range(self.population_size)]
        
        # Create a ProcessPoolExecutor to handle evaluations in parallel
        print(f"Starting EDA with {num_workers} parallel workers...")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=init_worker, 
            initargs=(self.n, epsilon)
        ) as executor:
            
            for gen in range(generations):
                # 1. Parallel Evaluation
                sorted_pop, sorted_fitness = self.evaluate_population_parallel(population, executor)
                
                # 2. Track Best
                if sorted_fitness[0] < self.best_fitness:
                    self.best_fitness = sorted_fitness[0]
                    self.best_solution = sorted_pop[0]
                
                # 3. Learning & Sampling
                selected = sorted_pop[:self.selection_size]
                self.learn(selected)
                population = [self.sample_permutation() for _ in range(self.population_size)]
                
                print(f"Gen {gen:03d} | Best Fitness: {self.best_fitness:.8f}")
                
        return self.best_solution, self.best_fitness

# ==========================================
# 4. Execution Block
# ==========================================
if __name__ == "__main__":
    import multiprocessing
    
    # Configuration
    N = 20  
    POP_SIZE = 20
    SELECTION_SIZE = 5
    GENERATIONS = 100
    EPSILON = 0.0001
    
    WORKERS = max(1, multiprocessing.cpu_count() - 1)

    pipeline = DSM_EDA_Pipeline(n=N, population_size=POP_SIZE, selection_size=SELECTION_SIZE)
    
    best_p, best_f = pipeline.run(epsilon=EPSILON, generations=GENERATIONS, num_workers=WORKERS)

    print("\n--- Final Result ---")
    print(f"Best Permutation: {best_p}")
    print(f"Best Fitness:     {best_f}")

import numpy as np
import gurobipy as gp
from gurobipy import GRB
import concurrent.futures
from PersistentSDPSolver import PersistentSDPProblem
from PermutationHashMap import PermutationHashMap

global_solver = None

def init_worker(n, epsilon):
    global global_solver
    global_solver = PersistentSDPProblem(n, epsilon)

def worker_evaluate(permutation):
    return global_solver.evaluate(permutation)

class DSM_EDA_Pipeline:
    def __init__(self, n, population_size, selection_size):
        self.n = n
        self.population_size = population_size
        self.selection_size = selection_size
        self.dsm = np.full((n, n), 1.0 / n)
        self.best_fitness = float('inf')
        self.best_solution = None
        self.cache = PermutationHashMap()
        self.dsm_corrections = 0

    def evaluate_population_parallel(self, population, executor):
        uncached = [p for p in population if not self.cache.contains(p)]
        if uncached:
            new_scores = list(executor.map(worker_evaluate, uncached))
            for perm, score in zip(uncached, new_scores):
                self.cache.set(perm, score)

        fitness_scores = [self.cache.get(p) for p in population]

        sorted_indices = np.argsort(fitness_scores)
        return [population[i] for i in sorted_indices], [fitness_scores[i] for i in sorted_indices]

    def _is_doubly_stochastic(self, matrix, tol=1e-6):
        if np.any(matrix < 0):
            return False
        if not np.allclose(matrix.sum(axis=1), 1.0, atol=tol):
            return False
        if not np.allclose(matrix.sum(axis=0), 1.0, atol=tol):
            return False
        return True

    def sinkhorn_knopp(self, matrix, iterations=10):
        mat = matrix.copy()
        for _ in range(iterations):
            mat /= (mat.sum(axis=1, keepdims=True) + 1e-15)
            mat /= (mat.sum(axis=0, keepdims=True) + 1e-15)
        return mat

    def learn(self, selected_permutations, alpha=0.1):
        m = len(selected_permutations)
        
        self.dsm = np.full((self.n, self.n), alpha / self.n)
        
        w = (1.0 - alpha) / m
        
        freq = np.zeros((self.n, self.n))
        rows = np.tile(np.arange(self.n), m)
        cols = np.array(selected_permutations).flatten() - 1  # 0-indexing
        
        np.add.at(freq, (rows, cols), 1.0)
        
        self.dsm += freq * w

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
                
        print(f"DSM corrections needed: {self.dsm_corrections}")
        return self.best_solution, self.best_fitness

if __name__ == "__main__":
    import multiprocessing
    
    # Configuration
    N = 8 
    POP_SIZE = 200
    SELECTION_SIZE = 20
    GENERATIONS = 100
    EPSILON = 0.0001
    
    WORKERS = max(1, multiprocessing.cpu_count() - 1)

    pipeline = DSM_EDA_Pipeline(n=N, population_size=POP_SIZE, selection_size=SELECTION_SIZE)
    
    best_p, best_f = pipeline.run(epsilon=EPSILON, generations=GENERATIONS, num_workers=WORKERS)

    print("\n--- Final Result ---")
    print(f"Best Permutation: {best_p}")
    print(f"Best Fitness:     {best_f}")

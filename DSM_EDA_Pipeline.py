import numpy as np
from enum import Enum
import gurobipy as gp
from gurobipy import GRB
import concurrent.futures
import os
import yaml
import argparse
from PersistentSDPSolver import PersistentSDPProblem
from PermutationHashMap import PermutationHashMap
from GenerationLogger import GenerationLogger
from Visualize import Visualize

global_solver = None

class LearningStrategy(Enum):
    BIRKHOFF = "birkhoff"
    PBIL = "pbil"

def init_worker(n, epsilon):
    global global_solver
    global_solver = PersistentSDPProblem(n, epsilon)

def worker_evaluate(permutation):
    return global_solver.evaluate(permutation)

class DSM_EDA_Pipeline:
    def __init__(self, n, population_size, selection_size, learning_method, gen_log_file="generation_log.h5"):
        self.n = n
        self.learning_method = learning_method
        self.population_size = population_size
        self.selection_size = max(1, int(population_size * selection_size / 100))
        self.dsm = np.full((n, n), 1.0 / n)
        self.best_fitness = float('inf')
        self.best_solution = None
        self.cache = PermutationHashMap()
        self.dsm_corrections = 0
        self._gen_log_file = gen_log_file

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

    # Applies the Sinkhorn-Knopp algorithm to convert this matrix into a Doubly Stochastic Matrix (DSM).
    # Zero rows/columns (caused by zeroing out singular values) are replaced with a uniform distribution
    # before normalizing, so that the algorithm can still converge.
    def sinkhorn_knopp(self, matrix, tol=1e-6, max_iter=10000):
        result = matrix.copy().astype(float)
        n_rows, n_cols = result.shape
        for _ in range(max_iter):
            # Normalize rows — replace zero rows with uniform 1/n_cols
            row_sums = result.sum(axis=1, keepdims=True)
            zero_rows = (row_sums == 0).flatten()
            result[zero_rows] = 1.0 / n_cols
            row_sums[zero_rows] = 1.0
            result /= row_sums

            # Normalize columns — replace zero columns with uniform 1/n_rows
            col_sums = result.sum(axis=0, keepdims=True)
            zero_cols = (col_sums == 0).flatten()
            result[:, zero_cols] = 1.0 / n_rows
            col_sums[:, zero_cols] = 1.0
            result /= col_sums

            # Check convergence: rows and columns should all sum to 1
            row_err = np.max(np.abs(result.sum(axis=1) - 1))
            col_err = np.max(np.abs(result.sum(axis=0) - 1))
            if row_err < tol and col_err < tol:
                break
        return result

    def learn_pbil(self, selected_permutations, learning_rate=0.1, mutation_rate=0.01):
        m = len(selected_permutations)
        
        target_matrix = np.zeros((self.n, self.n))
        rows = np.tile(np.arange(self.n), m)
        cols = np.array(selected_permutations).flatten() - 1  # 0-indexing
        
        np.add.at(target_matrix, (rows, cols), 1.0 / m)
        
        self.dsm = (1.0 - learning_rate) * self.dsm + learning_rate * target_matrix
        
        if mutation_rate > 0:
            uniform_matrix = 1.0 / self.n
            self.dsm = (1.0 - mutation_rate) * self.dsm + mutation_rate * uniform_matrix

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
        self._gen_logger = GenerationLogger(self._gen_log_file, metadata={
            "n":               self.n,
            "population_size": self.population_size,
            "selection_size":  self.selection_size,
            "learning_method": self.learning_method.value,
            "generations":     generations,
            "epsilon":         epsilon,
            "num_workers":     num_workers,
        })
        self._gen_log_file = self._gen_logger.file_path

        # Initialize random population
        population = [np.random.permutation(self.n) + 1 for _ in range(self.population_size)]

        # Create a ProcessPoolExecutor to handle evaluations in parallel
        print(f"Starting EDA with {num_workers} parallel workers...")
        try:
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
                    
                    if(self.learning_method == LearningStrategy.BIRKHOFF):
                        self.learn(selected, alpha=getattr(self, "birkhoff_alpha", 0.1))

                    if(self.learning_method == LearningStrategy.PBIL):
                        self.learn_pbil(selected, learning_rate=getattr(self, "pbil_learning_rate", 0.1), mutation_rate=getattr(self, "pbil_mutation_rate", 0.01))

                    self._gen_logger.log({
                        "generation":   gen,
                        "population":   np.array(sorted_pop,     dtype=np.int64),
                        "fitness":      np.array(sorted_fitness,  dtype=np.float64),
                        "best_fitness": float(self.best_fitness),
                        "dsm":          self.dsm.copy(),
                    })

                    population = [self.sample_permutation() for _ in range(self.population_size)]
                    
                    print(f"Gen {gen:03d} | Best Fitness: {self.best_fitness:.8f}")
        finally:
            self._gen_logger.close()
        
        print(f"DSM corrections needed: {self.dsm_corrections}")
        return self.best_solution, self.best_fitness

    def visualize(self, output_dir="plots", plot_fitness=True, plot_dsm=True, plot_combined=True, save_animation=False, save_combined_animation=False):
        """
        Generates visualizations from the logged data.
        """
        if not os.path.exists(self._gen_log_file):
            print(f"Cannot visualize: {self._gen_log_file} not found.")
            return

        print("\n--- Generating Visualizations ---")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        viz = Visualize(self._gen_log_file)
        
        if plot_fitness:
            viz.plot_fitness(save_path=os.path.join(output_dir, "fitness_evolution.png"))
            
        if plot_dsm:
            viz.plot_best_dsm(save_path=os.path.join(output_dir, "final_dsm.png"))
            
        if plot_combined:
            viz.plot_dsm_with_fitness(save_path=os.path.join(output_dir, "combined_viz.png"))
            
        if save_animation:
            viz.plot_dsm_history(save_mp4=True, filename=os.path.join(output_dir, "dsm_evolution.mp4"))

        if save_combined_animation:
            viz.plot_dsm_with_fitness_history(save_mp4=True, filename=os.path.join(output_dir, "dsm_fitness_evolution.mp4"))

def main(config_path: str):
    import multiprocessing

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    p_cfg = cfg["pipeline"]
    r_cfg = cfg["run"]
    pbil_cfg = cfg.get("pbil", {})
    birkhoff_cfg = cfg.get("birkhoff", {})
    log_cfg = cfg.get("logging", {})
    visualize_cfg = cfg.get("visualization", {})

    N = p_cfg["n"]
    POP_SIZE = p_cfg["population_size"]
    SELECTION_SIZE = p_cfg["selection_size"]
    LEARNING_METHOD = LearningStrategy(p_cfg["learning_method"])

    GENERATIONS = r_cfg["generations"]
    EPSILON = r_cfg["epsilon"]
    WORKERS = r_cfg.get("num_workers") or max(1, multiprocessing.cpu_count() - 1)

    gen_log_file = log_cfg.get("gen_log_file", "generation_log.h5")
    gen_log_dir = os.path.dirname(gen_log_file)
    if gen_log_dir and not os.path.isdir(gen_log_dir):
        os.makedirs(gen_log_dir, exist_ok=True)

    pipeline = DSM_EDA_Pipeline(n=N, population_size=POP_SIZE, selection_size=SELECTION_SIZE, learning_method=LEARNING_METHOD, gen_log_file=gen_log_file)

    if LEARNING_METHOD == LearningStrategy.PBIL:
        pipeline.pbil_learning_rate = pbil_cfg.get("learning_rate", 0.1)
        pipeline.pbil_mutation_rate = pbil_cfg.get("mutation_rate", 0.01)
    if LEARNING_METHOD == LearningStrategy.BIRKHOFF:
        pipeline.birkhoff_alpha = birkhoff_cfg.get("alpha", 0.1)

    best_p, best_f = pipeline.run(epsilon=EPSILON, generations=GENERATIONS, num_workers=WORKERS)

    if visualize_cfg.get("enabled", False):
        pipeline.visualize(
            output_dir=visualize_cfg.get("output_dir", "plots"),
            plot_fitness=visualize_cfg.get("plot_fitness", True),
            plot_dsm=visualize_cfg.get("plot_dsm", True),
            plot_combined=visualize_cfg.get("plot_combined", True),
            save_animation=visualize_cfg.get("save_animation", False),
            save_combined_animation=visualize_cfg.get("save_combined_animation", False)
        )

    print("\n--- Final Result ---")
    print(f"Best Permutation: {best_p}")
    print(f"Best Fitness:     {best_f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DSM EDA Pipeline execution.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config YAML")
    args = parser.parse_args()
    main(args.config)

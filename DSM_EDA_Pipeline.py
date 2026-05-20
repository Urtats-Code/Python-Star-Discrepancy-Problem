import numpy as np
from enum import Enum
import gurobipy as gp
from gurobipy import GRB
import concurrent.futures
import threading
import queue
import os
import yaml
from PersistentSDPSolver import PersistentSDPProblem
from PermutationHashMap import PermutationHashMap
from GenerationLogger import GenerationLogger

global_solver = None

class LearningStrategy(Enum):                                                                                                                                                                                        
    BIRKHOFF = "birkhoff"                                                                                                                                                                                            
    PBIL = "pbil"      

def init_worker(n, epsilon):
    global global_solver
    global_solver = PersistentSDPProblem(n, epsilon)

def worker_evaluate(permutation):
    return global_solver.evaluate(permutation)

def _learn_logger(log_queue, log_file):
    with open(log_file, "w") as f:
        f.write("call,generation,method\n")
        while True:
            item = log_queue.get()
            if item is None:
                break
            f.write(f"{item['call']},{item['generation']},{item['method']}\n")
            f.flush()


class DSM_EDA_Pipeline:
    def __init__(self, n, population_size, selection_size, learning_method, log_file="learn_log.csv", gen_log_file="generation_log.h5"):
        self.n = n
        self.learning_method = learning_method
        self.population_size = population_size
        self.selection_size = selection_size
        self.dsm = np.full((n, n), 1.0 / n)
        self.best_fitness = float('inf')
        self.best_solution = None
        self.cache = PermutationHashMap()
        self.dsm_corrections = 0
        self._learn_call_count = 0
        self._log_queue = queue.Queue()
        self._log_thread = threading.Thread(
            target=_learn_logger,
            args=(self._log_queue, log_file),
            daemon=True,
        )
        self._log_thread.start()
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

    def sinkhorn_knopp(self, matrix, iterations=10):
        mat = matrix.copy()
        for _ in range(iterations):
            mat /= (mat.sum(axis=1, keepdims=True) + 1e-15)
            mat /= (mat.sum(axis=0, keepdims=True) + 1e-15)
        return mat

    def _log_learn(self, method):
        self._learn_call_count += 1
        self._log_queue.put({
            "call": self._learn_call_count,
            "generation": getattr(self, "_current_gen", -1),
            "method": method,
        })

    def learn_pbil(self, selected_permutations, learning_rate=0.1, mutation_rate=0.01):
        self._log_learn("pbil")
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
        self._log_learn("birkhoff")
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
                self._current_gen = gen
                # 1. Parallel Evaluation
                sorted_pop, sorted_fitness = self.evaluate_population_parallel(population, executor)
                
                # 2. Track Best
                if sorted_fitness[0] < self.best_fitness:
                    self.best_fitness = sorted_fitness[0]
                    self.best_solution = sorted_pop[0]
                
                # 3. Learning & Sampling
                selected = sorted_pop[:self.selection_size]
                
                if(self.learning_method == LearningStrategy.BIRKHOFF):
                    self.learn(selected)

                if(self.learning_method == LearningStrategy.PBIL):
                    self.learn_pbil(selected)

                self._gen_logger.log({
                    "generation":   gen,
                    "population":   np.array(sorted_pop,     dtype=np.int64),
                    "fitness":      np.array(sorted_fitness,  dtype=np.float64),
                    "best_fitness": float(self.best_fitness),
                    "dsm":          self.dsm.copy(),
                })

                population = [self.sample_permutation() for _ in range(self.population_size)]
                
                print(f"Gen {gen:03d} | Best Fitness: {self.best_fitness:.8f}")
                
        self._log_queue.put(None)
        self._log_thread.join()
        self._gen_logger.close()
        print(f"DSM corrections needed: {self.dsm_corrections}")
        return self.best_solution, self.best_fitness

if __name__ == "__main__":
    import multiprocessing

    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    p_cfg = cfg["pipeline"]
    r_cfg = cfg["run"]
    pbil_cfg = cfg.get("pbil", {})
    birkhoff_cfg = cfg.get("birkhoff", {})
    log_cfg = cfg.get("logging", {})

    N = p_cfg["n"]
    POP_SIZE = p_cfg["population_size"]
    SELECTION_SIZE = p_cfg["selection_size"]
    LEARNING_METHOD = LearningStrategy(p_cfg["learning_method"])

    GENERATIONS = r_cfg["generations"]
    EPSILON = r_cfg["epsilon"]
    WORKERS = r_cfg["num_workers"] or max(1, multiprocessing.cpu_count() - 1)

    log_file = log_cfg.get("log_file", "learn_log.csv")
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.isdir(log_dir):
        raise ValueError(f"Log directory does not exist: {log_dir}")
    if os.path.exists(log_file):
        os.remove(log_file)

    gen_log_file = log_cfg.get("gen_log_file", "generation_log.h5")
    gen_log_dir = os.path.dirname(gen_log_file)
    if gen_log_dir and not os.path.isdir(gen_log_dir):
        raise ValueError(f"Generation log directory does not exist: {gen_log_dir}")

    pipeline = DSM_EDA_Pipeline(n=N, population_size=POP_SIZE, selection_size=SELECTION_SIZE, learning_method=LEARNING_METHOD, log_file=log_file, gen_log_file=gen_log_file)

    if LEARNING_METHOD == LearningStrategy.PBIL:
        pipeline.pbil_learning_rate = pbil_cfg.get("learning_rate", 0.1)
        pipeline.pbil_mutation_rate = pbil_cfg.get("mutation_rate", 0.01)
    if LEARNING_METHOD == LearningStrategy.BIRKHOFF:
        pipeline.birkhoff_alpha = birkhoff_cfg.get("alpha", 0.1)

    best_p, best_f = pipeline.run(epsilon=EPSILON, generations=GENERATIONS, num_workers=WORKERS)

    print("\n--- Final Result ---")
    print(f"Best Permutation: {best_p}")
    print(f"Best Fitness:     {best_f}")

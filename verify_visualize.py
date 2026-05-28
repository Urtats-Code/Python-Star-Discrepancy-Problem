from Visualize import Visualize
import os

def test():
    log_file = "logs/generation_log.h5"
    if not os.path.exists(log_file):
        print(f"Skipping test: {log_file} not found.")
        return

    viz = Visualize()
    print("Parsing log file...")
    data = viz.parse(log_file)
    print("Metadata:", data["metadata"])
    print("Number of generations:", len(data["generations"]))

    print("Testing plot_fitness...")
    viz.plot_fitness(data, save_path="test_fitness.png")

    print("Testing plot_best_dsm...")
    viz.plot_best_dsm(data, save_path="test_dsm.png")

    print("Testing plot_dsm_with_fitness...")
    viz.plot_dsm_with_fitness(data, save_path="test_combined.png")

    print("Testing plot_dsm_history (saving MP4)...")
    viz.plot_dsm_history(data, save_mp4=True, filename="test_dsm_history.mp4")

    print("Verification complete. Check test_*.png and test_*.mp4 files.")

if __name__ == "__main__":
    test()

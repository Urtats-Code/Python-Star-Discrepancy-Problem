import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

class Visualize:
    def __init__(self, h5_path=None):
        self.data = None
        if h5_path:
            self.data = self.parse(h5_path)

    def parse(self, h5_path):
        """
        Parses the HDF5 log file and returns a structured dictionary.
        """
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"Log file not found: {h5_path}")

        data = {
            "metadata": {},
            "generations": [],
            "dsm_history": [],
            "best_fitness_history": []
        }

        with h5py.File(h5_path, "r") as f:
            # Parse metadata
            if "metadata" in f:
                meta_grp = f["metadata"]
                for key in meta_grp:
                    val = meta_grp[key][()]
                    if isinstance(val, bytes):
                        val = val.decode()
                    data["metadata"][key] = val

            # Parse generations
            gen_keys = sorted([k for k in f.keys() if k.startswith("generation_")])
            for key in gen_keys:
                grp = f[key]
                gen_idx = int(key.split("_")[1])
                data["generations"].append(gen_idx)
                data["dsm_history"].append(grp["dsm"][()])
                data["best_fitness_history"].append(grp["best_fitness"][()])

        # Convert to numpy arrays for easier handling
        data["dsm_history"] = np.array(data["dsm_history"])
        data["best_fitness_history"] = np.array(data["best_fitness_history"])
        return data

    def plot_fitness(self, data=None, save_path=None):
        """
        Plots the evolution of the best fitness.
        """
        d = data or self.data
        if d is None:
            print("No data to plot.")
            return

        fig = plt.figure(figsize=(10, 6))
        plt.plot(d["generations"], d["best_fitness_history"], marker='.', linestyle='-')
        plt.title("Best Fitness Evolution")
        plt.xlabel("Generation")
        plt.ylabel("Best Fitness")
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
        plt.close(fig)

    def plot_best_dsm(self, data=None, save_path=None):
        """
        Plots the DSM from the final generation.
        """
        d = data or self.data
        if d is None:
            print("No data to plot.")
            return

        best_dsm = d["dsm_history"][-1]
        n = d["metadata"].get("n", best_dsm.shape[0])

        fig = plt.figure(figsize=(8, 8))
        im = plt.imshow(best_dsm, cmap='viridis', interpolation='nearest')
        plt.colorbar(im, label='Probability')
        plt.title(f"Final DSM (Generation {d['generations'][-1]}) - n={n}")
        plt.xlabel("Column Index")
        plt.ylabel("Row Index")
        
        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
        plt.close(fig)

    def plot_dsm_history(self, data=None, save_mp4=True, filename="dsm_history.mp4"):
        """
        Generates an animation of the DSM evolution.
        """
        d = data or self.data
        if d is None:
            print("No data to plot.")
            return

        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(d["dsm_history"][0], cmap='viridis', interpolation='nearest', animated=True)
        plt.colorbar(im, label='Probability')
        title = ax.set_title(f"DSM Evolution (Generation {d['generations'][0]})")

        def update(frame):
            im.set_array(d["dsm_history"][frame])
            title.set_text(f"DSM Evolution (Generation {d['generations'][frame]})")
            return im, title

        ani = animation.FuncAnimation(fig, update, frames=len(d["generations"]), interval=200, blit=False)

        if save_mp4:
            try:
                ani.save(filename, writer='ffmpeg')
                print(f"Animation saved as {filename}")
            except Exception as e:
                print(f"Could not save MP4: {e}. Ensure ffmpeg is installed.")

        if not save_mp4:
            plt.show()
        plt.close(fig)

    def plot_dsm_with_fitness(self, data=None, save_path=None):
        """
        Plots the final DSM and fitness evolution next to each other.
        """
        d = data or self.data
        if d is None:
            print("No data to plot.")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        # DSM Plot
        best_dsm = d["dsm_history"][-1]
        im = ax1.imshow(best_dsm, cmap='viridis', interpolation='nearest')
        plt.colorbar(im, ax=ax1, label='Probability')
        ax1.set_title(f"Final DSM (Gen {d['generations'][-1]})")
        ax1.set_xlabel("Column Index")
        ax1.set_ylabel("Row Index")

        # Fitness Plot
        ax2.plot(d["generations"], d["best_fitness_history"], marker='.', linestyle='-')
        ax2.set_title("Best Fitness Evolution")
        ax2.set_xlabel("Generation")
        ax2.set_ylabel("Best Fitness")
        ax2.grid(True)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
        plt.close(fig)

    def plot_dsm_with_fitness_history(self, data=None, save_mp4=True, filename="dsm_fitness_history.mp4"):
        """
        Generates an animation of both DSM and fitness evolution side by side.
        """
        d = data or self.data
        if d is None:
            print("No data to plot.")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        # Init DSM
        im = ax1.imshow(d["dsm_history"][0], cmap='viridis', interpolation='nearest', animated=True)
        plt.colorbar(im, ax=ax1, label='Probability')
        ax1_title = ax1.set_title(f"DSM (Gen {d['generations'][0]})")

        # Init Fitness
        line, = ax2.plot([], [], marker='o', color='red', markersize=4)
        full_line, = ax2.plot(d["generations"], d["best_fitness_history"], alpha=0.3, color='blue')
        ax2.set_xlim(min(d["generations"]), max(d["generations"]))
        ax2.set_ylim(min(d["best_fitness_history"]) * 0.95, max(d["best_fitness_history"]) * 1.05)
        ax2.set_title("Fitness Progress")
        ax2.set_xlabel("Generation")
        ax2.set_ylabel("Best Fitness")
        ax2.grid(True)

        def update(frame):
            # Update DSM
            im.set_array(d["dsm_history"][frame])
            ax1_title.set_text(f"DSM (Gen {d['generations'][frame]})")
            
            # Update Fitness marker
            line.set_data([d["generations"][frame]], [d["best_fitness_history"][frame]])
            return im, ax1_title, line

        ani = animation.FuncAnimation(fig, update, frames=len(d["generations"]), interval=200, blit=False)

        if save_mp4:
            try:
                ani.save(filename, writer='ffmpeg')
                print(f"Animation saved as {filename}")
            except Exception as e:
                print(f"Could not save MP4: {e}. Ensure ffmpeg is installed.")

        plt.tight_layout()
        if not save_mp4:
            plt.show()
        plt.close(fig)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Visualize EDA logs.")
    parser.add_argument("--file", default="logs/generation_log.h5", help="Path to .h5 log file")
    parser.add_argument("--action", choices=["fitness", "dsm", "history", "both", "both-history"], default="both", help="Visualization to perform")
    args = parser.parse_args()

    viz = Visualize()
    try:
        data = viz.parse(args.file)
        if args.action == "fitness":
            viz.plot_fitness(data)
        elif args.action == "dsm":
            viz.plot_best_dsm(data)
        elif args.action == "history":
            viz.plot_dsm_history(data)
        elif args.action == "both":
            viz.plot_dsm_with_fitness(data)
        elif args.action == "both-history":
            viz.plot_dsm_with_fitness_history(data)
    except FileNotFoundError as e:
        print(e)

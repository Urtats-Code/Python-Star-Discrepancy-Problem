import json
import os
import queue
import threading
from datetime import datetime, timezone

import h5py
import numpy as np


class GenerationLogger:
    def __init__(self, file_path: str, metadata: dict):
        if os.path.exists(file_path):
            base, ext = os.path.splitext(file_path)
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            file_path = f"{base}_{timestamp}{ext}"

        with h5py.File(file_path, "w") as f:
            grp = f.create_group("metadata")
            grp.create_dataset("n",               data=metadata["n"])
            grp.create_dataset("population_size", data=metadata["population_size"])
            grp.create_dataset("selection_size",  data=metadata["selection_size"])
            grp.create_dataset("learning_method", data=metadata["learning_method"])
            grp.create_dataset("generations",     data=metadata["generations"])
            grp.create_dataset("epsilon",         data=metadata["epsilon"])
            grp.create_dataset("num_workers",     data=metadata["num_workers"])
            grp.create_dataset(
                "created_at",
                data=datetime.now(timezone.utc).isoformat(),
            )

        self._file_path = file_path
        self._queue = queue.Queue()
        self._thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._thread.start()

    def log(self, generation_data: dict) -> None:
        self._queue.put(generation_data)

    def close(self) -> None:
        self._queue.put(None)
        self._thread.join()

    @staticmethod
    def jsonnify(h5_path: str, json_path: str = None) -> str:
        if json_path is None:
            json_path = os.path.splitext(h5_path)[0] + ".json"

        def _read_group(grp):
            return {key: _read_item(grp[key]) for key in grp}

        def _read_item(item):
            if isinstance(item, h5py.Group):
                return _read_group(item)
            val = item[()]
            if isinstance(val, bytes):
                return val.decode()
            if isinstance(val, np.ndarray):
                return val.tolist()
            if isinstance(val, (np.integer,)):
                return int(val)
            if isinstance(val, (np.floating,)):
                return float(val)
            return val

        with h5py.File(h5_path, "r") as f:
            data = _read_group(f)

        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)

        return json_path

    def _writer_loop(self) -> None:
        with h5py.File(self._file_path, "a") as f:
            while True:
                item = self._queue.get()
                if item is None:
                    break
                gen = item["generation"]
                grp = f.create_group(f"generation_{gen:06d}")
                grp.create_dataset("population",   data=np.asarray(item["population"],   dtype=np.int64),   compression="gzip", compression_opts=4)
                grp.create_dataset("fitness",      data=np.asarray(item["fitness"],      dtype=np.float64))
                grp.create_dataset("best_fitness", data=float(item["best_fitness"]))
                grp.create_dataset("dsm",          data=np.asarray(item["dsm"],          dtype=np.float64))
                f.flush()

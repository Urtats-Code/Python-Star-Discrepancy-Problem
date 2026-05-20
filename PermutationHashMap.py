class PermutationHashMap:
    """Maps permutations (as sequences of integers) to float values."""

    def __init__(self):
        self._map: dict[tuple, float] = {}

    def _key(self, permutation) -> tuple:
        return tuple(int(x) for x in permutation)

    def set(self, permutation, value: float):
        self._map[self._key(permutation)] = float(value)

    def get(self, permutation) -> float | None:
        return self._map.get(self._key(permutation))

    def contains(self, permutation) -> bool:
        return self._key(permutation) in self._map

    def __len__(self) -> int:
        return len(self._map)

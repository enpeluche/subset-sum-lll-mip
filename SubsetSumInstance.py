import random
from fpylll import IntegerMatrix
import math


class SubsetSumInstance:
    weights: list[int]
    target: int
    solution: list[int] | None
    status: str

    def __init__(
        self,
        weights: list[int],
        target: int,
        solution: list[int] | None,
        status: str = "unknown",
    ) -> None:
        self.weights = weights
        self.target = target
        self.n = len(weights)

        # ground truth, for benchmarking only
        self.solution = solution
        self.status = "feasible" if solution is not None else status


    @property
    def is_trivially_infeasible(self) -> bool:
        """Returns True if the target exceeds the sum of all weights."""
        return sum(self.weights) < self.target

    @classmethod
    def create_uniform_feasible(
        cls, n: int, min_weight: int, max_weight: int
    ) -> "SubsetSumInstance":

        weights = [random.randint(min_weight, max_weight) for _ in range(n)]

        solution = [random.choice([0, 1]) for _ in range(n)]

        target = sum(w * s for w, s in zip(weights, solution))

        return cls(weights, target, solution)

    @classmethod
    def create_super_increasing_feasible(cls, n: int) -> "SubsetSumInstance":
        weights: list[int] = []
        total_sum = 0

        for _ in range(n):
            new_weight = total_sum + random.randint(1, 10)
            weights.append(new_weight)
            total_sum += new_weight

        solution: list[int] = [random.choice([0, 1]) for _ in range(n)]
        target: int = sum(w * s for w, s in zip(weights, solution))

        return cls(weights, target, solution)

    def to_knapsack_matrix(self, M: int | None = None):
        """
        Build the (n+1) x (n+1) knapsack lattice matrix for Subset Sum.

        The matrix encodes the problem so that short vectors in the reduced basis
        correspond to candidate solutions. The scaling factor M amplifies the
        weight column relative to the identity block, steering LLL toward
        binary solutions.

        Structure:
            [ a_1*M   1  0  ...  0 ]
            [ a_2*M   0  1  ...  0 ]
            [  ...                 ]
            [ a_n*M   0  0  ...  1 ]
            [ -T*M    0  0  ...  0 ]

        Args:
            weights: List of n positive integers.
            T:       Target sum.
            M:       Scaling factor (default 1; set to 2^n for standard reduction).

        Returns:
            An fpylll IntegerMatrix of shape (n+1) x (n+1).
        """
        if M is None:
            M = sum(self.weights)

        n = self.n
        weights = self.weights
        T = self.target

        matrix = [[0] * (n + 1) for _ in range(n + 1)]
        for i in range(n):
            matrix[i][0] = weights[i] * M
            matrix[i][i + 1] = 1
        matrix[n][0] = -T * M

        return IntegerMatrix.from_matrix(matrix)

    def is_solution(self, candidate: list[int]) -> bool:
        """
        Check whether a binary vector is a valid solution to the Subset Sum instance.

        Args:
            candidate: Binary vector of length n.

        Returns:
            True if sum(weights[i] * candidate[i]) == T, False otherwise.
        """

        if len(candidate) != self.n or not all(x in (0, 1) for x in candidate):
            return False

        return sum(w * x for w, x in zip(self.weights, candidate)) == self.target

    @property
    def density(self) -> float:
        if not self.weights:
            return 0.0

        max_w = max(self.weights)

        if max_w <= 1:
            return float("inf")

        return self.n / math.log2(max_w)

    @staticmethod
    def get_min_safe_density(
        n: int, bits: int = 64, safety_margin: float = 2.0
    ) -> float:
        """
        Computes the minimum allowed density for dimension n.

        Ensures that the total sum of the weights will not exceed the limit
        of a signed integer for the specified number of bits.

        Args:
            n: The number of elements in the instance.
            bits: The target architecture of the solver (default: 64 for CP-SAT).
            safety_margin: Safety margin on the exponent (default: 2.0).

        Returns:
            The minimum safe density value for dimension n.
        """

        if n <= 0:
            raise ValueError("n must > 0.")
        if bits <= 1:
            raise ValueError("bits must be > 1.")

        max_exponent = bits - 1.0
        denom = max_exponent - math.log2(n) - safety_margin

        if denom <= 0:
            raise ValueError(f"n={n} is too big for {bits} bits.")

        return n / denom

    @classmethod
    def create_crypto_density_feasible(
        cls, n: int, density: float
    ) -> "SubsetSumInstance":
        """
        Generates a random instance for a given cryptographic density.

        Weights are sampled uniformly from [1, 2^(n/density)].
        A secret binary solution is generated to guarantee feasibility.

        Args:
            n: The number of elements.
            density: The target density (d = n / log2(max_weight)).

        Returns:
            A new SubsetSumInstance.

        Raises:
            ValueError: If the density is too low (risk of 64-bit integer overflow).
        """

        # 1. Safety check using the static method we created
        min_density = cls.get_min_safe_density(n)
        if density < min_density:
            raise ValueError(
                f"Density too low: for n={n}, d must be > {min_density:.3f} to avoid integer overflow in CP-SAT (must not exceed 64 bits)."
            )

        # 2. Upper bound calculation and weight sampling
        max_weight = int(2 ** (n / density))
        weights: list[int] = [random.randint(1, max_weight) for _ in range(n)]

        # 3. Generate the secret solution and compute target T
        solution: list[int] = [random.choice([0, 1]) for _ in range(n)]
        target: int = sum((w * s for w, s in zip(weights, solution)), 0)

        return cls(weights, target, solution=solution)

    @classmethod
    def create_crypto_density_no_overflow_feasible(
        cls, n: int, density: float
    ) -> "SubsetSumInstance":
        """
        Generates an instance without the 64-bit constraint.

        Weights can be arbitrarily large. This is intended for solvers that
        support arbitrary precision (like pure Python algorithms or LLL)
        and WILL cause integer overflows in C++ solvers like CP-SAT.

        Args:
            n: The number of elements.
            density: The target density (d = n / log2(max_weight)).

        Returns:
            A new SubsetSumInstance.
        """
        import random

        bit_length = int(n / density)
        max_weight = 1 << bit_length

        weights: list[int] = [random.randint(1, max_weight) for _ in range(n)]

        solution: list[int] = [random.choice([0, 1]) for _ in range(n)]
        target: int = sum((w * s for w, s in zip(weights, solution)), 0)

        return cls(weights, target, solution=solution)

    def get_sorted(self, reverse: bool = True) -> "SubsetSumInstance":
        if self.solution is not None:
            paired = list(zip(self.weights, self.solution))
            paired.sort(key=lambda item: item[0], reverse=reverse)

            new_weights = [item[0] for item in paired]
            new_solution = [item[1] for item in paired]

            return SubsetSumInstance(
                new_weights, self.target, new_solution, status=self.status
            )
        else:
            new_weights = sorted(self.weights, reverse=reverse)
            return SubsetSumInstance(new_weights, self.target, None, status=self.status)

    def residual(self, candidate: list[int]) -> int:
        return sum(w * x for w, x in zip(self.weights, candidate)) - self.target

    def hamming_to_solution(self, candidate: list[int]) -> int | None:
        if self.solution is None:
            return None

        return sum(a != b for a, b in zip(candidate, self.solution))

    def __repr__(self) -> str:
        return (
            f"SubsetSum(n={self.n}, Density={self.density:.3f}, Status={self.status})"
        )

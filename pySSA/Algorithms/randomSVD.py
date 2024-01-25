from Algorithms.SVD import SVD
import numpy as np
from numpy import typing as npt


class randomSVD(SVD):
    def __init__(self, trajectory_matrix: npt.NDArray[np.float64], rank: int) -> None:
        # bool_ is a bool stored as a byte
        self.__U: npt.NDArray[np.bool_]
        self.__S: npt.NDArray[np.float64]
        self.__V: npt.NDArray[np.bool_]

        self.__calculate(trajectory_matrix, rank)

    def __call__(
        self, trajectory_matrix: npt.NDArray[np.float64], rank: int
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        self.__calculate(trajectory_matrix, rank)

        return self.singular_values, self.truncation()

    def __calculate(
        self, trajectory_matrix: npt.NDArray[np.float64], rank: int
    ) -> None:
        """
        Randomised SVD based on the Rangerfinder method. See (Halko et al. 2011) for further details.
        """
        n = trajectory_matrix.shape[1]

        Omega = np.random.randn(n, rank)
        Q, _ = np.linalg.qr(trajectory_matrix @ Omega)
        C = Q.T @ trajectory_matrix
        Ubar, self.__S, self.__V = np.linalg.svd(C, full_matrices=False)
        self.__U = Q @ Ubar

    def truncation(self) -> npt.NDArray[np.float64]:
        return self.__U @ np.diag(self.__S) @ self.__V

    @property
    def singular_values(self) -> npt.NDArray[np.float64]:
        return self.__S

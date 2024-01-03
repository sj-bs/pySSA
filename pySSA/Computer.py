from numbers import Number
from typing import Any, Dict, Tuple

import numpy as np
import numpy.typing as npt
from numpy import dtype, ndarray
from pathos.multiprocessing import ProcessPool as Pool
from pathos.multiprocessing import cpu_count

from pySSA.Logger import CustomLogger as Logger
from pySSA.models.Config import Config
from pySSA.models.TimeSeries import TimeSeries


def randomised_svd(
    trajectory_matrix: npt.NDArray[np.float64], rank: int
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Compute the SVD of a matrix input_traj_mat using the
    randomised SVD algorithm, using the Rangefinder method
    (Halko et al. 2011).
    """
    n = trajectory_matrix.shape[1]
    Omega = np.random.randn(n, rank)
    Q, _ = np.linalg.qr(trajectory_matrix @ Omega)
    C = Q.T @ trajectory_matrix
    Ubar, S, V = np.linalg.svd(C, full_matrices=False)
    U = Q @ Ubar
    return U, S, V


def get_series_from_truncated_svd(
    truncated_svd: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Given a truncated svd, return the reconstructed time series
    """
    flipped = np.flipud(truncated_svd)
    min_index, max_index = -truncated_svd.shape[0] + 1, truncated_svd.shape[1]
    reconstructed_series = np.array(
        [np.mean(np.diag(flipped, i)) for i in range(min_index, max_index)]
    )
    return reconstructed_series


def get_elementary_matrix(
    sing_val: Number,
    U: npt.NDArray[np.float64],
    V: npt.NDArray[np.float64],
    rank_of_sing_vec: int,
) -> npt.NDArray[np.float64]:
    """
    Given a rank r, compute the r^th elementary matrix.
    associated with the SVD sum, i.e. sigma_r u_r v_r^T.
    """
    X_r = sing_val * np.outer(U[:, rank_of_sing_vec], V[rank_of_sing_vec, :])
    return X_r


def _parse_components(
    components: npt.NDArray[np.float64],
    S: npt.NDArray[np.float64],
    U: npt.NDArray[np.float64],
    V: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Given the singular values and the left and right singular vectors,
    compute the reconstructed series corresponding to each rank-1
    elementary matrix. The sum of the first n of these components corresponds
    to a rank-n approximation of the original series.
    """
    for i, singular_value in enumerate(S):
        reconstruction = get_elementary_matrix(singular_value, U, V, i)
        components[i] = get_series_from_truncated_svd(reconstruction)
    return components


class Computer:
    def __init__(
        self,
        time_series: TimeSeries,
        hankel_matrices: Dict[int, npt.NDArray[np.float64]],
        config: Config,
    ):
        """
        Given the hankel matrices, TimeSeries object, and appropriate configuration,
        compute the relevant SSA mode
        """
        self.logger = Logger(__name__).get_logger()
        self.rank = time_series.rank
        self.window_size = time_series.window_size
        self.num_series = time_series.batch_size
        if self.num_series == 1:
            self.series_length = time_series.data.shape[0]
        else:
            self.series_length = time_series.data.shape[1]
        self.Config = config
        self.hankel_matrices = hankel_matrices

    def generate_data(self) -> Dict[int, Dict[str, npt.NDArray[np.float64]]]:
        """
        Generate the data for the given configuration
        """
        if self.Config.parallel:
            num_jobs = cpu_count()  # use all cores
        else:
            num_jobs = 1
        self.logger.info("Generating data. If parallel, this will use all cores")
        with Pool(processes=num_jobs) as pool:
            full_data = dict(
                pool.map(self._single_series_ssa, list(range(self.num_series)))
            )
        self.logger.info("Data generated")
        return full_data

    def _single_series_ssa(
        self, series_index: int
    ) -> tuple[int, dict[str, ndarray[Any, dtype[np.float64]]]]:
        """
        Given an index for a time series, perform ssa on it and return a
        tuple containing that index as well as the reconstructed time series
        """
        self.logger.debug(f"Finished series {series_index}")
        return series_index, self._ssa(
            trajectory_matrix=self.hankel_matrices[series_index],
            rank=self.rank,
            config=self.Config,
        )

    def _ssa(
        self,
        trajectory_matrix: npt.NDArray[np.float64],
        rank: int,
        config: Config,
    ) -> Dict[str, npt.NDArray[np.float64]]:
        """
        Given a single time series, perform ssa on it

        Chain of logic is such:
        1. Check if we want to use a randomised SVD. If so, we can only return a
           reconstructed time series, so we do so
        2. Else, we have 3 subcases:
           2.1. We want to return the singular values. In this case, we return the
                singular values, which we can generate without the full SVD
           2.2. We want to return the reconstructed time series. In this case, we
                return the reconstructed time series, which we need the full SVD for
           2.3. We want to return the full reconstruction. This requires the singular
                values, the components, and the reconstructed time series (the last
                of which we can generate from the components)
        """
        if config.svd_method == "randomized":
            """
            We can only perform a series reconstruction if a randomised svd is chosen
            """
            self.logger.info("Performing randomised svd on hankelised series")
            U, S, V = randomised_svd(trajectory_matrix, rank)
            truncated_svd = U @ np.diag(S) @ V
            self.logger.info("Returning reconstructed time series")
            series = get_series_from_truncated_svd(truncated_svd)
            return {"reconstruction": series}
        else:
            self.logger.info("Performing full svd on hankelised series")
            if config.return_data == "singular_values":
                S = np.linalg.svd(trajectory_matrix, compute_uv=False)
                rank = self._rank_validator(rank, S)
                return {"singular_values": S[:rank]}
            U, S, V = np.linalg.svd(trajectory_matrix)
            rank = self._rank_validator(rank, S)
            if config.return_data == "reconstruction":
                truncated_svd = U[:, :rank] @ np.diag(S[:rank]) @ V[:rank, :]
                self.logger.info("Returning reconstructed time series")
                series = get_series_from_truncated_svd(truncated_svd)
                return {"reconstruction": series}
            else:
                self.logger.info(
                    "Computing full reconstruction: parsing components,"
                    " singular values, and reconstruction"
                )
                time_series_components = np.zeros(
                    (len(S), self.series_length)
                )  # empty array of shape (rank, series_length),
                # with the ith row being the ith rank component
                self.logger.info(f"Shape of components: {time_series_components.shape}")
                time_series_components = _parse_components(
                    time_series_components, S, U, V
                )
                self.logger.info("Components parsed")
                reconstructed_series = np.sum(time_series_components[:rank], axis=0)
                return {
                    "components": time_series_components,
                    "singular_values": S,
                    "reconstruction": reconstructed_series,
                }

    def _rank_validator(self, rank: int, S: npt.NDArray[np.float64]):
        """
        Check that the rank is less than the rank of the trajectory matrix
        """
        rank_of_s = len(S)
        if rank > rank_of_s:
            self.logger.warning(
                f"Rank was higher than the rank of the trajectory matrix. "
                f"Setting rank to {rank_of_s}"
            )
            rank = rank_of_s
        return rank

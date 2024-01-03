import logging
import sys
from unittest import TestCase, main
from unittest.mock import patch

import numpy as np
from pathos.multiprocessing import cpu_count

from pySSA.Computer import (
    randomised_svd,
    get_elementary_matrix,
    get_series_from_truncated_svd,
)

from pySSA.models.Config import Config
from pySSA.models.TimeSeries import TimeSeries
from pySSA.SSA import SSA

np.random.seed(42)

logger = logging.getLogger()
logger.level = logging.INFO
stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stream_handler)


class TestComputation(TestCase):
    def setUp(self):
        known_series = np.array([1, 1, 3, 5, 8, 13, 21])
        full_series = np.random.rand(3, 7)
        full_series[0] = known_series
        self.series = full_series
        self.window_size = 3
        self.rank = 2

    def test_multiple_outputs_for_full_reconstruction(
        self,
    ):
        time_series = TimeSeries(
            data=self.series, window_size=self.window_size, rank=self.rank
        )
        config = Config(svd_method="full", return_data="full", parallel=False)

        ssa = SSA(time_series, config)
        ssa.get_hankel_matrices()
        ssa.compute_ssa()

        assert bool(ssa.singular_values)
        assert bool(ssa.ssa_reconstructions)
        assert bool(ssa.components)

    def test_correct_outputs_for_sing_val(
        self,
    ):
        time_series = TimeSeries(
            data=self.series, window_size=self.window_size, rank=self.rank
        )
        config = Config(
            svd_method="full", return_data="singular_values", parallel=False
        )

        ssa = SSA(time_series, config)
        ssa.get_hankel_matrices()
        ssa.compute_ssa()

        assert bool(ssa.singular_values)
        assert not bool(ssa.ssa_reconstructions)
        assert not bool(ssa.components)

    def test_correct_outputs_for_reconstruction(
        self,
    ):
        time_series = TimeSeries(
            data=self.series, window_size=self.window_size, rank=self.rank
        )
        config = Config(svd_method="full", return_data="reconstruction", parallel=False)

        ssa = SSA(time_series, config)
        ssa.get_hankel_matrices()
        ssa.compute_ssa()

        assert not bool(ssa.singular_values)
        assert bool(ssa.ssa_reconstructions)
        assert not bool(ssa.components)

    def test_ssa_calculation(
        self,
    ):
        time_series = TimeSeries(
            data=self.series, window_size=self.window_size, rank=self.rank
        )
        config = Config(svd_method="full", return_data="reconstruction", parallel=False)

        ssa = SSA(time_series, config)
        ssa.get_hankel_matrices()
        ssa.compute_ssa()
        reconstructed_series = ssa.ssa_reconstructions[0]
        expected = np.array([0.83, 1.32, 2.86, 5.00, 8.02, 12.98, 21.01])
        assert np.allclose(reconstructed_series, expected, atol=0.01)

    def test_hankelisation(
        self,
    ):
        time_series = TimeSeries(
            data=self.series, window_size=self.window_size, rank=self.rank
        )
        config = Config(svd_method="full", return_data="reconstruction", parallel=False)

        ssa = SSA(time_series, config)
        stream_handler.stream = sys.stdout
        hankels = ssa.get_hankel_matrices()
        logger.info("Checking hankelisation")
        for matrix in hankels.values():
            assert np.all(matrix[1:, :-1] == matrix[:-1, 1:])

    def test_rank_reset(self):
        stream_handler.stream = sys.stdout
        window_size = 6
        rank = 5  # actually of rank 2
        time_series = TimeSeries(data=self.series, window_size=window_size, rank=rank)
        config = Config(svd_method="full", return_data="full", parallel=False)
        # check for specific text in logs:
        ssa = SSA(time_series, config)
        ssa.get_hankel_matrices()
        ssa.compute_ssa()
        for series in ssa.singular_values.values():
            assert len(series) < rank

    def test_randomised_svd(self):
        input_matrix = np.random.rand(200, 30)
        rank = 10
        Ru, Rs, Rv = randomised_svd(input_matrix, rank)
        random_trunc_svd = np.dot(Ru, np.dot(np.diag(Rs), Rv))
        u, s, v = np.linalg.svd(input_matrix)
        u = u[:, :rank]
        s = s[:rank]
        v = v[:rank, :]
        trun_svd = np.dot(u, np.dot(np.diag(s), v))
        scale_factor = np.linalg.norm(
            input_matrix - trun_svd, ord="fro"
        ) / np.linalg.norm(input_matrix - random_trunc_svd, ord="fro")
        assert (
            scale_factor <= 1
        )  # Testing Theorem 15.1 from C6.1 NLA Lecture Notes, Nakatsukasa (2023)

    def test_elementary_matrics(self):
        time_series = TimeSeries(
            data=self.series, window_size=self.window_size, rank=self.rank
        )
        config = Config(svd_method="full", return_data="reconstruction", parallel=False)
        ssa = SSA(time_series, config)
        hankels = ssa.get_hankel_matrices()
        for matrix in hankels.values():
            u, s, v = np.linalg.svd(matrix)
            rank_of_matrix = np.linalg.matrix_rank(matrix)
            for i, sing_val in enumerate(s[:rank_of_matrix]):
                elementary_matrix = get_elementary_matrix(sing_val, u, v, i)
                _, s2, _ = np.linalg.svd(elementary_matrix)
                rank_of_elementary_matrix = np.sum(s2 > 1e-10)
                assert rank_of_elementary_matrix == 1

    def test_series_length_correctly_generated_multi_series(self):
        time_series = TimeSeries(
            data=self.series, window_size=self.window_size, rank=self.rank
        )
        config = Config(svd_method="full", return_data="reconstruction", parallel=False)

        ssa = SSA(time_series, config)
        ssa.get_hankel_matrices()
        ssa.compute_ssa()
        assert len(ssa.ssa_reconstructions.values()) == self.series.shape[0]

    def test_series_length_correctly_generated_single_series(self):
        time_series = TimeSeries(
            data=self.series[0], window_size=self.window_size, rank=self.rank
        )
        config = Config(svd_method="full", return_data="reconstruction", parallel=False)

        ssa = SSA(time_series, config)
        ssa.get_hankel_matrices()
        ssa.compute_ssa()
        assert len(ssa.ssa_reconstructions.values()) == 1

    @patch("pySSA.Computer.Pool")
    def test_parallel_computation(self, mock_pool):
        time_series = TimeSeries(
            data=self.series, window_size=self.window_size, rank=self.rank
        )
        config = Config(svd_method="full", return_data="full", parallel=True)

        ssa = SSA(time_series, config)
        ssa.get_hankel_matrices()
        ssa.compute_ssa()

        expected_processes = cpu_count()
        mock_pool.assert_called_once_with(processes=expected_processes)

    @patch("pySSA.Computer.Pool")
    def test_single_computation(self, mock_pool):
        time_series = TimeSeries(
            data=self.series, window_size=self.window_size, rank=self.rank
        )
        config = Config(svd_method="full", return_data="full", parallel=False)

        ssa = SSA(time_series, config)
        ssa.get_hankel_matrices()
        ssa.compute_ssa()

        mock_pool.assert_called_once_with(processes=1)

    @patch("pySSA.SSA.Pool")
    def test_parallel_generation(self, mock_pool):
        time_series = TimeSeries(
            data=self.series, window_size=self.window_size, rank=self.rank
        )
        config = Config(svd_method="full", return_data="full", parallel=True)

        ssa = SSA(time_series, config)
        ssa.get_hankel_matrices()

        expected_processes = cpu_count()
        mock_pool.assert_called_once_with(processes=expected_processes)

    @patch("pySSA.SSA.Pool")
    def test_single_generation(self, mock_pool):
        time_series = TimeSeries(
            data=self.series, window_size=self.window_size, rank=self.rank
        )
        config = Config(svd_method="full", return_data="full", parallel=False)

        ssa = SSA(time_series, config)
        ssa.get_hankel_matrices()

        mock_pool.assert_called_once_with(processes=1)

    def test_getting_series_from_truncated(self):
        stream_handler.stream = sys.stdout
        time_series = TimeSeries(
            data=self.series[0], window_size=self.window_size, rank=self.rank
        )
        config = Config(svd_method="full", return_data="full", parallel=False)

        ssa = SSA(time_series, config)
        hankel = ssa.get_hankel_matrices()

        for key, matrix in hankel.items():
            reconstructed_series = get_series_from_truncated_svd(matrix)
            assert np.all(reconstructed_series == self.series[key])


if __name__ == "__main__":
    main()

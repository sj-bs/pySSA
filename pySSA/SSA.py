from typing import Dict, Tuple

import numpy as np
import numpy.typing as npt
from pathos.multiprocessing import ProcessPool as Pool  # type: ignore
from pathos.multiprocessing import cpu_count  # type: ignore

from pySSA.Computer import Computer
from pySSA.Logger import CustomLogger as Logger
from pySSA.models.Config import Config
from pySSA.models.TimeSeries import TimeSeries


class SSA:
    def __init__(self, time_series: TimeSeries, config: Config):
        """
        Initializes the SSA object.

        Parameters:
        time_series (TimeSeries): The time series data, shape N x M where N is the number of time series and M is the length of each.
        config (Config): The configuration settings.

        """
        self.TimeSeries = time_series
        self.Config = config
        self.logger = Logger(__name__).get_logger()

        self.hankel_matrices: Dict[int, npt.NDArray[np.float64]] = {}
        self.components: Dict[int, npt.NDArray[np.float64]] = {}
        self.singular_values: Dict[int, npt.NDArray[np.float64]] = {}
        self.ssa_reconstructions: Dict[int, npt.NDArray[np.float64]] = {}

        if self.TimeSeries.batch_size > 1:
            self.logger.info("Multiple time series detected")
        if self.Config.parallel:
            self.logger.info("Parallel computation enabled")
        else:
            self.logger.info("Parallel computation disabled")

    def _parallel_hankelise(
        self, series_index: int
    ) -> Tuple[int, npt.NDArray[np.float64]]:
        """
        Given an index for a time series, hankelise it and return a
        tuple containing that index as well as the hankelised time series
        """
        if self.TimeSeries.batch_size == 1:
            series = self.TimeSeries.data
        else:
            series = self.TimeSeries.data[series_index]
        hankelised_series = self._hankelise_single_series(
            series, self.TimeSeries.window_size
        )
        return series_index, hankelised_series

    def _hankelise_single_series(
        self, series: npt.NDArray[np.float64], window_size: int
    ) -> npt.NDArray[np.float64]:
        """
        Given a single time series, hankelise it
        """
        n = series.size - window_size + 1
        return np.array([series[i : i + window_size] for i in range(n)])

    def get_hankel_matrices(self) -> Dict[int, npt.NDArray[np.float64]]:
        """
        For however many time series there are, return a list of hankel
        matrices for each

        Returns a dictionary with the index of the time series as the key and
        the hankelised time series as the value
        """
        if self.Config.parallel:
            num_jobs = cpu_count()  # use all cores
        else:
            num_jobs = 1
        with Pool(processes=num_jobs) as pool:
            self.hankel_matrices = dict(
                pool.map(
                    self._parallel_hankelise,
                    list(range(self.TimeSeries.batch_size)),
                )
            )
        self.logger.debug("Hankel matrices computated")
        return self.hankel_matrices

    def compute_ssa(self) -> None:
        computer = Computer(
            time_series=self.TimeSeries,
            hankel_matrices=self.hankel_matrices,
            config=self.Config,
        )
        full_data = computer.generate_data()

        self.logger.debug("Parsing output...")
        if self.Config.return_data == "full":
            self.ssa_reconstructions = {
                index: full_data[index]["reconstruction"] for index in full_data.keys()
            }
            self.singular_values = {
                index: full_data[index]["singular_values"] for index in full_data.keys()
            }
            self.components = {
                index: full_data[index]["components"] for index in full_data.keys()
            }
        elif self.Config.return_data == "reconstruction":
            self.ssa_reconstructions = {
                index: full_data[index]["reconstruction"] for index in full_data.keys()
            }
        elif self.Config.return_data == "singular_values":
            self.singular_values = {
                index: full_data[index]["singular_values"] for index in full_data.keys()
            }


if __name__ == "__main__":
    pass

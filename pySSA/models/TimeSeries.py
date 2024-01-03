from typing import Any, Generic, List, TypeVar

import numpy as np
import numpy.typing as npt
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    ValidationInfo,
    field_validator,
)

from pySSA.Logger import CustomLogger as Logger

logger = Logger(__name__).get_logger()


class TimeSeries(BaseModel):
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    data: npt.NDArray[np.float64] = Field(...)
    window_size: int = Field(..., gt=0)
    rank: int = Field(..., gt=0)

    def __init__(self, **kwargs):  # type: ignore
        try:
            super().__init__(**kwargs)
            logger.info(
                f"TimeSeries created with with batch_size {self.batch_size}, "
                f"window size: {self.window_size}, and rank: {self.rank}"
            )
        except ValidationError as error:
            logger.error(f"Failed to create TimeSeries: {error}")
            raise

    @field_validator("data")
    def check_data_valid(cls, data: npt.NDArray[np.float64], values: ValidationInfo):
        if (not np.isfinite(data).any()) or (data.size == 0):
            raise ValueError("Data contains invalid values.")
        else:
            return data

    @field_validator("rank")
    def check_rank_less_than_window_size(cls, rank: int, values: ValidationInfo):
        try:
            if rank >= values.data["window_size"]:
                raise ValueError(
                    f"Rank must be less than window size. Rank: {rank}, "
                    f"window size: {values.data['window_size']}"
                )
            return rank
        except KeyError:
            raise ValueError("Window size not set.")

    @property
    def batch_size(self) -> int:
        """
        Returns the number of time series
        If Data passed in is N x M, then batch_size is N
        """
        shape = self.data.shape
        if len(shape) == 1:
            return 1
        elif len(shape) == 2:
            return shape[0]
        else:
            raise ValueError(f"Data must be 1 or 2 dimensional. Data shape: {shape}")


if __name__ == "__main__":
    pass

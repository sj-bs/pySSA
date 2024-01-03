from enum import Enum

from pydantic import BaseModel, ValidationError

from pySSA.Logger import CustomLogger as Logger

logger = Logger(__name__).get_logger()


class ReturnData(str, Enum):
    singular_values = "singular_values"
    reconstruction = "reconstruction"
    full = "full"


class SvdMethod(str, Enum):
    randomized = "randomized"
    full = "full"


class Config(BaseModel):
    return_data: ReturnData = ReturnData.full
    svd_method: SvdMethod = SvdMethod.randomized
    parallel: bool = False

    def __init__(self, **kwargs):  # type: ignore
        try:
            super().__init__(**kwargs)
            logger.info(
                f"Config created with return_data: {self.return_data}, "
                f"svd_method: {self.svd_method}, and parallel: {self.parallel}"
            )
        except ValidationError as error:
            logger.error(f"Failed to create Config: {error}")
            raise


if __name__ == "__main__":
    pass

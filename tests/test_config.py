from unittest import TestCase, main

from pydantic import ValidationError

from pySSA.models.Config import Config, ReturnData, SvdMethod


class TestConfig(TestCase):
    def test_create_config(self):
        config = Config(return_data=ReturnData.full, svd_method=SvdMethod.randomized)
        self.assertEqual(config.return_data, ReturnData.full)
        self.assertEqual(config.svd_method, SvdMethod.randomized)

    def test_invalid_return_data(self):
        with self.assertRaises(ValidationError):
            Config(return_data="invalid", svd_method=SvdMethod.randomized)

    def test_invalid_svd_method(self):
        with self.assertRaises(ValidationError):
            Config(return_data=ReturnData.full, svd_method="invalid")

    def test_invalid_parallel_method(self):
        with self.assertRaises(ValidationError):
            Config(return_data=ReturnData.full, svd_method="randomized", parallel="foo")


if __name__ == "__main__":
    main()

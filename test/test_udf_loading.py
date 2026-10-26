import builtins
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from main_openeo import get_udf


class UdfLoadingTests(unittest.TestCase):
    def test_udf_loads_from_another_directory_without_local_helper_imports(self):
        original_import = builtins.__import__

        def isolated_import(name, *args, **kwargs):
            if name in {"openeo_mountains_snow", "fit_distribution"}:
                raise ModuleNotFoundError(name)
            return original_import(name, *args, **kwargs)

        previous_directory = os.getcwd()
        try:
            with tempfile.TemporaryDirectory() as directory:
                os.chdir(directory)
                with patch("builtins.__import__", side_effect=isolated_import):
                    udf = get_udf("representative_pixels.py")
                    namespace = {}
                    exec(compile(udf.code, "representative_pixels.py", "exec"), namespace)
        finally:
            os.chdir(previous_directory)

        self.assertTrue(callable(namespace["apply_datacube"]))
        self.assertEqual(udf.context, {"classify": True})
        values = np.zeros((5, 5))
        result = namespace["fit_distribution_and_median"](
            values, np.ones_like(values, dtype=bool), default_median=0.6
        )
        self.assertEqual(result["fitted_median"], 0.6)


if __name__ == "__main__":
    unittest.main()

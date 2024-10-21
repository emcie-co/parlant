from pathlib import Path
import tempfile
from typing import Iterator
from pytest import fixture

from tests.test_utilities import ContextOfTest


@fixture
def context() -> Iterator[ContextOfTest]:
    with tempfile.TemporaryDirectory(prefix="emcie-sdk-test_") as home_dir:
        home_dir_path = Path(home_dir)

        yield ContextOfTest(
            home_dir=home_dir_path,
        )

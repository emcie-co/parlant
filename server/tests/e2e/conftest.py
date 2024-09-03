from pathlib import Path
import shutil
import tempfile
from typing import Iterator
from pytest import fixture

from emcie.server.contextual_correlator import ContextualCorrelator
from emcie.server.logger import StdoutLogger
from tests.e2e.test_utilities import _TestContext, get_package_path


@fixture
def context() -> Iterator[_TestContext]:
    with tempfile.TemporaryDirectory(prefix="emcie-server_cli_test_") as home_dir:
        home_dir_path = Path(home_dir)
        active_config_file_path = home_dir_path / "config.json"

        config_template_file = get_package_path() / "_config.json"
        shutil.copy(config_template_file, active_config_file_path)

        yield _TestContext(
            logger=StdoutLogger(ContextualCorrelator()),
            home_dir=home_dir_path,
            config_file=active_config_file_path,
            index_file=home_dir_path / "index.json",
        )

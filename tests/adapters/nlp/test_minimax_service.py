# Copyright 2026 Emcie Co Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from unittest.mock import patch

from parlant.adapters.nlp.minimax_service import MiniMaxService


def test_that_missing_api_key_returns_error_message() -> None:
    """Test that missing MINIMAX_API_KEY returns error message."""
    with patch.dict(os.environ, {}, clear=True):
        error = MiniMaxService.verify_environment()
        assert error is not None
        assert "MINIMAX_API_KEY is not set" in error


def test_that_verify_environment_returns_none_when_api_key_is_set() -> None:
    """Test that verify_environment returns None when MINIMAX_API_KEY is set."""
    with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"}, clear=True):
        error = MiniMaxService.verify_environment()
        assert error is None

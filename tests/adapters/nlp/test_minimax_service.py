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


def test_that_default_model_is_m3() -> None:
    """Test that the default model is MiniMax-M3."""
    from unittest.mock import MagicMock

    with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"}, clear=True):
        logger = MagicMock()
        tracer = MagicMock()
        meter = MagicMock()
        service = MiniMaxService(logger=logger, tracer=tracer, meter=meter)
        assert service._model_name == "MiniMax-M3"


def test_that_m3_and_m2_7_model_classes_exist() -> None:
    """Test that MiniMax M3 and M2.7 model classes are importable and have correct model names."""
    from parlant.adapters.nlp.minimax_service import (
        MiniMax_M3,
        MiniMax_M2_7,
        MiniMax_M2_7_Highspeed,
    )
    from unittest.mock import MagicMock

    logger = MagicMock()
    tracer = MagicMock()
    meter = MagicMock()

    with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"}):
        gen_m3 = MiniMax_M3[dict](logger=logger, tracer=tracer, meter=meter)
        assert gen_m3.model_name == "MiniMax-M3"
        assert gen_m3.max_tokens == 512_000

        gen = MiniMax_M2_7[dict](logger=logger, tracer=tracer, meter=meter)
        assert gen.model_name == "MiniMax-M2.7"

        gen_hs = MiniMax_M2_7_Highspeed[dict](logger=logger, tracer=tracer, meter=meter)
        assert gen_hs.model_name == "MiniMax-M2.7-highspeed"

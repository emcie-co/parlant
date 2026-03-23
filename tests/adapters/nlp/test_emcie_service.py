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

import json
import os
from unittest.mock import AsyncMock, Mock, patch

import pytest
from lagom import Container

from parlant.adapters.nlp.emcie_service import EmcieAPIError, Jackal
from parlant.core.common import DefaultBaseModel
from parlant.core.loggers import Logger
from parlant.core.meter import Meter
from parlant.core.tracer import Tracer


class TestSchema(DefaultBaseModel):
    """Test schema for type checking."""


@pytest.mark.asyncio
@patch("parlant.adapters.nlp.emcie_service.AsyncClient")
async def test_that_emcie_generator_handles_500_without_json_body(
    mock_client_class: Mock, container: Container
) -> None:
    mock_client = AsyncMock()

    response = Mock(status_code=500, text="Server error")
    response.json.side_effect = json.JSONDecodeError("Expecting value", "", 0)

    mock_client.post = AsyncMock(return_value=response)
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = False

    mock_client_class.return_value = mock_client

    with patch.dict(os.environ, {"EMCIE_API_KEY": "test-key"}, clear=True):
        generator = Jackal[TestSchema](
            logger=container[Logger],
            tracer=container[Tracer],
            meter=container[Meter],
            model_role="auto",
        )

        with pytest.raises(EmcieAPIError, match="Emcie API error: 500"):
            await generator._do_generate("Test prompt")

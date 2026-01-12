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
import pytest
from unittest.mock import patch

from parlant.adapters.nlp.qwen_service import (
    QwenService,
    get_qwen_base_url,
    QWEN_REGION_BASE_URLS,
)


def test_that_missing_api_key_returns_error_message() -> None:
    """Test that missing DASHSCOPE_API_KEY returns error message."""
    with patch.dict(os.environ, {}, clear=True):
        error = QwenService.verify_environment()
        assert error is not None
        assert "DASHSCOPE_API_KEY is not set" in error


def test_that_verify_environment_returns_error_for_invalid_region() -> None:
    """Test that verify_environment returns error for invalid QWEN_REGION."""
    with patch.dict(
        os.environ,
        {"DASHSCOPE_API_KEY": "test-key", "QWEN_REGION": "invalid-region"},
        clear=True,
    ):
        error = QwenService.verify_environment()
        assert error is not None
        assert "Invalid QWEN_REGION 'invalid-region'" in error
        assert "Must be one of: international, domestic" in error


def test_that_get_qwen_base_url_returns_international_by_default() -> None:
    """Test that get_qwen_base_url returns international URL by default."""
    with patch.dict(os.environ, {}, clear=True):
        url = get_qwen_base_url()
        assert url == QWEN_REGION_BASE_URLS["international"]
        assert url == "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"


def test_that_get_qwen_base_url_returns_domestic_url_when_region_is_domestic() -> None:
    """Test that get_qwen_base_url returns domestic URL when QWEN_REGION is domestic."""
    with patch.dict(os.environ, {"QWEN_REGION": "domestic"}, clear=True):
        url = get_qwen_base_url()
        assert url == QWEN_REGION_BASE_URLS["domestic"]
        assert url == "https://dashscope.aliyuncs.com/compatible-mode/v1"


def test_that_get_qwen_base_url_returns_international_url_when_region_is_international() -> None:
    """Test that get_qwen_base_url returns international URL when QWEN_REGION is international."""
    with patch.dict(os.environ, {"QWEN_REGION": "international"}, clear=True):
        url = get_qwen_base_url()
        assert url == QWEN_REGION_BASE_URLS["international"]


def test_that_get_qwen_base_url_is_case_insensitive() -> None:
    """Test that QWEN_REGION is case insensitive."""
    with patch.dict(os.environ, {"QWEN_REGION": "DOMESTIC"}, clear=True):
        url = get_qwen_base_url()
        assert url == QWEN_REGION_BASE_URLS["domestic"]

    with patch.dict(os.environ, {"QWEN_REGION": "Domestic"}, clear=True):
        url = get_qwen_base_url()
        assert url == QWEN_REGION_BASE_URLS["domestic"]

    with patch.dict(os.environ, {"QWEN_REGION": "INTERNATIONAL"}, clear=True):
        url = get_qwen_base_url()
        assert url == QWEN_REGION_BASE_URLS["international"]


def test_that_get_qwen_base_url_raises_error_for_invalid_region() -> None:
    """Test that get_qwen_base_url raises ValueError for invalid region."""
    with patch.dict(os.environ, {"QWEN_REGION": "invalid_region"}, clear=True):
        with pytest.raises(ValueError) as exc_info:
            get_qwen_base_url()
        assert "Invalid QWEN_REGION" in str(exc_info.value)
        assert "international" in str(exc_info.value)
        assert "domestic" in str(exc_info.value)


def test_that_qwen_base_url_env_var_takes_priority() -> None:
    """Test that QWEN_BASE_URL environment variable takes priority over QWEN_REGION."""
    custom_url = "https://custom.api.url/v1"
    with patch.dict(
        os.environ,
        {"QWEN_BASE_URL": custom_url, "QWEN_REGION": "domestic"},
        clear=True,
    ):
        url = get_qwen_base_url()
        assert url == custom_url


def test_that_qwen_base_url_env_var_works_alone() -> None:
    """Test that QWEN_BASE_URL works without QWEN_REGION set."""
    custom_url = "https://custom.api.url/v1"
    with patch.dict(os.environ, {"QWEN_BASE_URL": custom_url}, clear=True):
        url = get_qwen_base_url()
        assert url == custom_url

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

import pytest

from parlant.core.session_clients import web_origin


@pytest.mark.parametrize(
    "url, origin",
    [
        ("https://shop.example", "https://shop.example"),
        ("HTTPS://Shop.Example:8443/checkout?x=1", "https://shop.example:8443"),
        ("http://localhost:3000", "http://localhost:3000"),
        ("http://192.0.2.10", "http://192.0.2.10"),
        ("http://[2001:db8::1]:8080", "http://[2001:db8::1]:8080"),
        ("https://b\u00fccher.example", "https://xn--bcher-kva.example"),
    ],
)
def test_that_a_web_origin_keeps_only_scheme_host_and_port(url: str, origin: str) -> None:
    assert web_origin(url) == origin


@pytest.mark.parametrize(
    "url",
    [
        "https://<script>x</script>",
        "https://shop example.com",
        "https://" + "a" * 5000 + ".example",
        "https://" + ".".join(["a" * 63] * 5),
        "https://-leading-hyphen.example",
        "https://under_score.example",
        "null",
        "javascript:alert(1)",
        "chrome-extension://abcdef",
    ],
)
def test_that_a_malformed_or_oversized_host_is_not_a_web_origin(url: str) -> None:
    assert web_origin(url) is None

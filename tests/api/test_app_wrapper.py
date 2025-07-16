# Copyright 2025 Emcie Co Ltd.
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

# from exceptiongroup import ExceptionGroup
from fastapi import FastAPI

from parlant.api.app import APIConfiguration, AppWrapper
from lagom import Container


async def test_that_custom_deployment_app_wrapper_has_only_the_expected_routes(
    container: Container,
) -> None:
    original_app_wrapper = AppWrapper(FastAPI(), container)
    async with original_app_wrapper:
        orig_routes_number = len(original_app_wrapper.app.routes)

    deploy_container = Container(container)
    deploy_container[APIConfiguration] = {"sessions": ["read_session"]}

    deploy_app_wrapper = AppWrapper(FastAPI(), deploy_container)
    async with deploy_app_wrapper:
        deploy_routes_number = len(deploy_app_wrapper.app.routes)

    assert deploy_routes_number < orig_routes_number

    orig_routes_names = [r.name for r in original_app_wrapper.app.routes]
    deploy_routes_names = [r.name for r in deploy_app_wrapper.app.routes]
    assert "read_session" in orig_routes_names
    assert "read_session" in deploy_routes_names
    assert "create_session" in orig_routes_names
    assert "create_session" not in deploy_routes_names

    orig_session_routes = [r for r in original_app_wrapper.app.routes if "session" in r.name]
    deploy_session_routes = [r for r in deploy_app_wrapper.app.routes if "session" in r.name]

    assert len(orig_session_routes) == 6
    assert len(deploy_session_routes) == 1

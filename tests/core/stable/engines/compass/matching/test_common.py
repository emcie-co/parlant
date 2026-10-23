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

from parlant.core.engines.compass.matching.common import aggregate_generation_info
from parlant.core.nlp.common import UsageInfo
from parlant.core.nlp.generation_info import GenerationInfo


def test_that_generation_info_aggregation_preserves_fixed_point_reported_costs() -> None:
    result = aggregate_generation_info(
        [
            GenerationInfo(
                schema_name="TestSchema",
                model="openrouter:model",
                duration=0.1,
                usage=UsageInfo(
                    input_tokens=10,
                    output_tokens=1,
                    extra={"reported_cost": "0.0000001", "cache_write_input_tokens": 2},
                ),
            ),
            GenerationInfo(
                schema_name="TestSchema",
                model="openrouter:model",
                duration=0.2,
                usage=UsageInfo(
                    input_tokens=20,
                    output_tokens=2,
                    extra={"reported_cost": "0.0000002", "cache_write_input_tokens": 3},
                ),
            ),
        ]
    )

    assert result.usage.extra == {
        "reported_cost": "0.0000003",
        "cache_write_input_tokens": 5,
    }

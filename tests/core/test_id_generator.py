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

from parlant.core.common import IdGenerator


async def test_that_id_generator_generates_different_ids_for_objects_with_similar_small_content() -> (
    None
):
    generator = IdGenerator()

    small_content_1 = "test"
    small_content_2 = "test"

    id1 = generator.generate(small_content_1)
    id2 = generator.generate(small_content_2)

    assert id1 != id2
    assert len(id1) == 10
    assert len(id2) == 10


async def test_that_id_generator_generates_different_ids_for_objects_with_similar_big_content() -> (
    None
):
    generator = IdGenerator()

    big_content_1 = "a" * 1000
    big_content_2 = "a" * 1000

    id1 = generator.generate(big_content_1)
    id2 = generator.generate(big_content_2)

    assert id1 != id2
    assert len(id1) == 10
    assert len(id2) == 10

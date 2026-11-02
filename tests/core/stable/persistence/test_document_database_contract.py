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

from inspect import signature
from typing import get_type_hints

from parlant.core.persistence.document_database import DocumentCollection, FindResult


def test_that_document_find_keeps_the_existing_required_total_contract() -> None:
    assert list(signature(DocumentCollection.find).parameters) == [
        "self",
        "filters",
        "limit",
        "cursor",
        "sort_direction",
    ]
    assert get_type_hints(FindResult)["total_count"] is int

# Copyright 2024 Emcie Co Ltd.
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

from typing import Mapping, TypedDict

from parlant.core.common import CustomerId
from parlant.core.persistence.common import ObjectId
from parlant.core.tags import TagId


class _CustomerDocument_v1(TypedDict, total=False):
    id: ObjectId
    creation_utc: str
    name: str
    extra: Mapping[str, str]


class _CustomerTagAssociationDocument_v1(TypedDict, total=False):
    id: ObjectId
    creation_utc: str
    customer_id: CustomerId
    tag_id: TagId

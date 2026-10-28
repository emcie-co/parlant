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

from parlant.core import metadata as metadata_utils


def test_public_metadata_rejects_the_reserved_key() -> None:
    with pytest.raises(metadata_utils.ReservedMetadataKeyError):
        metadata_utils.ensure_public_metadata_key(metadata_utils.PARLANT_METADATA_KEY)

    with pytest.raises(metadata_utils.ReservedMetadataKeyError):
        metadata_utils.ensure_public_metadata({metadata_utils.PARLANT_METADATA_KEY: {}})


def test_public_metadata_returns_only_a_defensive_copy_of_application_values() -> None:
    source = metadata_utils.add_internal_metadata({"application": {"value": 1}}, {"usage": {}})

    public = metadata_utils.public_metadata(source)
    public["application"] = {"value": 2}

    assert source["application"] == {"value": 1}
    assert metadata_utils.public_metadata(source) == {"application": {"value": 1}}


def test_internal_path_helpers_preserve_user_metadata() -> None:
    source = {"application": {"priority": "high"}}

    updated = metadata_utils.set_internal_value(source, ("compass", "discovered_rule_ids"), ["r1"])

    assert source == {"application": {"priority": "high"}}
    assert updated["application"] == {"priority": "high"}
    assert metadata_utils.internal_value(updated, ("compass", "discovered_rule_ids")) == ["r1"]
    assert metadata_utils.internal_metadata(updated)["schema_version"] == 1


def test_internal_builder_initializes_the_schema_for_an_empty_namespace() -> None:
    metadata = metadata_utils.add_internal_metadata({"application": True}, {})

    assert metadata["application"] is True
    assert metadata[metadata_utils.PARLANT_METADATA_KEY] == {"schema_version": 1}


def test_internal_path_helpers_defensively_copy_nested_values() -> None:
    value = {"ids": ["r1"]}
    updated = metadata_utils.set_internal_value({}, ("compass", "state"), value)

    value["ids"].append("r2")
    read = metadata_utils.internal_value(updated, ("compass", "state"))
    assert read == {"ids": ["r1"]}

    assert isinstance(read, dict)
    read["ids"].append("r3")
    assert metadata_utils.internal_value(updated, ("compass", "state")) == {"ids": ["r1"]}


def test_unset_internal_value_removes_only_the_selected_path() -> None:
    source = metadata_utils.set_internal_value({}, ("compass", "rules"), ["r1"])
    source = metadata_utils.set_internal_value(source, ("compass", "terms"), {"t1": 3})

    updated = metadata_utils.unset_internal_value(source, ("compass", "rules"))

    assert metadata_utils.internal_value(updated, ("compass", "rules"), "missing") == "missing"
    assert metadata_utils.internal_value(updated, ("compass", "terms")) == {"t1": 3}


def test_malformed_internal_schema_is_rejected() -> None:
    with pytest.raises(ValueError, match="schema version"):
        metadata_utils.internal_metadata(
            {metadata_utils.PARLANT_METADATA_KEY: {"schema_version": 999}}
        )

    with pytest.raises(ValueError, match="must be an object"):
        metadata_utils.internal_metadata({metadata_utils.PARLANT_METADATA_KEY: "invalid"})

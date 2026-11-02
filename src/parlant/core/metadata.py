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

"""Ownership and path helpers for Parlant-managed domain metadata.

Application metadata remains at the top level of each metadata mapping.
Parlant-owned values live below the single reserved ``__parlant__`` key.  This
module deliberately contains no engine or persistence logic: it defines the
shared ownership boundary used by stores, APIs, the SDK, Compass, voice, and
event consumers.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy

from parlant.core.common import JSONSerializable


PARLANT_METADATA_KEY = "__parlant__"
"""The reserved top-level key containing all Parlant-owned metadata."""

PARLANT_METADATA_SCHEMA_VERSION = 1
"""The fresh-start schema version stored in every internal metadata root."""

MetadataPath = Sequence[str]
"""A path relative to the ``__parlant__`` root."""


class ReservedMetadataKeyError(ValueError):
    """Raised when a public caller attempts to write Parlant-owned metadata."""


def ensure_public_metadata_key(key: str) -> None:
    """Reject a public write targeting the reserved Parlant namespace.

    Args:
        key: Top-level metadata key requested by an API or SDK caller.

    Raises:
        ReservedMetadataKeyError: If ``key`` is ``__parlant__``.
    """

    if key == PARLANT_METADATA_KEY:
        raise ReservedMetadataKeyError(
            f"Metadata key '{PARLANT_METADATA_KEY}' is reserved for Parlant"
        )


def ensure_public_metadata(metadata: Mapping[str, JSONSerializable]) -> None:
    """Validate that a public metadata mapping does not contain the reserved root.

    Args:
        metadata: User-authored top-level metadata.

    Raises:
        ReservedMetadataKeyError: If the reserved root is present.
    """

    if PARLANT_METADATA_KEY in metadata:
        ensure_public_metadata_key(PARLANT_METADATA_KEY)


def public_metadata(
    metadata: Mapping[str, JSONSerializable] | None,
) -> dict[str, JSONSerializable]:
    """Return a defensive copy containing only application-owned metadata.

    Args:
        metadata: Complete persisted metadata, including Parlant's reserved root.

    Returns:
        A copy without the ``__parlant__`` namespace. Internal state is exposed
        through purpose-built API fields rather than the public metadata bag.
    """

    return deepcopy(
        {key: value for key, value in (metadata or {}).items() if key != PARLANT_METADATA_KEY}
    )


def add_internal_metadata(
    metadata: Mapping[str, JSONSerializable],
    values: Mapping[str, JSONSerializable],
) -> dict[str, JSONSerializable]:
    """Add trusted top-level Parlant values to validated public metadata.

    Args:
        metadata: User-authored metadata that must not contain the reserved root.
        values: Trusted values placed immediately below ``__parlant__``.

    Returns:
        A new namespaced metadata mapping.

    Raises:
        ReservedMetadataKeyError: If ``metadata`` already contains the root.
        ValueError: If ``values`` attempts to supply ``schema_version``.
    """

    ensure_public_metadata(metadata)
    if "schema_version" in values:
        raise ValueError("Internal metadata values cannot override schema_version")

    result = deepcopy(dict(metadata))
    result[PARLANT_METADATA_KEY] = {"schema_version": PARLANT_METADATA_SCHEMA_VERSION}
    for key, value in values.items():
        result = set_internal_value(result, (key,), value)
    return result


def initialize_internal_metadata(
    metadata: Mapping[str, JSONSerializable],
) -> dict[str, JSONSerializable]:
    """Return trusted metadata with a validated internal schema root.

    Args:
        metadata: Complete metadata supplied by an internal producer.

    Returns:
        A defensive copy. A missing ``__parlant__`` root is initialized with
        the current schema version; an existing root is preserved.

    Raises:
        ValueError: If an existing internal root is malformed.
    """

    if PARLANT_METADATA_KEY not in metadata:
        return add_internal_metadata(metadata, {})

    internal_metadata(metadata)
    return deepcopy(dict(metadata))


def internal_metadata(
    metadata: Mapping[str, JSONSerializable] | None,
) -> Mapping[str, JSONSerializable]:
    """Return the validated Parlant-owned root from a metadata mapping.

    Missing roots are treated as empty. Because this feature has a fresh-data
    contract, malformed roots are rejected instead of being interpreted as a
    legacy representation.

    Args:
        metadata: Complete metadata mapping, or ``None``.

    Returns:
        The internal metadata root, or an empty mapping when absent.

    Raises:
        ValueError: If the root or schema version is malformed.
    """

    if not metadata or PARLANT_METADATA_KEY not in metadata:
        return {}

    root = metadata[PARLANT_METADATA_KEY]
    if not isinstance(root, Mapping):
        raise ValueError(f"Metadata '{PARLANT_METADATA_KEY}' value must be an object")

    version = root.get("schema_version")
    if version != PARLANT_METADATA_SCHEMA_VERSION:
        raise ValueError(f"Unsupported {PARLANT_METADATA_KEY} schema version: {version!r}")

    return root


def internal_value(
    metadata: Mapping[str, JSONSerializable] | None,
    path: MetadataPath,
    default: JSONSerializable = None,
) -> JSONSerializable:
    """Read a value below the Parlant root without exposing mutable internals.

    Args:
        metadata: Complete metadata mapping.
        path: Path relative to ``__parlant__``.
        default: Value returned when any path component is absent.

    Returns:
        A defensive copy of the stored value, or ``default``.

    Raises:
        ValueError: If the internal root or an intermediate component is
            malformed.
    """

    current: JSONSerializable | Mapping[str, JSONSerializable] = internal_metadata(metadata)
    for component in path:
        if not isinstance(current, Mapping):
            raise ValueError(
                f"Internal metadata path '{'.'.join(path)}' crosses a non-object value"
            )
        if component not in current:
            return deepcopy(default)
        current = current[component]

    return deepcopy(current)


def set_internal_value(
    metadata: Mapping[str, JSONSerializable],
    path: MetadataPath,
    value: JSONSerializable,
) -> dict[str, JSONSerializable]:
    """Return metadata with one internal path set on a defensive copy.

    Args:
        metadata: Complete source metadata.
        path: Non-empty path relative to ``__parlant__``.
        value: JSON-serializable value to store.

    Returns:
        A new complete metadata dictionary.

    Raises:
        ValueError: If ``path`` is empty or crosses a non-object value.
    """

    if not path:
        raise ValueError("Internal metadata path must not be empty")

    result = deepcopy(dict(metadata))
    existing_root = result.get(PARLANT_METADATA_KEY)
    if existing_root is None:
        root: dict[str, JSONSerializable] = {"schema_version": PARLANT_METADATA_SCHEMA_VERSION}
        result[PARLANT_METADATA_KEY] = root
    elif isinstance(existing_root, Mapping):
        root = deepcopy(dict(existing_root))
        version = root.get("schema_version")
        if version != PARLANT_METADATA_SCHEMA_VERSION:
            raise ValueError(f"Unsupported {PARLANT_METADATA_KEY} schema version: {version!r}")
        result[PARLANT_METADATA_KEY] = root
    else:
        raise ValueError(f"Metadata '{PARLANT_METADATA_KEY}' value must be an object")

    cursor = root
    for component in path[:-1]:
        child = cursor.get(component)
        if child is None:
            next_cursor: dict[str, JSONSerializable] = {}
            cursor[component] = next_cursor
            cursor = next_cursor
        elif isinstance(child, Mapping):
            next_cursor = deepcopy(dict(child))
            cursor[component] = next_cursor
            cursor = next_cursor
        else:
            raise ValueError(
                f"Internal metadata path '{'.'.join(path)}' crosses a non-object value"
            )

    cursor[path[-1]] = deepcopy(value)
    return result


def unset_internal_value(
    metadata: Mapping[str, JSONSerializable],
    path: MetadataPath,
) -> dict[str, JSONSerializable]:
    """Return metadata with one internal path removed.

    Missing paths are a no-op. Empty internal dictionaries are retained so the
    operation never needs to infer whether a parent object has domain meaning.

    Args:
        metadata: Complete source metadata.
        path: Non-empty path relative to ``__parlant__``.

    Returns:
        A new complete metadata dictionary.

    Raises:
        ValueError: If ``path`` is empty or crosses a non-object value.
    """

    if not path:
        raise ValueError("Internal metadata path must not be empty")

    result = deepcopy(dict(metadata))
    root_value = result.get(PARLANT_METADATA_KEY)
    if root_value is None:
        return result
    if not isinstance(root_value, Mapping):
        raise ValueError(f"Metadata '{PARLANT_METADATA_KEY}' value must be an object")

    root = deepcopy(dict(root_value))
    version = root.get("schema_version")
    if version != PARLANT_METADATA_SCHEMA_VERSION:
        raise ValueError(f"Unsupported {PARLANT_METADATA_KEY} schema version: {version!r}")
    result[PARLANT_METADATA_KEY] = root

    cursor = root
    for component in path[:-1]:
        child = cursor.get(component)
        if child is None:
            return result
        if not isinstance(child, Mapping):
            raise ValueError(
                f"Internal metadata path '{'.'.join(path)}' crosses a non-object value"
            )
        next_cursor = deepcopy(dict(child))
        cursor[component] = next_cursor
        cursor = next_cursor

    cursor.pop(path[-1], None)
    return result

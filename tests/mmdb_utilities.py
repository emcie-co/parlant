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

"""A minimal MaxMind DB writer for building GeoIP test fixtures in-process.

Tests build their own tiny City database instead of vendoring a real one: the
records are synthetic, use documentation address ranges (RFC 5737 / RFC 3849),
and so carry no third-party licence. The layout mirrors DB-IP City Lite (an
IPv6 tree with IPv4 mapped under ::/96 and the GeoIP2 City record shape), which
is what production deployments provision.

Format reference: https://maxmind.github.io/MaxMind-DB/
"""

from dataclasses import dataclass
import ipaddress
from pathlib import Path
import struct
from typing import Mapping, Sequence, TypeAlias, Union

_METADATA_MARKER = b"\xab\xcd\xefMaxMind.com"
_DATA_SECTION_SEPARATOR = b"\x00" * 16
_RECORD_SIZE = 32


@dataclass(frozen=True)
class U16:
    value: int


@dataclass(frozen=True)
class U32:
    value: int


@dataclass(frozen=True)
class U64:
    value: int


MmdbValue: TypeAlias = Union[
    str,
    bool,
    int,
    float,
    U16,
    U32,
    U64,
    Sequence["MmdbValue"],
    Mapping[str, "MmdbValue"],
]


def _control(type_number: int, size: int) -> bytes:
    if type_number <= 7:
        first = type_number << 5
        extended = b""
    else:
        first = 0
        extended = bytes([type_number - 7])

    if size < 29:
        return bytes([first | size]) + extended
    if size < 29 + 256:
        return bytes([first | 29]) + extended + bytes([size - 29])
    if size < 285 + 65536:
        return bytes([first | 30]) + extended + (size - 285).to_bytes(2, "big")
    return bytes([first | 31]) + extended + (size - 65821).to_bytes(3, "big")


def _unsigned(type_number: int, value: int) -> bytes:
    payload = value.to_bytes((value.bit_length() + 7) // 8, "big") if value else b""
    return _control(type_number, len(payload)) + payload


def _encode(value: MmdbValue) -> bytes:
    if isinstance(value, bool):
        return _control(14, int(value))
    if isinstance(value, str):
        payload = value.encode("utf-8")
        return _control(2, len(payload)) + payload
    if isinstance(value, U16):
        return _unsigned(5, value.value)
    if isinstance(value, U32):
        return _unsigned(6, value.value)
    if isinstance(value, U64):
        return _unsigned(9, value.value)
    if isinstance(value, int):
        return _unsigned(6, value)
    if isinstance(value, float):
        return _control(3, 8) + struct.pack(">d", value)
    if isinstance(value, Mapping):
        encoded = _control(7, len(value))
        for key, item in value.items():
            encoded += _encode(key) + _encode(item)
        return encoded
    encoded = _control(11, len(value))
    for item in value:
        encoded += _encode(item)
    return encoded


def _network_bits(network: str) -> list[int]:
    parsed = ipaddress.ip_network(network, strict=True)
    if parsed.version == 4:
        # IPv4 lives under ::/96 in an IPv6 tree, as in DB-IP and GeoLite2.
        address = int(parsed.network_address)
        prefix = 96 + parsed.prefixlen
    else:
        address = int(parsed.network_address)
        prefix = parsed.prefixlen
    return [(address >> (127 - i)) & 1 for i in range(prefix)]


class _Node:
    def __init__(self) -> None:
        self.children: list["_Node | int | None"] = [None, None]


def write_city_database(
    path: Path,
    records: Mapping[str, Mapping[str, MmdbValue]],
    database_type: str = "DBIP-City-Lite",
    build_epoch: int = 1_790_000_000,
) -> Path:
    """Write an IPv6 City database mapping each network to its record."""

    data = b""
    root = _Node()

    for network, record in records.items():
        offset = len(data)
        data += _encode(record)

        node = root
        bits = _network_bits(network)
        for bit in bits[:-1]:
            child = node.children[bit]
            if not isinstance(child, _Node):
                child = _Node()
                node.children[bit] = child
            node = child
        node.children[bits[-1]] = offset

    nodes: list[_Node] = []
    queue = [root]
    while queue:
        current = queue.pop(0)
        nodes.append(current)
        queue.extend(child for child in current.children if isinstance(child, _Node))

    index = {id(node): number for number, node in enumerate(nodes)}
    node_count = len(nodes)

    def record_value(child: "_Node | int | None") -> int:
        if child is None:
            return node_count
        if isinstance(child, _Node):
            return index[id(child)]
        return node_count + 16 + child

    tree = b"".join(
        record_value(node.children[0]).to_bytes(_RECORD_SIZE // 8, "big")
        + record_value(node.children[1]).to_bytes(_RECORD_SIZE // 8, "big")
        for node in nodes
    )

    metadata: Mapping[str, MmdbValue] = {
        "binary_format_major_version": U16(2),
        "binary_format_minor_version": U16(0),
        "build_epoch": U64(build_epoch),
        "database_type": database_type,
        "description": {"en": "Synthetic Parlant test fixture"},
        "ip_version": U16(6),
        "languages": ["en"],
        "node_count": U32(node_count),
        "record_size": U16(_RECORD_SIZE),
    }

    path.write_bytes(tree + _DATA_SECTION_SEPARATOR + data + _METADATA_MARKER + _encode(metadata))
    return path


def city_record(
    country_code: str,
    country: str,
    region: str | None = None,
    city: str | None = None,
) -> dict[str, MmdbValue]:
    """A GeoIP2/DB-IP City record with English names only."""

    record: dict[str, MmdbValue] = {
        "country": {"iso_code": country_code, "names": {"en": country}},
    }
    if region is not None:
        record["subdivisions"] = [{"names": {"en": region}}]
    if city is not None:
        record["city"] = {"names": {"en": city}}
    return record

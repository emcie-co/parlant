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

"""Common utilities for application modules, including pagination cursor handling."""

import base64
from parlant.core.persistence.common import Cursor, ObjectId


def encode_cursor(cursor: Cursor) -> str:
    """Encode a cursor to a base64 string for API responses."""
    cursor_str = f"{cursor.creation_utc}|{cursor.id}"
    return base64.b64encode(cursor_str.encode("utf-8")).decode()


def decode_cursor(cursor_str: str) -> Cursor | None:
    """Decode a base64 cursor string from API requests. Returns None if invalid."""
    try:
        decoded_str = base64.b64decode(cursor_str.encode()).decode("utf-8")
        creation_utc, cursor_id = decoded_str.split("|", 1)
        return Cursor(creation_utc=creation_utc, id=ObjectId(cursor_id))
    except Exception:
        return None

import base64
from parlant.core.persistence.common import Cursor, ObjectId


def encode_cursor(cursor: Cursor) -> str:
    """Encode a cursor to a base64 string for API responses"""
    # Simple format: "sort_value|id"
    cursor_str = f"{cursor.sort_value}|{cursor.id}"
    return base64.b64encode(cursor_str.encode("utf-8")).decode()


def decode_cursor(cursor_str: str) -> Cursor | None:
    """Decode a base64 cursor string from API requests. Returns None if invalid."""
    try:
        decoded_str = base64.b64decode(cursor_str.encode()).decode("utf-8")
        sort_value, cursor_id = decoded_str.split("|", 1)
        return Cursor(sort_value=sort_value, id=ObjectId(cursor_id))
    except Exception:
        return None

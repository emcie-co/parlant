from typing import NewType, Optional, TypeAlias
import nanoid  # type: ignore

import emcie.common.types.common

JSONSerializable: TypeAlias = emcie.common.types.common.JSONSerializable

UniqueId = NewType("UniqueId", str)


class ItemNotFoundError(Exception):
    def __init__(self, item_id: UniqueId, message: Optional[str] = None) -> None:
        super().__init__(f"Item '{item_id}' not found" + (f": {message}" if message else ""))


def generate_id() -> UniqueId:
    return UniqueId(nanoid.generate(size=10))

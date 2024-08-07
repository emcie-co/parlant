from typing import NewType
import nanoid  # type: ignore

import emcie.common.types

JSONSerializable = emcie.common.types.JSONSerializable

UniqueId = NewType("UniqueId", str)


def generate_id() -> UniqueId:
    return UniqueId(nanoid.generate(size=10))

from typing import Mapping, NewType, Sequence, TypeVar, Union
import nanoid  # type: ignore


JSONSerializable = Union[
    str,
    int,
    float,
    bool,
    None,
    Mapping[str, "JSONSerializable"],
    Sequence["JSONSerializable"],
]

UniqueId = NewType("UniqueId", str)
T = TypeVar("T")


def generate_id() -> UniqueId:
    return UniqueId(nanoid.generate(size=10))

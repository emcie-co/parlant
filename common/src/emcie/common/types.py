from typing import Mapping, Sequence, Union, TypeAlias


JSONSerializable: TypeAlias = Union[
    str,
    int,
    float,
    bool,
    None,
    Mapping[str, "JSONSerializable"],
    Sequence["JSONSerializable"],
]

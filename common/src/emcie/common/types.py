from typing import Mapping, Sequence, Union


JSONSerializable = Union[
    str,
    int,
    float,
    bool,
    None,
    Mapping[str, "JSONSerializable"],
    Sequence["JSONSerializable"],
]

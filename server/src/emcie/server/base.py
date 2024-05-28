from pydantic import BaseModel, ConfigDict


class EmcieBase(BaseModel):
    """
    Base class for all Emcie Pydantic models.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,
        use_enum_values=True,
    )

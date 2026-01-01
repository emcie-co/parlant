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

from datetime import datetime
import dateutil.parser
from fastapi import APIRouter, Path, Query, Request, status
from pydantic import Field
from typing import Annotated, Mapping, Sequence, TypeAlias

from parlant.api.authorization import AuthorizationPolicy, Operation
from parlant.api.common import (
    SortDirectionDTO,
    apigen_config,
    ExampleJson,
    example_json_content,
    sort_direction_dto_to_sort_direction,
)
from parlant.core.app_modules.common import decode_cursor, encode_cursor
from parlant.core.app_modules.customers import (
    CustomerMetadataUpdateParams,
    CustomerTagUpdateParams,
)
from parlant.core.application import Application
from parlant.core.common import DefaultBaseModel
from parlant.core.customers import CustomerId
from parlant.core.tags import TagId

API_GROUP = "customers"

CustomerNameField: TypeAlias = Annotated[
    str,
    Field(
        description="An arbitrary string that identifies and/or describes the customer",
        examples=["Scooby", "Johan the Mega-VIP"],
        min_length=1,
        max_length=100,
    ),
]

CustomerMetadataField: TypeAlias = Annotated[
    Mapping[str, str],
    Field(
        description="Key-value pairs (`str: str`) to describe the customer",
        examples=[{"email": "scooby@dooby.do", "VIP": "Yes"}],
    ),
]


customer_creation_params_example: ExampleJson = {
    "name": "Scooby",
    "metadata": {
        "email": "scooby@dooby.do",
        "VIP": "Yes",
    },
}


CustomerIdPath: TypeAlias = Annotated[
    CustomerId,
    Path(
        description="Unique identifier for the customer",
        examples=["ck_IdAXUtp"],
        min_length=1,
    ),
]


CustomerCreationUTCField: TypeAlias = Annotated[
    datetime,
    Field(
        description="UTC timestamp of when the customer was created",
        examples=[dateutil.parser.parse("2024-03-24T12:00:00Z")],
    ),
]

TagIdField: TypeAlias = Annotated[
    TagId,
    Field(
        description="Unique identifier for the tag",
        examples=["t9a8g703f4"],
    ),
]

TagIdSequenceField: TypeAlias = Annotated[
    Sequence[TagIdField],
    Field(
        description="Collection of ids of tags that describe the customer",
        examples=[["t9a8g703f4", "4gIAXU4tp"], []],
    ),
]

customer_example: ExampleJson = {
    "id": "ck_IdAXUtp",
    "creation_utc": "2024-03-24T12:00:00Z",
    "name": "Scooby",
    "metadata": {
        "email": "scooby@dooby.do",
        "VIP": "Yes",
    },
    "tags": ["VIP", "New User"],
}


LimitQuery: TypeAlias = Annotated[
    int,
    Query(
        description="Maximum number of items to return",
        ge=1,
        le=100,
        examples=[10, 25],
    ),
]

CursorQuery: TypeAlias = Annotated[
    str,
    Query(
        description="Pagination cursor for fetching the next page of results",
        examples=["AAABjnBU9gBl/0BQt1axI0VniQI="],
    ),
]

SortQuery: TypeAlias = Annotated[
    SortDirectionDTO,
    Query(
        description="Sort direction for results",
        examples=["asc", "desc"],
    ),
]


class CustomerDTO(
    DefaultBaseModel,
    json_schema_extra={"example": customer_example},
):
    """
    Represents a customer in the system.

    Customers are entities that interact with agents through sessions. Each customer
    can have metadata stored in the metadata field and can be tagged for categorization.
    """

    id: CustomerIdPath
    creation_utc: CustomerCreationUTCField
    name: CustomerNameField
    metadata: CustomerMetadataField
    tags: TagIdSequenceField


class PaginatedCustomersDTO(DefaultBaseModel):
    """Paginated response for customers"""

    items: Sequence[CustomerDTO]
    total_count: int
    has_more: bool
    next_cursor: str | None = None


class CustomerCreationParamsDTO(
    DefaultBaseModel,
    json_schema_extra={"example": customer_creation_params_example},
):
    """Parameters for creating a new customer.

    Optional fields:
    - `id`: Custom identifier for the customer. If not provided, an ID will be
      automatically generated. Custom IDs can be any string format and are useful
      for maintaining consistent identifiers across deployments or integrations.
    - `metadata`: Key-value pairs to describe the customer
    - `tags`: List of tag IDs to associate with the customer
    """

    name: CustomerNameField
    id: CustomerIdPath | None = None
    metadata: CustomerMetadataField | None = None
    tags: TagIdSequenceField | None = None


CustomerMetadataUnsetField: TypeAlias = Annotated[
    Sequence[str],
    Field(
        description="Extra metadata keys to remove",
        examples=[["old_email", "old_title"], []],
    ),
]

customer_metadata_update_params_example: ExampleJson = {
    "add": {
        "email": "scooby@dooby.do",
        "VIP": "Yes",
    },
    "remove": ["old_email", "old_title"],
}


class CustomerMetadataUpdateParamsDTO(
    DefaultBaseModel,
    json_schema_extra={"example": customer_metadata_update_params_example},
):
    """Parameters for updating a customer's extra metadata."""

    set: CustomerMetadataField | None = None
    unset: CustomerMetadataUnsetField | None = None


CustomerTagUpdateAddField: TypeAlias = Annotated[
    Sequence[TagIdField],
    Field(
        description="Optional collection of tag ids to add to the customer's tags",
    ),
]

CustomerTagUpdateRemoveField: TypeAlias = Annotated[
    Sequence[TagIdField],
    Field(
        description="Optional collection of tag ids to remove from the customer's tags",
    ),
]

tags_update_params_example: ExampleJson = {
    "add": [
        "t9a8g703f4",
        "tag_456abc",
    ],
    "remove": [
        "tag_789def",
        "tag_012ghi",
    ],
}


class CustomerTagUpdateParamsDTO(
    DefaultBaseModel,
    json_schema_extra={"example": tags_update_params_example},
):
    """
    Parameters for updating a customer's tags.

    Allows adding new tags to and removing existing tags from a customer.
    Both operations can be performed in a single request.
    """

    add: CustomerTagUpdateAddField | None = None
    remove: CustomerTagUpdateRemoveField | None = None


customer_update_params_example: ExampleJson = {
    "name": "Scooby",
    "metadata": customer_metadata_update_params_example,
    "tags": tags_update_params_example,
}


class CustomerUpdateParamsDTO(
    DefaultBaseModel,
    json_schema_extra={"example": customer_update_params_example},
):
    """Parameters for updating a customer's attributes."""

    name: CustomerNameField | None = None
    metadata: CustomerMetadataUpdateParamsDTO | None = None
    tags: CustomerTagUpdateParamsDTO | None = None


def create_router(
    authorization_policy: AuthorizationPolicy,
    app: Application,
) -> APIRouter:
    router = APIRouter()

    @router.post(
        "",
        operation_id="create_customer",
        status_code=status.HTTP_201_CREATED,
        response_model=CustomerDTO,
        responses={
            status.HTTP_201_CREATED: {
                "description": "Customer successfully created. Returns the new customer object.",
                "content": example_json_content(customer_example),
            },
            status.HTTP_422_UNPROCESSABLE_CONTENT: {
                "description": "Validation error in request parameters"
            },
        },
        **apigen_config(group_name=API_GROUP, method_name="create"),
    )
    async def create_customer(
        request: Request,
        params: CustomerCreationParamsDTO,
    ) -> CustomerDTO:
        """
        Creates a new customer in the system.

        A customer may be created with as little as a `name`.
        `metadata` key-value pairs and additional `tags` may be attached to a customer.
        """
        await authorization_policy.authorize(
            request=request,
            operation=Operation.CREATE_CUSTOMER,
        )

        customer = await app.customers.create(
            name=params.name,
            extra=params.metadata if params.metadata else {},
            tags=params.tags,
            id=params.id,
        )

        return CustomerDTO(
            id=customer.id,
            creation_utc=customer.creation_utc,
            name=customer.name,
            metadata=customer.extra,
            tags=customer.tags,
        )

    @router.get(
        "/{customer_id}",
        operation_id="read_customer",
        response_model=CustomerDTO,
        responses={
            status.HTTP_200_OK: {
                "description": "Customer details successfully retrieved. Returns the Customer object.",
                "content": example_json_content(customer_example),
            },
            status.HTTP_404_NOT_FOUND: {
                "description": "Customer not found. The specified customer_id does not exist"
            },
        },
        **apigen_config(group_name=API_GROUP, method_name="retrieve"),
    )
    async def read_customer(
        request: Request,
        customer_id: CustomerIdPath,
    ) -> CustomerDTO:
        """
        Retrieves details of a specific customer by ID.

        Returns a complete customer object including their metadata and tags.
        The customer must exist in the system.
        """
        await authorization_policy.authorize(
            request=request,
            operation=Operation.READ_CUSTOMER,
        )

        customer = await app.customers.read(customer_id=customer_id)

        return CustomerDTO(
            id=customer.id,
            creation_utc=customer.creation_utc,
            name=customer.name,
            metadata=customer.extra,
            tags=customer.tags,
        )

    @router.get(
        "",
        operation_id="list_customers",
        response_model=PaginatedCustomersDTO | Sequence[CustomerDTO],
        responses={
            status.HTTP_200_OK: {
                "description": (
                    "If a cursor is provided, a paginated list of customers will be returned. "
                    "Otherwise, the full list of customers will be returned."
                ),
                "content": {
                    "application/json": {
                        "example": {
                            "items": [customer_example],
                            "total_count": 1,
                            "has_more": False,
                            "next_cursor": None,
                        }
                    }
                },
            },
            status.HTTP_422_UNPROCESSABLE_CONTENT: {
                "description": "Validation error in the request parameters."
            },
        },
        **apigen_config(group_name=API_GROUP, method_name="list"),
    )
    async def list_customers(
        request: Request,
        limit: LimitQuery | None = None,
        cursor: CursorQuery | None = None,
        sort: SortQuery | None = None,
    ) -> PaginatedCustomersDTO | Sequence[CustomerDTO]:
        """
        Retrieves a list of customers from the system.

        If a cursor is provided, the results are returned using cursor-based pagination
        with a configurable sort direction. If no cursor is provided, the full list of
        customers is returned.

        Returns an empty list if no customers exist.

        Note:
            When using paginated results, the first page will always include the special
            'guest' customer as first item.
        """
        await authorization_policy.authorize(
            request=request,
            operation=Operation.LIST_CUSTOMERS,
        )

        customers_result = await app.customers.find(
            limit=limit,
            cursor=decode_cursor(cursor) if cursor else None,
            sort_direction=sort_direction_dto_to_sort_direction(sort) if sort else None,
        )

        if limit is None:
            return [
                CustomerDTO(
                    id=customer.id,
                    creation_utc=customer.creation_utc,
                    name=customer.name,
                    metadata=customer.extra,
                    tags=customer.tags,
                )
                for customer in customers_result.items
            ]

        return PaginatedCustomersDTO(
            items=[
                CustomerDTO(
                    id=customer.id,
                    creation_utc=customer.creation_utc,
                    name=customer.name,
                    metadata=customer.extra,
                    tags=customer.tags,
                )
                for customer in customers_result.items
            ],
            total_count=customers_result.total_count,
            has_more=customers_result.has_more,
            next_cursor=encode_cursor(customers_result.next_cursor)
            if customers_result.next_cursor
            else None,
        )

    @router.patch(
        "/{customer_id}",
        operation_id="update_customer",
        response_model=CustomerDTO,
        responses={
            status.HTTP_200_OK: {
                "description": "Customer successfully updated. Returns the updated Customer object.",
                "content": example_json_content(customer_example),
            },
            status.HTTP_404_NOT_FOUND: {
                "description": "Customer not found. The specified customer_id does not exist"
            },
            status.HTTP_422_UNPROCESSABLE_CONTENT: {
                "description": "Validation error in update parameters"
            },
        },
        **apigen_config(group_name=API_GROUP, method_name="update"),
    )
    async def update_customer(
        request: Request,
        customer_id: CustomerIdPath,
        params: CustomerUpdateParamsDTO,
    ) -> CustomerDTO:
        """
        Updates an existing customer's attributes.

        Only provided attributes will be updated; others remain unchanged.
        The customer's ID and creation timestamp cannot be modified.
        Extra metadata and tags can be added or removed independently.
        """
        await authorization_policy.authorize(
            request=request,
            operation=Operation.UPDATE_CUSTOMER,
        )

        customer = await app.customers.update(
            customer_id=customer_id,
            name=params.name,
            metadata=CustomerMetadataUpdateParams(
                set=params.metadata.set,
                unset=params.metadata.unset,
            )
            if params.metadata
            else None,
            tags=CustomerTagUpdateParams(
                add=params.tags.add,
                remove=params.tags.remove,
            )
            if params.tags
            else None,
        )

        return CustomerDTO(
            id=customer.id,
            creation_utc=customer.creation_utc,
            name=customer.name,
            metadata=customer.extra,
            tags=customer.tags,
        )

    @router.delete(
        "/{customer_id}",
        operation_id="delete_customer",
        status_code=status.HTTP_204_NO_CONTENT,
        responses={
            status.HTTP_204_NO_CONTENT: {
                "description": "Customer successfully deleted. No content returned."
            },
            status.HTTP_404_NOT_FOUND: {
                "description": "Customer not found. The specified customer_id does not exist"
            },
        },
        **apigen_config(group_name=API_GROUP, method_name="delete"),
    )
    async def delete_customer(
        request: Request,
        customer_id: CustomerIdPath,
    ) -> None:
        """
        Deletes a customer from the agent.

        Deleting a non-existent customer will return 404.
        No content will be returned from a successful deletion.
        """
        await authorization_policy.authorize(
            request=request,
            operation=Operation.DELETE_CUSTOMER,
        )

        await app.customers.delete(customer_id=customer_id)

    return router

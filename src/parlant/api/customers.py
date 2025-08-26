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

from datetime import datetime
import dateutil.parser
from fastapi import APIRouter, Path, Request, status
from pydantic import Field
from typing import Annotated, Mapping, Optional, Sequence, TypeAlias

from parlant.api.authorization import AuthorizationPolicy, Operation
from parlant.api.common import apigen_config, ExampleJson, example_json_content
from parlant.core.agents import AgentStore, AgentId
from parlant.core.common import DefaultBaseModel
from parlant.core.customers import CustomerId, CustomerStore
from parlant.core.tags import Tag, TagId, TagStore

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


class CustomerCreationParamsDTO(
    DefaultBaseModel,
    json_schema_extra={"example": customer_creation_params_example},
):
    """Parameters for creating a new customer."""

    name: CustomerNameField
    metadata: Optional[CustomerMetadataField] = None
    tags: Optional[TagIdSequenceField] = None


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

    set: Optional[CustomerMetadataField] = None
    unset: Optional[CustomerMetadataUnsetField] = None


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

    add: Optional[CustomerTagUpdateAddField] = None
    remove: Optional[CustomerTagUpdateRemoveField] = None


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

    name: Optional[CustomerNameField] = None
    metadata: Optional[CustomerMetadataUpdateParamsDTO] = None
    tags: Optional[CustomerTagUpdateParamsDTO] = None


def create_router(
    authorization_policy: AuthorizationPolicy,
    customer_store: CustomerStore,
    tag_store: TagStore,
    agent_store: AgentStore,
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
            status.HTTP_422_UNPROCESSABLE_ENTITY: {
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

        tags = []

        if params.tags:
            for tag_id in params.tags:
                if agent_id := Tag.extract_agent_id(tag_id):
                    _ = await agent_store.read_agent(agent_id=AgentId(agent_id))
                else:
                    _ = await tag_store.read_tag(tag_id=tag_id)

            tags = list(set(params.tags))

        customer = await customer_store.create_customer(
            name=params.name,
            extra=params.metadata if params.metadata else {},
            tags=tags or None,
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

        customer = await customer_store.read_customer(customer_id=customer_id)

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
        response_model=Sequence[CustomerDTO],
        responses={
            status.HTTP_200_OK: {
                "description": "List of all customers in the system.",
                "content": example_json_content(customer_example),
            },
        },
        **apigen_config(group_name=API_GROUP, method_name="list"),
    )
    async def list_customers(request: Request) -> Sequence[CustomerDTO]:
        """
        Retrieves a list of all customers in the system.

        Returns an empty list if no customers exist.
        Customers are returned in no guaranteed order.
        """
        await authorization_policy.authorize(
            request=request,
            operation=Operation.LIST_CUSTOMERS,
        )

        customers = await customer_store.list_customers()

        return [
            CustomerDTO(
                id=customer.id,
                creation_utc=customer.creation_utc,
                name=customer.name,
                metadata=customer.extra,
                tags=customer.tags,
            )
            for customer in customers
        ]

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
            status.HTTP_422_UNPROCESSABLE_ENTITY: {
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

        if params.name:
            _ = await customer_store.update_customer(
                customer_id=customer_id,
                params={"name": params.name},
            )

        if params.metadata:
            if params.metadata.set:
                await customer_store.add_extra(customer_id, params.metadata.set)
            if params.metadata.unset:
                await customer_store.remove_extra(customer_id, params.metadata.unset)

        if params.tags:
            if params.tags.add:
                for tag_id in params.tags.add:
                    if agent_id := Tag.extract_agent_id(tag_id):
                        _ = await agent_store.read_agent(agent_id=AgentId(agent_id))
                    else:
                        _ = await tag_store.read_tag(tag_id=tag_id)
                    await customer_store.upsert_tag(customer_id, tag_id)
            if params.tags.remove:
                for tag_id in params.tags.remove:
                    await customer_store.remove_tag(customer_id, tag_id)

        customer = await customer_store.read_customer(customer_id=customer_id)

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

        await customer_store.delete_customer(customer_id=customer_id)

    return router

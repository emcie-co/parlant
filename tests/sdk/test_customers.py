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

from parlant.core.customers import CustomerStore
import parlant.sdk as p
from tests.sdk.utils import Context, SDKTest


class Test_that_a_customer_can_be_read(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.customer = await server.create_customer(
            name="John Doe", metadata={"email": "john.doe@example.com"}
        )

    async def run(self, ctx: Context) -> None:
        customer_store = ctx.container[CustomerStore]

        customer = await customer_store.read_customer(self.customer.id)

        assert customer.name == "John Doe"
        assert customer.extra == {"email": "john.doe@example.com"}
        assert customer.id == self.customer.id


class Test_that_customers_can_be_listed(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.c1 = await server.create_customer(
            name="John Doe",
        )

        self.c2 = await server.create_customer(
            name="Jane Smith",
        )

        self.customers = await server.list_customers()

    async def run(self, ctx: Context) -> None:
        assert len(self.customers) == 3  # Including the guest customer
        assert self.customers[0].id == CustomerStore.GUEST_ID
        assert self.customers[1].id == self.c1.id
        assert self.customers[2].id == self.c2.id


class Test_that_a_customer_can_be_found_by_name(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.c1 = await server.create_customer(
            name="John Doe",
        )

        self.c2 = await server.create_customer(
            name="Jane Smith",
        )

        self.customer = await server.find_customer(name="John Doe")

    async def run(self, ctx: Context) -> None:
        assert self.customer is not None
        assert self.customer.id == self.c1.id


class Test_that_a_customer_can_be_found_by_id(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.c1 = await server.create_customer(
            name="John Doe",
        )

        self.customer = await server.find_customer(id=self.c1.id)

    async def run(self, ctx: Context) -> None:
        assert self.customer is not None
        assert self.customer.id == self.c1.id


class Test_that_a_customer_can_be_created_with_custom_id(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.customer = await server.create_customer(
            id="my-custom-customer-id",
            name="Custom ID Customer",
            metadata={"email": "custom@example.com"},
        )

    async def run(self, ctx: Context) -> None:
        assert self.customer.id == "my-custom-customer-id"

        # Verify the customer can be retrieved with the custom ID
        retrieved_customer = await ctx.server.find_customer(id="my-custom-customer-id")
        assert retrieved_customer is not None
        assert retrieved_customer.id == "my-custom-customer-id"
        assert retrieved_customer.name == "Custom ID Customer"

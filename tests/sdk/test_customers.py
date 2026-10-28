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
            name="John Doe", metadata={"email": "john.doe@example.com"}, id="customers-cust-1"
        )

    async def run(self, ctx: Context) -> None:
        customer_store = ctx.container[CustomerStore]

        customer = await customer_store.read_customer(self.customer.id)

        assert customer.name == "John Doe"
        assert customer.metadata == {"email": "john.doe@example.com"}
        assert customer.id == self.customer.id


class Test_that_customers_can_be_listed(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.c1 = await server.create_customer(
            name="John Doe",
            id="customers-cust-2",
        )

        self.c2 = await server.create_customer(
            name="Jane Smith",
            id="customers-cust-3",
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
            id="customers-cust-4",
        )

        self.c2 = await server.create_customer(
            name="Jane Smith",
            id="customers-cust-5",
        )

        self.customer = await server.find_customer(name="John Doe")

    async def run(self, ctx: Context) -> None:
        assert self.customer is not None
        assert self.customer.id == self.c1.id


class Test_that_a_customer_can_be_found_by_id(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.c1 = await server.create_customer(
            name="John Doe",
            id="customers-cust-6",
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


class Test_that_a_customers_group_memberships_can_be_read_and_updated(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.vip = await server.create_group(name="vip")
        self.beta = await server.create_group(name="beta")

        self.customer = await server.create_customer(
            name="Grouped",
            id="customers-cust-groups",
            groups=[self.vip],
        )

    async def run(self, ctx: Context) -> None:
        # Sync reads over the initial snapshot.
        assert len(self.customer.groups) == 1
        assert self.vip in self.customer.groups
        assert self.beta not in self.customer.groups
        assert self.customer.groups[0].id == self.vip.id
        assert [group.id for group in self.customer.groups] == [self.vip.id]

        # A group id is also accepted by membership checks.
        assert self.vip.id in self.customer.groups

        # add() persists and refreshes the snapshot.
        await self.customer.groups.add(self.beta)
        assert len(self.customer.groups) == 2
        assert self.beta in self.customer.groups

        # remove() persists and refreshes the snapshot.
        await self.customer.groups.remove(self.vip)
        assert self.vip not in self.customer.groups
        assert len(self.customer.groups) == 1

        # The change is durable — a freshly read customer reflects it.
        reread = await ctx.server.get_customer(id=self.customer.id)
        assert self.beta in reread.groups
        assert self.vip not in reread.groups


class Test_that_a_customers_kind_is_readable_from_every_retrieval_path(SDKTest):
    """The kind has to survive each way of getting a customer back.

    Each accessor rebuilds the SDK wrapper from the stored customer, so a path
    that forgets to carry the kind reports a persona as real. That failure is
    silent and it fails in the permissive direction, which is exactly the one
    worth a test.
    """

    async def setup(self, server: p.Server) -> None:
        self.persona = await server.create_customer(
            name="Dana Levi",
            id="customers-persona-1",
            kind=p.CustomerKind.PERSONA,
        )

        self.real = await server.create_customer(
            name="John Doe",
            id="customers-real-1",
        )

        self.fetched_persona = await server.get_customer(id=self.persona.id)
        self.found_persona = await server.find_customer(name="Dana Levi")
        self.listed = await server.list_customers()

    async def run(self, ctx: Context) -> None:
        assert self.persona.kind is p.CustomerKind.PERSONA
        assert self.real.kind is p.CustomerKind.REAL

        assert self.fetched_persona.kind is p.CustomerKind.PERSONA

        assert self.found_persona is not None
        assert self.found_persona.kind is p.CustomerKind.PERSONA

        by_id = {customer.id: customer for customer in self.listed}

        assert by_id[self.persona.id].kind is p.CustomerKind.PERSONA
        assert by_id[self.real.id].kind is p.CustomerKind.REAL
        assert by_id[CustomerStore.GUEST_ID].kind is p.CustomerKind.REAL

        assert p.Customer.guest.kind is p.CustomerKind.REAL


class Test_that_a_customer_reports_whether_it_is_a_persona(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.persona = await server.create_customer(
            name="Dana Levi",
            id="customers-persona-2",
            kind=p.CustomerKind.PERSONA,
        )

        self.real = await server.create_customer(
            name="John Doe",
            id="customers-real-2",
        )

    async def run(self, ctx: Context) -> None:
        assert self.persona.is_persona
        assert not self.real.is_persona
        assert not p.Customer.guest.is_persona

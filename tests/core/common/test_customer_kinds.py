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

"""A customer is either a real customer or a persona.

A persona stands in for a customer in tests, so that a test run never reaches a
real person's data or triggers a tool on their behalf. The distinction is only
worth anything if it cannot be set by accident, cannot be changed afterwards,
and defaults to the restrictive answer whenever it is absent.
"""

from datetime import datetime, timezone
from typing import cast

import pytest

from parlant.adapters.db.transient import TransientDocumentDatabase
from parlant.core.common import IdGenerator, Version
from parlant.core.customers import (
    CustomerDocumentStore,
    CustomerKind,
    CustomerStore,
)
from parlant.core.persistence.common import ObjectId
from parlant.core.persistence.document_database import BaseDocument


@pytest.mark.asyncio
async def test_that_a_customer_is_real_unless_created_as_a_persona() -> None:
    async with CustomerDocumentStore(IdGenerator(), TransientDocumentDatabase()) as store:
        customer = await store.create_customer(name="Real Customer")

        assert customer.kind is CustomerKind.REAL


@pytest.mark.asyncio
async def test_that_a_customer_created_as_a_persona_is_read_back_as_a_persona() -> None:
    async with CustomerDocumentStore(IdGenerator(), TransientDocumentDatabase()) as store:
        created = await store.create_customer(
            name="Persona Customer",
            kind=CustomerKind.PERSONA,
        )

        assert created.kind is CustomerKind.PERSONA
        assert (await store.read_customer(created.id)).kind is CustomerKind.PERSONA


@pytest.mark.asyncio
async def test_that_a_customer_document_written_without_a_kind_is_read_back_as_real() -> None:
    """Documents stored before the field existed carry no kind at all.

    The absent value has to resolve to the restrictive answer. A record that
    predates the distinction describes a customer nobody ever vetted as safe
    to test against, so reading it as a persona would hand a test run exactly
    the real data the field exists to keep it away from.

    The loader is where this is settled, rather than at each read. A query may
    filter on the kind, and `matches_filters` subscripts the candidate rather
    than probing it, so a document that reached a collection still missing the
    field would raise instead of simply failing to match. Filling it in on the
    way in keeps every later query honest. Note that the document could not be
    inserted in this state either: `ensure_is_total` requires every key, so a
    stored document is the only way to encounter one.
    """
    async with CustomerDocumentStore(IdGenerator(), TransientDocumentDatabase()) as store:
        stored_before_the_field_existed = {
            "id": ObjectId("legacy_customer"),
            "version": Version.String("0.1.0"),
            "creation_utc": datetime.now(timezone.utc).isoformat(),
            "modified_utc": datetime.now(timezone.utc).isoformat(),
            "name": "Legacy Customer",
            "name_lower": "legacy customer",
            "metadata": {},
        }

        loaded = await store._document_loader(cast(BaseDocument, stored_before_the_field_existed))

        assert loaded is not None
        assert loaded["kind"] == CustomerKind.REAL.value
        assert CustomerKind(loaded["kind"]) is CustomerKind.REAL


@pytest.mark.asyncio
async def test_that_a_customers_kind_cannot_be_changed_after_creation() -> None:
    """There is no way to promote a real customer into a persona.

    Flipping the mark on a live customer is the accident the mark exists to
    prevent, so the update surface does not carry it. This test fails at type
    check time as much as at run time: it is here so that adding ``kind`` to
    the update params later is a deliberate act with a red test attached.
    """
    async with CustomerDocumentStore(IdGenerator(), TransientDocumentDatabase()) as store:
        customer = await store.create_customer(name="Real Customer")

        from parlant.core.customers import CustomerUpdateParams

        assert "kind" not in CustomerUpdateParams.__annotations__

        await store.update_customer(customer.id, params={"name": "Renamed"})

        assert (await store.read_customer(customer.id)).kind is CustomerKind.REAL


@pytest.mark.asyncio
async def test_that_personas_can_be_listed_apart_from_real_customers() -> None:
    async with CustomerDocumentStore(IdGenerator(), TransientDocumentDatabase()) as store:
        real = await store.create_customer(name="Real Customer")
        persona = await store.create_customer(name="Persona", kind=CustomerKind.PERSONA)

        personas = await store.list_customers(kind=CustomerKind.PERSONA)
        reals = await store.list_customers(kind=CustomerKind.REAL)

        assert [c.id for c in personas.items] == [persona.id]
        assert real.id in [c.id for c in reals.items]
        assert persona.id not in [c.id for c in reals.items]


@pytest.mark.asyncio
async def test_that_the_guest_customer_is_real() -> None:
    """The guest is nobody in particular, but it is not a persona.

    Guest sessions carry production traffic, so a test run must not be handed
    one on the grounds that no named customer is attached.
    """
    async with CustomerDocumentStore(IdGenerator(), TransientDocumentDatabase()) as store:
        guest = await store.read_customer(CustomerStore.GUEST_ID)

        assert guest.kind is CustomerKind.REAL

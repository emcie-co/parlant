# Copyright 2024 Emcie Co Ltd.
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

from pytest_bdd import given, parsers

from parlant.core.customers import CustomerStore, CustomerId
from parlant.core.sessions import SessionStore, SessionId
from parlant.core.tags import TagStore
from tests.core.engines.alpha.utils import ContextOfTest, step


@step(given, parsers.parse('a customer with the name "{name}"'), target_fixture='customer_id')
def given_a_customer(
    context: ContextOfTest,
    name: str,
) -> CustomerId:
    customer_store = context.container[CustomerStore]

    customer = context.sync_await(customer_store.create_customer(name))

    return customer.id


@step(given, parsers.parse('a customer tagged as "{tag_name}"'))
def given_a_customer_tag(
    context: ContextOfTest,
    session_id: SessionId,
    tag_name: str,
) -> None:
    session_store = context.container[SessionStore]
    customer_store = context.container[CustomerStore]
    tag_store = context.container[TagStore]

    tag = context.sync_await(tag_store.create_tag(tag_name))

    customer_id = context.sync_await(session_store.read_session(session_id)).customer_id

    context.sync_await(
        customer_store.add_tag(
            customer_id=customer_id,
            tag_id=tag.id,
        )
    )

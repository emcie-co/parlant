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

"""The project `parlant init` generates.

The layout packages by *domain* rather than by type. A capability's policies,
tools, and the service they call live together, because that is how they are
written, reviewed, and deleted — and because tools attach to rules, so
splitting them across `tools/` and `policies/` separates two halves of one
thing.

Three scaffolds share that layout. `basic` is the structure and nothing else,
for someone who knows what they're building. `retail-store` fills it in with a
working support agent, and is where the framework's features are on display.
`lead-gen` fills it in with a website agent that qualifies visitors, scores
them in code, and books meetings.
"""

from dataclasses import dataclass
from enum import Enum
import shutil
import stat
from pathlib import Path


class PackageManager(Enum):
    UV = "uv"
    PIP = "pip"
    CONDA = "conda"
    POETRY = "poetry"


class Scaffold(Enum):
    BASIC = "basic"
    RETAIL_STORE = "retail-store"
    LEAD_GEN = "lead-gen"


@dataclass(frozen=True)
class ManagerDetection:
    manager: PackageManager
    installed: bool
    label: str
    note: str


def detect_package_managers() -> list[ManagerDetection]:
    """Which package managers this machine can actually use, in menu order."""
    return [
        ManagerDetection(
            PackageManager.UV,
            shutil.which("uv") is not None,
            "uv",
            "recommended — fastest, and manages the Python version too",
        ),
        ManagerDetection(
            # We are running under a Python interpreter, so this always works.
            PackageManager.PIP,
            True,
            "pip + requirements.txt",
            "works everywhere, no extra tooling",
        ),
        ManagerDetection(
            PackageManager.CONDA,
            shutil.which("conda") is not None,
            "conda",
            "for existing conda workflows",
        ),
        ManagerDetection(
            PackageManager.POETRY,
            shutil.which("poetry") is not None,
            "poetry",
            "for existing poetry workflows",
        ),
    ]


# The command each manager uses to run something inside the project.
_RUN_PREFIX = {
    PackageManager.UV: "uv run",
    PackageManager.PIP: "python",
    PackageManager.CONDA: "conda run --no-capture-output -n {env}  python",
    PackageManager.POETRY: "poetry run python",
}


def run_prefix(manager: PackageManager, env_name: str) -> str:
    return _RUN_PREFIX[manager].format(env=env_name)


def install_command(manager: PackageManager, env_name: str) -> str:
    return {
        PackageManager.UV: "uv sync",
        PackageManager.PIP: "python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt",
        PackageManager.CONDA: f"conda env create -f environment.yml -n {env_name}",
        PackageManager.POETRY: "poetry install",
    }[manager]


def run_install_sh(manager: PackageManager, env_name: str) -> str:
    """Idempotent dependency setup for the unix start script, run before each launch
    so ``./run.sh`` works on a fresh checkout with no separate install step. Empty for
    uv, whose ``uv run`` already syncs the environment itself."""
    return {
        PackageManager.UV: "",
        PackageManager.PIP: (
            "[ -d .venv ] || python -m venv .venv\n"
            ". .venv/bin/activate\n"
            "pip install -q -r requirements.txt"
        ),
        PackageManager.CONDA: (
            f'conda env list | grep -q "{env_name}" || '
            f"conda env create -f environment.yml -n {env_name}"
        ),
        PackageManager.POETRY: "poetry install",
    }[manager]


def run_install_cmd(manager: PackageManager, env_name: str) -> str:
    """Windows counterpart of :func:`run_install_sh`."""
    return {
        PackageManager.UV: "",
        PackageManager.PIP: (
            "if not exist .venv python -m venv .venv\n"
            "call .venv\\Scripts\\activate.bat\n"
            "pip install -q -r requirements.txt"
        ),
        PackageManager.CONDA: (
            f'conda env list | findstr /C:"{env_name}" >nul 2>&1 || '
            f"conda env create -f environment.yml -n {env_name}"
        ),
        PackageManager.POETRY: "poetry install",
    }[manager]


# --- basic ----------------------------------------------------------------


BASIC_MAIN_PY = '''"""The entry point: build the server, create the agent, install the domains."""

import asyncio

from config import create_server


async def main() -> None:
    async with create_server() as server:
        await server.create_agent(
            id="agent",
            name="Agent",
            instructions=(
                "Describe who this agent is and what it does. Instructions are meant "
                "to set the agent's general orientation, while policies guide its "
                "behavior where the model's default behavior isn't what you want."
            ),
        )

        # Install a domain per capability. Keep the agent above in a variable
        # and pass it in:
        #
        #     from domains import orders
        #     await orders.install(agent)


if __name__ == "__main__":
    asyncio.run(main())
'''


BASIC_CONFIG_PY = '''"""Server settings, and the only place that branches on the environment.

`PARLANT_ENV` selects it: development (the default), test, or production.
"""

import parlant.sdk as p
import services
from dotenv import load_dotenv


def create_server() -> p.Server:
    """Build the server for the current environment."""
    # Reads .env, if there is one. Real environment variables win.
    load_dotenv()

    services.use(_services_for(p.resolve_environment()))

    return p.Server(
        models={"*": "openrouter"},
        log_level=p.LogLevel.DEBUG,
    )


def _services_for(environment: p.Environment) -> services.Services:
    """The implementations this environment runs against, e.g.:

        if environment is p.Environment.PRODUCTION:
            return services.Services(orders=OrdersAPI(os.environ["ORDERS_API_URL"]))

        return services.Services(orders=OrdersAPI(os.environ["ORDERS_DEV_API_URL"]))
    """
    return services.Services()
'''


BASIC_DOMAINS_INIT = '''"""Each domain owns one capability: its policies, its tools, and the service
they call.

Add a domain as a package here:

    domains/<name>/__init__.py    exposes the domain's install(agent)
    domains/<name>/domain.py      install(agent): policies, observations,
                                  canned responses, glossary, variables
    domains/<name>/tools.py       thin tools that read services.py and call them
    domains/<name>/service.py     the port, plus a real and a fake adapter

Then install it in main.py.
"""
'''


BASIC_TESTS_INIT = '''"""Your tests go here, mirroring the code they cover:

    tests/domains/<name>/    the tests for domains/<name>/

Nothing calls a model. Empty for now, so pytest will report that it collected
no tests until you add the first one.
"""
'''


RETAIL_STORE_SERVICES_PY = '''"""Your services: tools read them from here, and config.py fills them in.

A test swaps them with one `use()` call.
"""

from dataclasses import dataclass

from domains.disputes.service import DisputesService
from domains.orders.service import OrdersService


@dataclass(frozen=True)
class Services:
    orders: OrdersService
    disputes: DisputesService


_installed: Services | None = None


def use(services: Services) -> None:
    """Install the services this process runs with. Called from config.py."""
    global _installed
    _installed = services


def installed() -> Services:
    if _installed is None:
        raise RuntimeError(
            "No services are installed. create_server() in config.py does this; "
            "a test should call services.use(...) with its own implementations."
        )

    return _installed


def orders() -> OrdersService:
    return installed().orders


def disputes() -> DisputesService:
    return installed().disputes
'''


BASIC_SERVICES_PY = '''"""Your services: tools read them from here, and config.py fills them in.

Add a field per service, and an accessor for tools to call:

    @dataclass(frozen=True)
    class Services:
        orders: OrdersService

    def orders() -> OrdersService:
        return installed().orders
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Services:
    """Add a field per service your tools depend on."""


_installed: Services | None = None


def use(services: Services) -> None:
    """Install the services this process runs with. Called from config.py."""
    global _installed
    _installed = services


def installed() -> Services:
    if _installed is None:
        raise RuntimeError(
            "No services are installed. create_server() in config.py does this; "
            "a test should call services.use(...) with its own implementations."
        )

    return _installed
'''


# --- retail-store -----------------------------------------------------------


RETAIL_STORE_MAIN_PY = '''"""The entry point: build the server, create the agent, install the domains."""

import asyncio

import parlant.sdk as p

from config import create_server
from domains import disputes, orders


async def main() -> None:
    async with create_server() as server:
        agent = await server.create_agent(
            id="support",
            name="Homeware Support",
            instructions=(
                "You are a support agent for an online homeware store. You "
                "help customers with order status, shipment tracking, "
                "disputes, and refunds. Be warm and concise."
            ),
            # The baseline for an ordinary turn. Individual rules raise it
            # where care matters more than cost; see domains/disputes.
            effort=p.Effort.MEDIUM,
        )

        # Agent-wide canned response: not tied to any one rule, so it is a
        # candidate on every turn and used only where it fits.
        await agent.create_canned_response(
            "Howdy partner, is there anything I can help you with today?",
            note="Talk like a cowboy, in general",
        )

        # Each domain installs what it owns: its policies, its tools, and the
        # vocabulary that goes with them.
        await orders.install(agent)
        await disputes.install(agent)


if __name__ == "__main__":
    asyncio.run(main())
'''


RETAIL_STORE_CONFIG_PY = '''"""Server settings, and the only place that branches on the environment.

`PARLANT_ENV` selects it: development (the default), test, or production.
"""

import os

import parlant.sdk as p
import services
from dotenv import load_dotenv
from domains.disputes.service import DisputesAPI, FakeDisputes
from domains.orders.service import FakeOrders, OrdersAPI


def create_server() -> p.Server:
    """Build the server for the current environment."""
    # Reads .env, if there is one. Real environment variables win.
    load_dotenv()

    services.use(_services_for(p.resolve_environment()))

    return p.Server(
        models={"*": "openrouter"},
        log_level=p.LogLevel.DEBUG,
    )


def _services_for(environment: p.Environment) -> services.Services:
    """The implementations this environment runs against.

    This example uses the fakes outside production so it runs with no backend
    and no credentials. A real project usually points development at a
    development instance of the service instead.
    """
    if environment is p.Environment.PRODUCTION:
        return services.Services(
            orders=OrdersAPI(os.environ["ORDERS_API_URL"]),
            disputes=DisputesAPI(os.environ["DISPUTES_API_URL"]),
        )

    return services.Services(orders=FakeOrders(), disputes=FakeDisputes())
'''


RETAIL_STORE_DOMAINS_INIT = '''"""Each domain owns one capability: its policies, its tools, and the service
they call. Add a domain by adding a package here and installing it in main.py.

    orders/     order status and shipment tracking
    disputes/   disputes and refunds, including the damaged-item flow

One domain may use another's service, never its tools: disputes reads
`OrdersService` to check delivery, and keeps its own tools for the rest.
"""
'''


ORDERS_INIT = '''"""The orders domain: where an order is, and where its shipment is."""

from domains.orders.domain import install

__all__ = ["install"]
'''


ORDERS_TOOLS = '''"""The domain's tools: read the service, call it, wrap the result.

Keep them thin — the logic worth testing belongs in service.py.
"""

import parlant.sdk as p
import services


@p.tool
async def check_order_status(order_id: str) -> p.ToolResult:
    """Look up one order's current status.

    The order number comes from the conversation; the customer's identity
    comes from the session. Never take identity from the conversation.
    """
    orders = services.orders()

    order = await orders.find(p.Customer.current.id, order_id)

    if order is None:
        # The same answer whether it doesn't exist or belongs to someone else.
        return p.ToolResult(
            data=(
                f"This customer has no order {order_id}. Ask them to "
                "double-check the number."
            )
        )

    return p.ToolResult(
        data=(
            f"Order {order.id} ({order.item}, ${order.total:.2f}): "
            f"{order.status}, expected {order.expected_delivery}."
        )
    )


@p.tool
async def list_open_orders(context: p.ToolContext) -> p.ToolResult:
    """Every order of this customer's that hasn't been delivered yet.

    Backs the `open_orders` context variable, so it takes no conversation
    arguments and runs on a schedule for whoever the customer is.
    """
    orders = services.orders()

    open_orders = [
        order
        for order in await orders.list_for_customer(context.customer_id)
        if not order.delivered
    ]

    if not open_orders:
        return p.ToolResult(data="This customer has no open orders.")

    return p.ToolResult(
        data=[
            f"Order {order.id} ({order.item}): {order.status}, "
            f"expected {order.expected_delivery}."
            for order in open_orders
        ]
    )


@p.tool
async def track_shipment(order_id: str) -> p.ToolResult:
    """Look up the carrier and tracking number for an order's shipment."""
    orders = services.orders()

    order = await orders.find(p.Customer.current.id, order_id)

    if order is None:
        return p.ToolResult(
            data=(
                f"This customer has no order {order_id}. Ask them to "
                "double-check the number."
            )
        )

    if not order.tracking_number:
        return p.ToolResult(
            data=(
                f"Order {order.id} is {order.status} and has no tracking "
                f"number yet. It is expected {order.expected_delivery}."
            )
        )

    # Fields fill the `{{ }}` placeholders in this turn's canned responses.
    return p.ToolResult(
        data=(
            f"Order {order.id} is on its way with {order.carrier}, "
            f"tracking {order.tracking_number}, expected "
            f"{order.expected_delivery}."
        ),
        canned_response_fields={
            "item": order.item,
            "carrier": order.carrier or "",
            "tracking_number": order.tracking_number,
            "expected_delivery": order.expected_delivery,
        },
    )
'''


ORDERS_DOMAIN = '''"""Everything this domain declares on the agent.

One `install`: glossary, canned responses, policies, observations, variables.
"""

import parlant.sdk as p

from domains.orders import tools


async def install(agent: p.Agent) -> None:
    # A glossary term: vocabulary the agent should read the way you mean it.
    await agent.create_term(
        name="delivery estimate",
        description=(
            "The day an order is currently expected to arrive. It is an "
            "estimate from the carrier, not a guarantee, and it can move."
        ),
        synonyms=["ETA", "expected delivery"],
    )

    # A context variable: loaded per customer, refreshed on a schedule, and
    # in the agent's context from the first turn.
    await agent.create_variable(
        name="open_orders",
        description=(
            "The customer's orders that have not been delivered yet, so the "
            "agent knows what they are most likely asking about."
        ),
        tool=tools.list_open_orders,
        freshness_rules="*/15 * * * *",
    )

    shipment_update = await agent.create_canned_response(
        template=(
            "Your {{ item }} is on its way with {{ carrier }} — tracking "
            "{{ tracking_number }}. It's expected {{ expected_delivery }}."
        ),
        # Declaring what the template needs keeps it out of turns where no
        # tool staged those values. Without this it would still be offered,
        # with its placeholders unfilled.
        field_dependencies=["item", "carrier", "tracking_number", "expected_delivery"],
        note=(
            "Give the tracking number in full. Say the delivery estimate as "
            "an estimate; never state it as a guaranteed date."
        ),
    )

    # A policy document: one complete section, given to the agent whole
    # whenever the situation comes up.
    await agent.create_policy(
        title="Order status and shipment",
        content="""\\
When the customer asks where an order is, what is happening with it, or when
it will arrive:

1. If they haven't said which order, ask for the order number first.
2. Look the order up with `check_order_status`.
3. If they ask specifically about the shipment, the carrier, or tracking,
   also call `track_shipment` and give them the carrier and tracking number.
4. Tell them the current status and the delivery estimate. An order that has
   not shipped yet has no tracking number — say so plainly rather than
   apologizing for it.
""",
        tools=[tools.check_order_status, tools.track_shipment],
        canned_responses=[shipment_update],
    )

    # A steering guideline: how the agent talks, in force alongside whatever
    # else is going on.
    await agent.create_policy(
        "Always give delivery dates as estimates, and never promise "
        "an exact delivery date or time",
    )
'''


ORDERS_SERVICE = '''"""The orders backend, behind a port.

Which implementation runs is decided in config.py, so nothing here is baked
into the tools.
"""

from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable


@dataclass(frozen=True)
class Order:
    id: str
    customer_id: str
    item: str
    status: str
    expected_delivery: str
    total: float
    carrier: str | None = None
    tracking_number: str | None = None

    @property
    def delivered(self) -> bool:
        return self.status == "delivered"


# runtime_checkable so the container can bind an implementation to this port.
@runtime_checkable
class OrdersService(Protocol):
    async def find(self, customer_id: str, order_id: str) -> Order | None:
        """This customer's order, or None if they have no such order.

        An order belonging to someone else must come back as None, exactly
        like one that doesn't exist.
        """
        ...

    async def list_for_customer(self, customer_id: str) -> Sequence[Order]:
        """Every order belonging to this customer."""
        ...


class OrdersAPI:
    """Your real orders backend. Implement this against whatever you have.

    Every method raises until you do; development runs use `FakeOrders`.
    """

    def __init__(self, base_url: str) -> None:
        raise NotImplementedError("Implement OrdersAPI against your orders backend.")

    async def find(self, customer_id: str, order_id: str) -> Order | None:
        raise NotImplementedError

    async def list_for_customer(self, customer_id: str) -> Sequence[Order]:
        raise NotImplementedError


# Sample data for the fake. It belongs to "guest", who you are when you chat
# locally without signing in.
SAMPLE_ORDERS = {
    "1234": Order(
        id="1234",
        customer_id="guest",
        item="ceramic table lamp",
        status="shipped",
        expected_delivery="Thursday",
        total=89.00,
        carrier="UPS",
        tracking_number="1Z999AA10123456784",
    ),
    "5678": Order(
        id="5678",
        customer_id="guest",
        item="wool throw blanket",
        status="being packed",
        expected_delivery="next Monday",
        total=54.50,
    ),
    "9012": Order(
        id="9012",
        customer_id="guest",
        item="espresso machine",
        status="delivered",
        expected_delivery="last Tuesday",
        total=249.00,
        carrier="FedEx",
        tracking_number="7712 3456 7890",
    ),
}


class FakeOrders:
    """Runs the agent without a backend."""

    def __init__(self, orders: dict[str, Order] | None = None) -> None:
        self._orders = dict(orders if orders is not None else SAMPLE_ORDERS)

    async def find(self, customer_id: str, order_id: str) -> Order | None:
        order = self._orders.get(order_id)

        if order is None or order.customer_id != customer_id:
            return None

        return order

    async def list_for_customer(self, customer_id: str) -> Sequence[Order]:
        return [order for order in self._orders.values() if order.customer_id == customer_id]
'''


DISPUTES_INIT = '''"""The disputes domain: filing a dispute on a delivered order, and refunds."""

from domains.disputes.domain import install

__all__ = ["install"]
'''


DISPUTES_SERVICE = '''"""The disputes backend, behind a port.

`requires_photo` and `parse_reason` are the domain's rules, so they live here
where they can be tested without a server.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable


class DisputeReason(Enum):
    DAMAGED = "damaged"
    NOT_RECEIVED = "not_received"
    WRONG_ITEM = "wrong_item"


# What a customer might say, mapped to the reason it means. The agent passes
# its own wording through; this keeps the tool from being brittle about it.
_REASON_WORDS = {
    "damaged": DisputeReason.DAMAGED,
    "damage": DisputeReason.DAMAGED,
    "broken": DisputeReason.DAMAGED,
    "cracked": DisputeReason.DAMAGED,
    "smashed": DisputeReason.DAMAGED,
    "not_received": DisputeReason.NOT_RECEIVED,
    "not received": DisputeReason.NOT_RECEIVED,
    "missing": DisputeReason.NOT_RECEIVED,
    "never arrived": DisputeReason.NOT_RECEIVED,
    "wrong_item": DisputeReason.WRONG_ITEM,
    "wrong item": DisputeReason.WRONG_ITEM,
    "incorrect item": DisputeReason.WRONG_ITEM,
}


def parse_reason(text: str) -> DisputeReason | None:
    """The reason the customer gave, or None if it isn't one we can file."""
    return _REASON_WORDS.get(text.strip().lower())


def requires_photo(reason: DisputeReason) -> bool:
    """Whether this reason cannot be filed without a photo."""
    return reason is DisputeReason.DAMAGED


@dataclass(frozen=True)
class Dispute:
    id: str
    order_id: str
    customer_id: str
    reason: DisputeReason
    status: str
    refund_amount: float
    photo_url: str | None = None


@runtime_checkable
class DisputesService(Protocol):
    async def file(
        self,
        customer_id: str,
        order_id: str,
        reason: DisputeReason,
        refund_amount: float,
        photo_url: str | None = None,
    ) -> Dispute:
        """Open a dispute and return it."""
        ...

    async def find_for_order(self, customer_id: str, order_id: str) -> Dispute | None:
        """This customer's dispute on that order, if they have one."""
        ...


class DisputesAPI:
    """Your real disputes backend. Implement this against whatever you have.

    Every method raises until you do; development runs use `FakeDisputes`.
    """

    def __init__(self, base_url: str) -> None:
        raise NotImplementedError("Implement DisputesAPI against your disputes backend.")

    async def file(
        self,
        customer_id: str,
        order_id: str,
        reason: DisputeReason,
        refund_amount: float,
        photo_url: str | None = None,
    ) -> Dispute:
        raise NotImplementedError

    async def find_for_order(self, customer_id: str, order_id: str) -> Dispute | None:
        raise NotImplementedError


class FakeDisputes:
    """Runs the agent without a backend. One open dispute per order."""

    def __init__(self) -> None:
        self._disputes: dict[tuple[str, str], Dispute] = {}
        self._next_id = 1001

    async def file(
        self,
        customer_id: str,
        order_id: str,
        reason: DisputeReason,
        refund_amount: float,
        photo_url: str | None = None,
    ) -> Dispute:
        existing = self._disputes.get((customer_id, order_id))

        if existing is not None:
            return existing

        dispute = Dispute(
            id=f"D-{self._next_id}",
            order_id=order_id,
            customer_id=customer_id,
            reason=reason,
            status="under review",
            refund_amount=refund_amount,
            photo_url=photo_url,
        )

        self._next_id += 1
        self._disputes[(customer_id, order_id)] = dispute

        return dispute

    async def find_for_order(self, customer_id: str, order_id: str) -> Dispute | None:
        return self._disputes.get((customer_id, order_id))
'''


DISPUTES_TOOLS = '''"""The domain's tools: read the service, call it, wrap the result.

`file_dispute` also enforces the photo requirement. The procedure asks for a
photo; a prompt can be talked around, so the tool refuses without one.
"""

from typing import Sequence

import parlant.sdk as p
import services
from domains.disputes.service import parse_reason, requires_photo

REFUND_WINDOW = "5-10 business days"


def first_image_url(attachments: Sequence[p.AttachmentRef]) -> str | None:
    """The first image among the attachments, if any.

    Attachments are references to files in your own storage. The model is told
    an image is present; only tools see its URL.
    """
    for attachment in attachments:
        if str(attachment.get("media_type", "")).startswith("image/"):
            return attachment.get("url")

    return None


def photo_from_conversation() -> str | None:
    """The most recent photo the customer attached in this conversation.

    Not just the last message: by the time the customer confirms the filing,
    the photo is a turn or two back.
    """
    for message in reversed(list(p.Session.current.interaction.messages)):
        if message.source is not p.EventSource.CUSTOMER:
            continue

        if url := first_image_url(message.attachments):
            return url

    return None


@p.tool
async def file_dispute(order_id: str, reason: str) -> p.ToolResult:
    """File a dispute on a delivered order.

    Reason is one of: damaged, not received, wrong item.
    """
    parsed = parse_reason(reason)

    if parsed is None:
        return p.ToolResult(
            data=(
                f"'{reason}' is not a reason we can file. Ask the customer "
                "whether the item arrived damaged, never arrived, or was the "
                "wrong item."
            )
        )

    orders = services.orders()
    disputes = services.disputes()
    customer_id = p.Customer.current.id

    order = await orders.find(customer_id, order_id)

    if order is None:
        return p.ToolResult(
            data=(
                f"This customer has no order {order_id}. Ask them to "
                "double-check the number."
            )
        )

    if not order.delivered:
        return p.ToolResult(
            data=(
                f"No dispute was filed. Order {order.id} is {order.status} "
                f"and is expected {order.expected_delivery}. Disputes open "
                "once an order has been delivered."
            )
        )

    photo_url = photo_from_conversation()

    if requires_photo(parsed) and photo_url is None:
        return p.ToolResult(
            data=(
                f"No dispute was filed. A photo of the damage is required "
                f"before a damage claim on order {order.id} can be opened. "
                "Ask the customer to attach a photo, then file the dispute."
            )
        )

    dispute = await disputes.file(
        customer_id=customer_id,
        order_id=order.id,
        reason=parsed,
        refund_amount=order.total,
        photo_url=photo_url,
    )

    return p.ToolResult(
        data=(
            f"Dispute {dispute.id} filed on order {order.id} "
            f"({dispute.reason.value}), status {dispute.status}, refund of "
            f"${dispute.refund_amount:.2f} if approved."
        ),
        canned_response_fields={
            "dispute_id": dispute.id,
            "refund_amount": f"${dispute.refund_amount:.2f}",
            "refund_window": REFUND_WINDOW,
        },
    )


@p.tool
async def check_dispute_eligibility(order_id: str) -> p.ToolResult:
    """Say whether a dispute can be opened on an order, and what it still needs."""
    orders = services.orders()
    disputes = services.disputes()
    customer_id = p.Customer.current.id

    order = await orders.find(customer_id, order_id)

    if order is None:
        return p.ToolResult(
            data=(
                f"This customer has no order {order_id}. Ask them to "
                "double-check the number."
            )
        )

    if not order.delivered:
        return p.ToolResult(
            data=(
                f"Order {order.id} ({order.item}) cannot be disputed yet: it "
                f"is {order.status} and expected {order.expected_delivery}. "
                "Disputes open once an order has been delivered."
            )
        )

    existing = await disputes.find_for_order(customer_id, order_id)

    if existing is not None:
        return p.ToolResult(
            data=(
                f"Order {order.id} already has dispute {existing.id}, which "
                f"is {existing.status}. Do not open another one."
            )
        )

    return p.ToolResult(
        data=(
            f"Order {order.id} ({order.item}, ${order.total:.2f}) was "
            "delivered and can be disputed. A damage claim also needs a photo "
            "of the damage before it can be filed."
        )
    )


@p.tool
async def check_dispute_status(order_id: str) -> p.ToolResult:
    """Look up the dispute on one of this customer's orders."""
    disputes = services.disputes()

    dispute = await disputes.find_for_order(p.Customer.current.id, order_id)

    if dispute is None:
        return p.ToolResult(data=f"There is no dispute on order {order_id}.")

    return p.ToolResult(
        data=(
            f"Dispute {dispute.id} on order {dispute.order_id} is "
            f"{dispute.status}, for a refund of ${dispute.refund_amount:.2f}."
        ),
        canned_response_fields={
            "dispute_id": dispute.id,
            "refund_amount": f"${dispute.refund_amount:.2f}",
            "refund_window": REFUND_WINDOW,
        },
    )
'''


DISPUTES_DOMAIN = '''"""Everything this domain declares on the agent.

Each authoring form appears once here: a policy document, a standard
operating procedure, steering guidelines, an observation that raises effort,
canned responses, and a glossary term.
"""

import parlant.sdk as p

from domains.disputes import tools


async def install(agent: p.Agent) -> None:
    # A glossary term: vocabulary the agent should read the way you mean it.
    await agent.create_term(
        name="dispute",
        description=(
            "A claim opened on a delivered order when something went wrong "
            "with it: damaged on arrival, never received, or the wrong item. "
            "An approved dispute refunds the order total."
        ),
        synonyms=["claim", "case"],
    )

    dispute_filed = await agent.create_canned_response(
        template=(
            "I've opened dispute {{ dispute_id }} for you. If it's approved, "
            "{{ refund_amount }} goes back to your original payment method "
            "within {{ refund_window }}."
        ),
        # Only offered on a turn where a dispute tool staged these values.
        field_dependencies=["dispute_id", "refund_amount", "refund_window"],
        note=(
            "Always state the refund window. Never promise the refund itself "
            "or an exact date — the dispute still has to be approved."
        ),
    )

    photo_request = await agent.create_canned_response(
        template=(
            "I'm sorry it arrived damaged. Could you attach a photo showing "
            "the damage? I need it to open the claim."
        ),
        signals=[
            "the agent needs a picture of a damaged item",
            "asking the customer to upload evidence",
        ],
    )

    # A policy document: the rules of the domain in one complete section.
    await agent.create_policy(
        title="Disputes and refunds",
        content="""\\
Disputes cover a delivered order that arrived damaged, never arrived, or was
the wrong item.

- A dispute can only be opened on an order that has been delivered. If the
  order has not been delivered yet, say where it is and offer to help once it
  arrives.
- A damage claim requires a photo of the damage. This is not optional: the
  claim cannot be opened without one.
- One dispute per order. If a dispute already exists, tell the customer its
  current status instead of opening another.
- An approved dispute refunds the order total to the original payment method
  within 5-10 business days. Say the window; never promise an exact date, and
  never state that the refund has been approved while it is under review.
""",
        tools=[
            tools.check_dispute_eligibility,
            tools.file_dispute,
            tools.check_dispute_status,
        ],
        canned_responses=[dispute_filed],
    )

    # A standard operating procedure: a flow whose steps the agent tracks.
    await agent.create_policy(
        title="Filing a damage dispute",
        content="""\\
When the customer says an item arrived damaged, broken, or cracked, follow
this procedure step by step, tracking which step the conversation is at:

1. Acknowledge the problem before anything else.
2. If they haven't said which order, ask for the order number.
3. Check with `check_dispute_eligibility` whether the order can be disputed
   at all. If it cannot — it hasn't been delivered, or it already has a
   dispute — tell them what it says and stop here. Do not ask for a photo for
   an order that cannot be disputed.
4. Ask the customer to attach a photo of the damage. This step is required:
   do not skip it, and do not promise that the claim is open before you have
   the photo.
5. Once you have the photo, summarize what you are about to file — the order
   and the refund amount — and get the customer's explicit confirmation.
   Opening a dispute is not something they can undo themselves.
6. After they confirm, file the dispute with `file_dispute` using the reason
   "damaged".
7. Tell them the dispute number and the refund window.
""",
        tools=[tools.check_dispute_eligibility, tools.file_dispute],
        canned_responses=[photo_request, dispute_filed],
    )

    # An observation: like a policy, except it never instructs the agent. It
    # scopes the rules that depend on it to the right turns, and changes how
    # those turns run — here, raising the effort level while it holds.
    upset = await agent.create_observation(
        condition=(
            "the customer is requesting compensation or threatening legal action"
        ),
        trigger_effort=p.Effort.HIGH,
    )

    # Steering guidelines: how the agent talks, in force alongside whatever
    # procedure is running.
    acknowledge_first = await agent.create_policy(
        "Acknowledge how the customer feels before proposing any solution",
    )

    # Considered only on turns where the observation holds.
    await acknowledge_first.depend_on(upset)

    await agent.create_policy(
        "Never tell a customer that a refund has been approved or issued "
        "while their dispute is still under review",
        weight=p.Weight.HIGH,
    )
'''


TEST_ORDERS_DOMAIN = '''"""Tests for the orders domain.

A placeholder: all that can be said about OrdersAPI is that it still raises.
Replace it with tests for your adapter and your domain rules once you write
them. Nothing here calls a model.
"""

import pytest

from domains.orders.service import OrdersAPI


def test_that_the_orders_backend_is_not_implemented_yet() -> None:
    with pytest.raises(NotImplementedError):
        OrdersAPI("https://api.example.com")
'''


TEST_DISPUTES_DOMAIN = '''"""Tests for the disputes domain.

A placeholder: all that can be said about DisputesAPI is that it still raises.
Replace it with tests for your adapter and your domain rules once you write
them. Nothing here calls a model.
"""

import pytest

from domains.disputes.service import DisputesAPI


def test_that_the_disputes_backend_is_not_implemented_yet() -> None:
    with pytest.raises(NotImplementedError):
        DisputesAPI("https://api.example.com")
'''


# --- lead-gen ---------------------------------------------------------------


LEAD_GEN_MAIN_PY = '''"""The entry point: build the server, create the agent, install the domains."""

import asyncio

from config import can_measure_spend, conversation_cap, create_server
from domains import handoff, leadgen


async def main() -> None:
    async with create_server() as server:
        agent = await server.create_agent(
            id="website",
            name="Hearth & Drape",
            instructions=(
                "You are the assistant on Hearth & Drape's website. Hearth & Drape makes "
                "made-to-measure curtains and blinds for homeowners and for trade clients such as "
                "interior designers, hotels and property developers. Help visitors understand "
                "whether we fit what they need, and when we do, help them take the next step "
                "with our team."
            ),
        )

        # The domain installs what it owns: its policies, its tools, its
        # canned responses, and the catalog the agent answers from.
        await leadgen.install(agent)

        # Hands a conversation to your team once it costs more than the cap
        # ($1.00 unless CONVERSATION_CAP_USD says otherwise). Installed only
        # when every model call reports its cost; see config.py.
        if can_measure_spend():
            await handoff.install_conversation_cap(server, agent, cap=conversation_cap())


if __name__ == "__main__":
    asyncio.run(main())
'''


LEAD_GEN_CONFIG_PY = '''"""Server settings, and the only place that branches on the environment.

`PARLANT_ENV` selects it: development (the default), test, or production.
"""

import os
from decimal import Decimal

import parlant.sdk as p
import services
from dotenv import load_dotenv
from domains.leadgen.service import (
    CalendarAPI,
    CrmAPI,
    FakeCalendar,
    FakeCrm,
    FakeMailer,
    FakeSalesInbox,
    MailerAPI,
    SalesInboxAPI,
    StaticCatalog,
)

MODELS = {"*": "openrouter"}

# Providers that report what each call cost. A session budget prices spend
# from those reports, so it needs no price table for them.
_COST_REPORTING_PROVIDERS = {"openrouter"}


def create_server() -> p.Server:
    """Build the server for the current environment."""
    # Reads .env, if there is one. Real environment variables win.
    load_dotenv()

    services.use(_services_for(p.resolve_environment()))

    return p.Server(
        models=MODELS,
        log_level=p.LogLevel.DEBUG,
    )


def can_measure_spend() -> bool:
    """Whether a session budget can price every model call this server makes.

    A budget that meets a call it cannot price stops the turn rather than
    under-count, so main.py installs the conversation cap only when this is
    true. Routing a model to another provider? Give the server its prices
    with `server.set_pricing(...)`, then return True here.
    """
    return all(route.split(":")[0] in _COST_REPORTING_PROVIDERS for route in MODELS.values())


def conversation_cap() -> Decimal:
    """What one conversation may cost before it goes to your team, in USD."""
    return Decimal(os.environ.get("CONVERSATION_CAP_USD", "1.00"))


def _services_for(environment: p.Environment) -> services.Services:
    """The implementations this environment runs against.

    This example uses the in-memory fakes outside production so it runs with
    no CRM, calendar, email provider or team inbox. A real project usually points
    development at sandbox accounts of those services instead.
    """
    if environment is p.Environment.PRODUCTION:
        return services.Services(
            crm=CrmAPI(os.environ["CRM_API_URL"]),
            calendar=CalendarAPI(os.environ["CALENDAR_API_URL"]),
            mailer=MailerAPI(os.environ["EMAIL_API_URL"]),
            sales_inbox=SalesInboxAPI(os.environ["SALES_INBOX_URL"]),
            catalog=StaticCatalog(),
        )

    return services.Services(
        crm=FakeCrm(),
        calendar=FakeCalendar(),
        mailer=FakeMailer(),
        sales_inbox=FakeSalesInbox(),
        catalog=StaticCatalog(),
    )
'''


LEAD_GEN_SERVICES_PY = '''"""Your services: tools read them from here, and config.py fills them in.

A test swaps them with one `use()` call.
"""

from dataclasses import dataclass

from domains.leadgen.service import (
    CalendarService,
    CatalogService,
    CrmService,
    MailerService,
    SalesInboxService,
)


@dataclass(frozen=True)
class Services:
    crm: CrmService
    calendar: CalendarService
    mailer: MailerService
    sales_inbox: SalesInboxService
    catalog: CatalogService


_installed: Services | None = None


def use(services: Services) -> None:
    """Install the services this process runs with. Called from config.py."""
    global _installed
    _installed = services


def installed() -> Services:
    if _installed is None:
        raise RuntimeError(
            "No services are installed. create_server() in config.py does this; "
            "a test should call services.use(...) with its own implementations."
        )

    return _installed


def crm() -> CrmService:
    return installed().crm


def calendar() -> CalendarService:
    return installed().calendar


def mailer() -> MailerService:
    return installed().mailer


def sales_inbox() -> SalesInboxService:
    return installed().sales_inbox


def catalog() -> CatalogService:
    return installed().catalog
'''


LEAD_GEN_DOMAINS_INIT = '''"""Each domain owns one capability: its policies, its tools, and the service
they call. Add a domain by adding a package here and installing it in main.py.

    leadgen/    qualifying website visitors, scoring them, and booking meetings
    handoff/    handing a conversation to your team when it reaches its cap
"""
'''


LEADGEN_INIT = '''"""The lead-gen domain: qualify a website visitor, score them, and book a meeting."""

from domains.leadgen.domain import install

__all__ = ["install"]
'''


LEADGEN_SERVICE = '''"""The lead-gen backends, behind ports: a CRM, a calendar, a mailer, the
sales inbox, and the product catalog.

`score_lead` and `band_for` are the domain's rules, so they live here where
they can be tested without a server. Which adapters run is decided in
config.py; the fakes keep everything in memory.
"""

import enum
import itertools
import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, Literal, Protocol, runtime_checkable
from zoneinfo import ZoneInfo

# --- Scoring --------------------------------------------------------------


class BuyingFor(enum.Enum):
    OWN_HOME = "own home"
    BUSINESS = "business or client"


Band = Literal["hot", "warm", "cold"]


def score_lead(
    buying_for: BuyingFor,
    role: str,
    window_count: int | None,
    timeline_days: int | None,
    named_competitor: bool,
) -> int:
    """Score a lead in code, so the same answers always get the same score."""
    score = 0

    if buying_for is BuyingFor.BUSINESS:  # trade buyers are where the volume is
        score += 25

    if window_count is not None:  # an unknown count scores nothing either way
        if window_count >= 50:
            score += 30
        elif window_count >= 15:  # a whole house, or a small trade job
            score += 25
        elif window_count >= 6:
            score += 10
        elif window_count <= 2:
            score -= 10

    if timeline_days is not None and timeline_days <= 90:
        score += 15

    if named_competitor:
        score += 10

    if any(term in role.lower() for term in ("student", "intern")):
        score -= 20

    return score


def band_for(score: int) -> Band:
    return "hot" if score >= 45 else "warm" if score >= 20 else "cold"


# --- CRM ------------------------------------------------------------------


@dataclass
class Lead:
    id: str
    email: str | None
    phone: str | None
    fields: dict[str, Any]
    notes: str
    score: int
    source: dict[str, Any]
    chat_session_id: str
    stage: str = "New"
    do_not_contact: bool = False
    is_customer: bool = False
    owner_name: str | None = None


@runtime_checkable
class CrmService(Protocol):
    async def upsert_lead(
        self,
        *,
        email: str | None,
        phone: str | None,
        fields: dict[str, Any],
        notes: str,
        score: int,
        source: dict[str, Any],
        chat_session_id: str,
    ) -> Lead:
        """Create the lead, or update the one with this email or phone number."""
        ...

    async def lead_brief(self, lead_id: str | None) -> str:
        """What the rep should read before the meeting."""
        ...

    async def set_stage(self, lead_id: str | None, stage: str) -> None: ...

    async def is_do_not_contact(self, email: str) -> bool:
        """Whether this address has asked not to be emailed."""
        ...


class CrmAPI:
    """Your real CRM. Implement this against whatever you use.

    Every method raises until you do; development runs use `FakeCrm`.
    """

    def __init__(self, base_url: str) -> None:
        raise NotImplementedError("Implement CrmAPI against your CRM.")

    async def upsert_lead(
        self,
        *,
        email: str | None,
        phone: str | None,
        fields: dict[str, Any],
        notes: str,
        score: int,
        source: dict[str, Any],
        chat_session_id: str,
    ) -> Lead:
        raise NotImplementedError

    async def lead_brief(self, lead_id: str | None) -> str:
        raise NotImplementedError

    async def set_stage(self, lead_id: str | None, stage: str) -> None:
        raise NotImplementedError

    async def is_do_not_contact(self, email: str) -> bool:
        raise NotImplementedError


# Sample data for the fake: one existing customer, and one address that has
# asked not to be contacted.
KNOWN_CUSTOMERS = {"maria.lopez@lopezinteriors.com": "Priya Shah"}
DO_NOT_CONTACT = {"unsubscribe@example.com"}


class FakeCrm:
    """An in-memory stand-in. Replace it with your CRM (see `CrmAPI`)."""

    def __init__(self) -> None:
        self._leads: dict[str, Lead] = {}
        self._by_contact: dict[str, str] = {}
        self._ids = itertools.count(1)

    async def upsert_lead(
        self,
        *,
        email: str | None,
        phone: str | None,
        fields: dict[str, Any],
        notes: str,
        score: int,
        source: dict[str, Any],
        chat_session_id: str,
    ) -> Lead:
        email = email.strip().lower() if email else None
        phone_digits = "".join(ch for ch in phone if ch.isdigit()) if phone else None
        keys = [key for key in (email, phone_digits) if key]
        known = next((self._by_contact[key] for key in keys if key in self._by_contact), None)

        if known is not None:
            lead = self._leads[known]
            lead.email = email or lead.email
            lead.phone = phone or lead.phone
            lead.fields.update(fields)
            lead.notes = notes or lead.notes
            lead.score = score
            lead.source = source
            lead.chat_session_id = chat_session_id
            self._by_contact.update({key: lead.id for key in keys})
            return lead

        lead = Lead(
            id=f"lead-{next(self._ids)}",
            email=email,
            phone=phone,
            fields=dict(fields),
            notes=notes,
            score=score,
            source=source,
            chat_session_id=chat_session_id,
            do_not_contact=email in DO_NOT_CONTACT,
            is_customer=email in KNOWN_CUSTOMERS,
            owner_name=KNOWN_CUSTOMERS.get(email or ""),
        )

        self._leads[lead.id] = lead
        self._by_contact.update({key: lead.id for key in keys})

        return lead

    async def is_do_not_contact(self, email: str) -> bool:
        return email in DO_NOT_CONTACT

    async def lead_brief(self, lead_id: str | None) -> str:
        lead = self._leads.get(lead_id or "")

        if lead is None:
            return "No lead on file."

        return json.dumps(
            {
                "email": lead.email,
                "phone": lead.phone,
                **lead.fields,
                "score": lead.score,
                "source": lead.source,
                "notes": lead.notes,
            },
            default=str,
        )

    async def set_stage(self, lead_id: str | None, stage: str) -> None:
        if lead := self._leads.get(lead_id or ""):
            lead.stage = stage


# --- Calendar -------------------------------------------------------------

# The team's own clock. Slots are stored in UTC and shown to each visitor in
# their own timezone; the team's hours are explained in this one.
BUSINESS_TZ = ZoneInfo("Europe/London")

# When a rep can start a 30-minute call, on the team's wall clock.
BUSINESS_HOURS = ((9, 30), (11, 0), (14, 0), (16, 30))


@dataclass(frozen=True)
class Slot:
    start: datetime  # timezone-aware, UTC
    end: datetime

    @property
    def id(self) -> str:
        return self.start.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class Meeting:
    id: str
    rep_name: str
    start: datetime  # timezone-aware, UTC
    end: datetime
    description: str


@runtime_checkable
class CalendarService(Protocol):
    async def open_slots(
        self, *, lead_id: str | None, start: datetime, end: datetime
    ) -> list[Slot]:
        """Open slots with this lead's rep that start in [start, end)."""
        ...

    async def try_book(
        self, *, lead_id: str | None, start: datetime, description: str
    ) -> Meeting | None:
        """Book the open slot that starts at `start`, or return None if it is no longer open."""
        ...


class CalendarAPI:
    """Your real calendar. Implement this against whatever you use.

    Every method raises until you do; development runs use `FakeCalendar`.
    """

    def __init__(self, base_url: str) -> None:
        raise NotImplementedError("Implement CalendarAPI against your calendar.")

    async def open_slots(
        self, *, lead_id: str | None, start: datetime, end: datetime
    ) -> list[Slot]:
        raise NotImplementedError

    async def try_book(
        self, *, lead_id: str | None, start: datetime, description: str
    ) -> Meeting | None:
        raise NotImplementedError


REPS = ["Tom Becker", "Aisha Reed", "Priya Shah", "Dan Okoro"]


def _sometimes_taken(local: datetime) -> bool:
    return (local.day + local.hour) % 5 == 0


class FakeCalendar:
    """An in-memory stand-in. Replace it with your calendar (see `CalendarAPI`).

    One calendar per rep, open on weekdays at `BUSINESS_HOURS` (London time)
    from tomorrow for `horizon_days`, with a few slots already taken. Pass
    `taken` to decide which, and `today` to pin the date in a test.
    """

    def __init__(
        self,
        *,
        today: date | None = None,
        horizon_days: int = 28,
        taken: Callable[[datetime], bool] = _sometimes_taken,
    ) -> None:
        self._today = today
        self._horizon = horizon_days
        self._taken = taken
        self._booked: dict[str, set[str]] = {rep: set() for rep in REPS}
        self._ids = itertools.count(1)
        self._rep_cycle = itertools.cycle(REPS)
        self._rep_for_lead: dict[str, str] = {}

    def _rep(self, lead_id: str | None) -> str:
        """Round-robin: each lead gets the next rep, and keeps them."""
        key = lead_id or "anonymous"

        if key not in self._rep_for_lead:
            self._rep_for_lead[key] = next(self._rep_cycle)

        return self._rep_for_lead[key]

    def _open_slots(self, rep: str) -> list[Slot]:
        today = self._today or datetime.now(BUSINESS_TZ).date()
        slots = []

        for offset in range(1, self._horizon + 1):
            day = today + timedelta(days=offset)

            if day.weekday() >= 5:
                continue

            for hour, minute in BUSINESS_HOURS:
                local = datetime(day.year, day.month, day.day, hour, minute, tzinfo=BUSINESS_TZ)

                if self._taken(local):
                    continue

                start = local.astimezone(timezone.utc)
                slot = Slot(start=start, end=start + timedelta(minutes=30))

                if slot.id not in self._booked[rep]:
                    slots.append(slot)

        return slots

    async def open_slots(
        self, *, lead_id: str | None, start: datetime, end: datetime
    ) -> list[Slot]:
        return [s for s in self._open_slots(self._rep(lead_id)) if start <= s.start < end]

    async def try_book(
        self, *, lead_id: str | None, start: datetime, description: str
    ) -> Meeting | None:
        rep = self._rep(lead_id)
        slot = next((s for s in self._open_slots(rep) if s.start == start), None)

        if slot is None:
            return None

        self._booked[rep].add(slot.id)

        return Meeting(
            id=f"mtg-{next(self._ids)}",
            rep_name=rep,
            start=slot.start,
            end=slot.end,
            description=description,
        )


# --- Mailer -------------------------------------------------------------


@runtime_checkable
class MailerService(Protocol):
    async def send(self, *, email: str, kind: str, session_id: str) -> None:
        """Send what was promised: an estimate, a price guide, a lookbook."""
        ...


class MailerAPI:
    """Your real email provider. Implement this against whatever you use.

    Every method raises until you do; development runs use `FakeMailer`.
    """

    def __init__(self, base_url: str) -> None:
        raise NotImplementedError("Implement MailerAPI against your email provider.")

    async def send(self, *, email: str, kind: str, session_id: str) -> None:
        raise NotImplementedError


class FakeMailer:
    """An in-memory stand-in. Replace it with your email provider (see
    `MailerAPI`). It sends nothing, and keeps what it would have sent."""

    def __init__(self) -> None:
        self.sent: list[dict[str, str]] = []

    async def send(self, *, email: str, kind: str, session_id: str) -> None:
        self.sent.append({"email": email, "kind": kind, "session_id": session_id})


# --- Sales inbox ----------------------------------------------------------


@runtime_checkable
class SalesInboxService(Protocol):
    async def notify(self, *, session_id: str, reason: str, details: dict[str, Any]) -> None:
        """Tell the sales team a conversation needs a person, and what they need to pick it up."""
        ...


class SalesInboxAPI:
    """Your team's queue: a ticketing system, a Slack channel, a pager.

    Every method raises until you implement it; development runs use
    `FakeSalesInbox`.
    """

    def __init__(self, base_url: str) -> None:
        raise NotImplementedError("Implement SalesInboxAPI against your team's queue.")

    async def notify(self, *, session_id: str, reason: str, details: dict[str, Any]) -> None:
        raise NotImplementedError


class FakeSalesInbox:
    """An in-memory stand-in. Replace it with your team's queue (see
    `SalesInboxAPI`). It keeps what it was told."""

    def __init__(self) -> None:
        self.waiting: list[dict[str, Any]] = []

    async def notify(self, *, session_id: str, reason: str, details: dict[str, Any]) -> None:
        self.waiting.append({"session_id": session_id, "reason": reason, "details": details})


# --- Catalog --------------------------------------------------------------


@runtime_checkable
class CatalogService(Protocol):
    def facts(self) -> dict[str, Any]:
        """What the agent may say about the products, prices and services."""
        ...


class StaticCatalog:
    """The catalog as a constant. Replace it with your product data if it
    lives somewhere else."""

    def facts(self) -> dict[str, Any]:
        return CATALOG


CATALOG: dict[str, Any] = {
    "company": (
        "Hearth & Drape: made-to-measure curtains and blinds, designed and sewn in our "
        "Leeds workroom. We serve homeowners and trade clients (interior designers, hotels, "
        "property developers, landlords)."
    ),
    "products": {
        "made-to-measure curtains": (
            "Pencil pleat, double pleat, wave and eyelet headings. Over 400 fabrics: linens, "
            "velvets, cottons, sheers, wools. From $180 per window (small window, standard lining)."
        ),
        "blackout curtains": (
            "Any curtain can take our blackout lining (blocks ~99% of light). "
            "Blackout lining adds from $40 per window."
        ),
        "thermal lining": "Thermal interlining for older, draughty homes, adds from $35 per window.",
        "roller blinds": "From $95 per window. Blackout, dim-out and sheer fabrics.",
        "roman blinds": "From $160 per window, in any of our curtain fabrics.",
        "motorized blinds and curtain tracks": (
            "Somfy motors, battery or hardwired, with app, remote, and Alexa/Google Home "
            "control. Motorization adds from $290 per window."
        ),
        "hotel and contract fabrics": (
            "Flame-retardant (FR) certified fabrics to BS 5867 and NFPA 701 for hotels, "
            "hospitality and rental properties."
        ),
    },
    "prices_note": (
        "Only 'from' prices are published. Exact prices depend on size, fabric and fitting, "
        "and are given in a written quote after measuring. Trade pricing, volume pricing and "
        "discounts are set by the trade team only, never in chat."
    ),
    "lead_times": {
        "curtains and roman blinds": "3 to 4 weeks from measuring",
        "roller blinds": "2 weeks",
        "motorized": "4 to 5 weeks",
        "trade projects (over 50 windows)": (
            "scheduled per project; the trade team plans phased installs"
        ),
    },
    "services": {
        "free measure-and-quote visit": (
            "Within 40 miles of Leeds and Manchester. A fitter measures, brings fabric "
            "samples, and a written quote follows within 48 hours."
        ),
        "fitting": "Professional fitting, from $45 per window.",
        "samples": "Up to 8 free fabric samples posted to you.",
        "trade programme": (
            "Dedicated project manager, FR-certified fabrics, phased installs, trade accounts."
        ),
    },
    "not_offered": "We don't sell shutters, awnings, carpets or wallpaper.",
    "links": {
        "support": "help@hearthanddrape.example (orders, deliveries, installations, repairs)",
        "careers": "hearthanddrape.example/careers",
        "partners": "partners@hearthanddrape.example (suppliers and vendors)",
        "pricing page": "hearthanddrape.example/pricing",
        "samples": "hearthanddrape.example/samples",
        "measuring guide": "hearthanddrape.example/measure",
    },
}
'''


LEADGEN_TOOLS = '''"""The domain's tools: read the service, call it, wrap the result.

Each tool also checks what the model passes it. A tool parameter's
description is a hint to the model, so the facts that matter (what the
visitor typed, which slot was offered, whether they may be contacted) are
checked here, in code.
"""

import re
from typing import Annotated, Any

import parlant.sdk as p
import services
from domains.leadgen.pages import visitor_page
from domains.leadgen.scheduling import Scheduler
from domains.leadgen.service import BuyingFor, band_for, score_lead

EMAIL = re.compile(r"[\\w.+-]+@[\\w-]+(?:\\.[\\w-]+)+")

# Offer only times in this window of the visitor's own day (None offers every
# time). A 9:30 call in London is 1:30 AM in Los Angeles.
WAKING_HOURS: tuple[int, int] | None = (7, 21)


def visitor_text() -> str:
    """Everything the visitor has typed in this conversation."""
    return " ".join(
        m.content
        for m in p.Session.current.interaction.messages
        if m.source == p.EventSource.CUSTOMER
    ).lower()


def typed_email(email: str, typed: str) -> str | None:
    """The email, if it is a complete address the visitor typed."""
    email = email.strip().rstrip(".").lower()
    return email if EMAIL.fullmatch(email) and email in typed else None


def typed_phone(phone: str, typed: str) -> str | None:
    """The phone number, if the visitor typed those digits."""
    digits = re.sub(r"\\D", "", phone)
    return phone if len(digits) >= 7 and digits in re.sub(r"\\D", "", typed) else None


@p.tool
async def record_lead(
    goal: str,
    buying_for: Annotated[
        BuyingFor,
        p.ToolParameterOptions(
            description="'business or client' for designers, hotels, landlords, developers, offices; "
            "'own home' for homeowners and renters",
        ),
    ],
    role: str,
    email: Annotated[
        str,
        p.ToolParameterOptions(
            description="The visitor's email address, exactly as they typed it. Empty if they "
            "gave a phone number instead.",
            source="customer",
        ),
    ] = "",
    phone: Annotated[
        str,
        p.ToolParameterOptions(
            description="The visitor's phone number, exactly as typed, if they prefer a call",
            source="customer",
        ),
    ] = "",
    window_count: Annotated[
        int | None,
        p.ToolParameterOptions(
            description="Roughly how many windows, only if the visitor said. For hotels or flats, "
            "use the number of rooms if they only gave rooms. Never guess.",
        ),
    ] = None,
    timeline_days: Annotated[
        int | None,
        p.ToolParameterOptions(
            description="Days until they want everything fitted, if they said",
        ),
    ] = None,
    notes_for_rep: str = "",
    named_competitor: bool = False,
) -> p.ToolResult:
    """Save what the visitor told you as a lead, scored in code."""
    session = p.Session.current

    # The model may fill in an address the visitor never typed, so check the
    # contact details against what they actually wrote. A phone number works
    # in place of an email.
    typed = visitor_text()
    email_ok = typed_email(email, typed)
    phone_ok = typed_phone(phone, typed)

    if not (email_ok or phone_ok):
        return p.ToolResult(
            data={
                "recorded": False,
                "reason": "The visitor hasn't typed an email address or phone number in this chat. "
                "Don't guess one: offer the estimate or price guide in exchange for an "
                "email (or take a phone number if they prefer), or carry on helping.",
            }
        )

    score = score_lead(buying_for, role, window_count, timeline_days, named_competitor)
    band = band_for(score)
    page = visitor_page()  # where they landed, from your site's page_view events

    lead = await services.crm().upsert_lead(
        email=email_ok,
        phone=phone_ok,
        fields={
            "goal": goal,
            "buying_for": buying_for.value,
            "role": role,
            "window_count": window_count,
            "timeline_days": timeline_days,
        },
        notes=notes_for_rep,
        score=score,
        source={"landing_page": page.landing_page, "campaign": page.utm_campaign},
        chat_session_id=session.id,
    )

    # A do-not-contact flag holds for the rest of the conversation, whatever
    # address the visitor types next.
    if lead.do_not_contact or await session.metadata.get("do_not_contact"):
        await session.metadata.set("do_not_contact", True)
        band = "cold"

    await session.metadata.set("lead_id", lead.id)
    await session.metadata.set("lead_band", band)  # read by the band policies' evaluation_func
    await session.metadata.set("lead_email", email_ok)
    await session.metadata.set("lead_phone", phone_ok)

    # Re-scoring replaces the band label rather than adding a second one.
    for other in ("hot", "warm", "cold"):
        if other != band and f"lead-{other}" in session.labels:
            await session.labels.remove(f"lead-{other}")
    await session.labels.add(f"lead-{band}")

    data = {"recorded": True, "band": band, "existing_customer": lead.is_customer}

    # A missing answer scores nothing, so a lead recorded early can look
    # colder than it is. Say what is still open, so the agent asks and re-scores.
    still_unknown = [
        name
        for name, value in (("window_count", window_count), ("timeline_days", timeline_days))
        if value is None
    ]
    if still_unknown:
        data["still_unknown"] = still_unknown
        data["note"] = (
            "If the visitor already told you any of these, call record_lead again now with "
            "them. Otherwise ask about one naturally, and call record_lead again with the "
            "answer to update the lead."
        )

    return p.ToolResult(data=data)


@p.tool
async def send_follow_up(
    email: Annotated[
        str,
        p.ToolParameterOptions(
            description="The email address the visitor gave",
            source="customer",
        ),
    ],
    what: Annotated[
        str,
        p.ToolParameterOptions(
            description="What was promised: 'estimate', 'trade price guide' or 'lookbook'",
        ),
    ],
) -> p.ToolResult:
    """Email the visitor what you promised them."""
    if await p.Session.current.metadata.get("do_not_contact"):
        return p.ToolResult(
            data={
                "sent": False,
                "reason": "This visitor asked not to be emailed. Point them to the pricing page instead.",
            }
        )

    # Send only to an address the visitor typed, never one the model completed.
    email = email.strip().rstrip(".").lower()
    if not EMAIL.fullmatch(email) or email not in visitor_text():
        return p.ToolResult(
            data={
                "sent": False,
                "reason": "That isn't a complete address the visitor typed. Ask them to check it.",
            }
        )

    # Check the address itself too: this can run before record_lead has
    # flagged the session.
    if await services.crm().is_do_not_contact(email):
        await p.Session.current.metadata.set("do_not_contact", True)
        return p.ToolResult(
            data={
                "sent": False,
                "reason": "This visitor asked not to be emailed. Point them to the pricing page instead.",
            }
        )

    await services.mailer().send(email=email, kind=what, session_id=p.Session.current.id)

    return p.ToolResult(data={"sent": True, "to": email, "what": what})


# --- Scheduling -------------------------------------------------------------
#
# The rules live in scheduling.py. These tools read the session into a state
# dict, call the scheduler, and write back what it offered, so the next step
# (and book_meeting) can check the visitor's pick against it.

_STATE_KEYS = (
    "lead_id",
    "lead_band",
    "do_not_contact",
    "offered_dates",
    "offered_times",
    "schedule_preference",
)


async def _state() -> dict[str, Any]:
    metadata = p.Session.current.metadata
    page = visitor_page()  # the browser's timezone and language, from the page_view events

    return {
        **{key: await metadata.get(key) for key in _STATE_KEYS},
        "timezone": page.timezone,
        "locale": page.locale,
    }


async def _save(updates: dict[str, Any]) -> None:
    for key, value in updates.items():
        await p.Session.current.metadata.set(key, value)


def _scheduler() -> Scheduler:
    return Scheduler(services.calendar(), waking_hours=WAKING_HOURS)


VisitorAsked = Annotated[
    bool,
    p.ToolParameterOptions(
        description="True only if the visitor themselves asked to talk to someone, for a call, "
        "or for a visit, in this conversation",
    ),
]


@p.tool
async def find_meeting_dates(
    start_date: Annotated[
        str,
        p.ToolParameterOptions(
            description="First day to look at, as YYYY-MM-DD, worked out from what the visitor "
            "said ('next week', 'from the 10th'). Empty means from tomorrow.",
        ),
    ] = "",
    end_date: Annotated[
        str,
        p.ToolParameterOptions(
            description="Last day to look at, as YYYY-MM-DD. Empty means two weeks after start_date.",
        ),
    ] = "",
    preference: Annotated[
        str,
        p.ToolParameterOptions(
            description="Time of day or weekdays the visitor asked for, in English: 'morning', "
            "'afternoon', 'thursday', 'thursday afternoon'. Empty if none.",
        ),
    ] = "",
    visitor_asked: VisitorAsked = False,
) -> p.ToolResult:
    """The days with open times with this lead's rep, offered as choices."""
    result = await _scheduler().dates(
        await _state(),
        start_date=start_date,
        end_date=end_date,
        preference=preference,
        visitor_asked=visitor_asked,
    )
    await _save(result.state)

    return p.ToolResult(data=result.data, metadata=result.metadata or None)


@p.tool
async def find_meeting_times(
    date: Annotated[
        str,
        p.ToolParameterOptions(
            description="The day the visitor picked: one of the offered days, as offered, or YYYY-MM-DD",
        ),
    ],
    preference: str = "",
    visitor_asked: VisitorAsked = False,
) -> p.ToolResult:
    """The open times on the day the visitor picked, offered as choices."""
    result = await _scheduler().times(
        await _state(), date_text=date, preference=preference, visitor_asked=visitor_asked
    )
    await _save(result.state)

    return p.ToolResult(data=result.data, metadata=result.metadata or None)


@p.tool(consequential=True, narration="Booking your call")
async def book_meeting(
    slot: Annotated[
        str,
        p.ToolParameterOptions(
            description="The time the visitor chose, exactly as offered (or as they wrote it)",
            source="customer",
        ),
    ],
) -> p.ToolResult:
    """Book the offered time the visitor picked. Picking one of the offered times (clicking or
    typing it) is the visitor's confirmation, so book it without asking them to confirm again."""
    session = p.Session.current
    state = await _state()

    if state["lead_id"] is None:
        return p.ToolResult(
            data={"booked": False, "reason": "Record the lead with record_lead first."}
        )

    brief = await services.crm().lead_brief(state["lead_id"])  # their answers, score, and notes

    # The visitor's pick is matched against the times offered, leniently, so
    # "4:30pm" books the 16:30 slot. A taken slot comes back with booked: false
    # and the remaining times, rather than an error.
    result = await _scheduler().book(state, slot_text=slot, description=brief)
    await _save(result.state)

    if result.booked is None:
        return p.ToolResult(data=result.data, metadata=result.metadata or None)

    await services.crm().set_stage(state["lead_id"], "Meeting booked")
    await session.labels.add("meeting-booked")

    email = await session.metadata.get("lead_email")
    phone = await session.metadata.get("lead_phone")

    return p.ToolResult(
        data={
            **result.data,
            "invite": "emailed" if email else f"no email on file; the rep will call {phone}",
        },
        canned_response_fields={
            **result.canned_response_fields,
            "next_step": "The invite is on its way to your inbox."
            if email
            else f"{result.booked.rep_name.split()[0]} will call you on {phone}.",
        },
    )


@p.tool
async def request_team_follow_up(
    note: Annotated[
        str,
        p.ToolParameterOptions(
            description="Why the team should get in touch, and anything that helps them, such as "
            "the times that would suit the visitor",
        ),
    ],
) -> p.ToolResult:
    """Ask the sales team to contact the visitor themselves, when none of the offered times work."""
    session = p.Session.current
    lead_id = await session.metadata.get("lead_id")

    if lead_id is None:
        return p.ToolResult(
            data={"requested": False, "reason": "Record the lead with record_lead first."}
        )

    if await session.metadata.get("do_not_contact"):
        return p.ToolResult(
            data={"requested": False, "reason": "This visitor asked not to be contacted."}
        )

    contact = await session.metadata.get("lead_email") or await session.metadata.get("lead_phone")

    await services.sales_inbox().notify(
        session_id=session.id,
        reason="Visitor asked to be contacted",
        details={"lead_id": lead_id, "contact": contact, "note": note},
    )
    await session.labels.add("follow-up-requested")

    return p.ToolResult(
        data={
            "requested": True,
            "contact": contact,
            "tell_visitor": "Someone from the team will contact them to find a time.",
        }
    )
'''


LEADGEN_SCHEDULING = '''"""Two-step scheduling, a day and then a time, in the visitor's own timezone.

Plain logic with no Parlant imports, so every rule here is tested without a
model or a server. The tools in tools.py read session metadata, and the
visitor's timezone and locale from your site's page_view events, into a `state`
dict, call a `Scheduler` method, write `Result.state` back, and return the
rest as a ToolResult.

The days and times are offered as the chat widget's built-in `choices`
widget, in the tool result's metadata (which the model never sees):

    {"widget": {"type": "choices", "id": "meeting-dates" | "meeting-times",
                "prompt": "Which day suits you?",
                "options": [{"id": "<ISO date, or UTC start>", "label": "<what the visitor reads>"}]}}

Clicking an option sends its label as the visitor's own message, so the
agent reads a click exactly like a typed answer.
"""

import re
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from domains.leadgen.service import BUSINESS_HOURS, BUSINESS_TZ, CalendarService, Meeting, Slot

MAX_DATES = 5
MAX_TIMES = 4
DEFAULT_RANGE_DAYS = 14
MAX_RANGE_DAYS = 60

# Zone names visitors recognise. Any other zone uses its own abbreviation
# (CET, AEDT, ...).
_SHORT = {
    "America/New_York": "ET",
    "America/Toronto": "ET",
    "America/Detroit": "ET",
    "America/Chicago": "CT",
    "America/Denver": "MT",
    "America/Phoenix": "MT",
    "America/Los_Angeles": "PT",
    "America/Vancouver": "PT",
    "Europe/London": "UK time",
}

_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

# Day and month names for the labels, by the language the browser reports
# (`navigator.language`, sent as `locale` in your site's page_view events).
_NAMES = {
    "de": (
        ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag"],
        [
            "Jan.",
            "Feb.",
            "März",
            "Apr.",
            "Mai",
            "Juni",
            "Juli",
            "Aug.",
            "Sept.",
            "Okt.",
            "Nov.",
            "Dez.",
        ],
        "{w}, {d}. {m}",
    ),
    "fr": (
        ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"],
        [
            "janv.",
            "févr.",
            "mars",
            "avr.",
            "mai",
            "juin",
            "juil.",
            "août",
            "sept.",
            "oct.",
            "nov.",
            "déc.",
        ],
        "{w} {d} {m}",
    ),
    "es": (
        ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"],
        [
            "ene.",
            "feb.",
            "mar.",
            "abr.",
            "may.",
            "jun.",
            "jul.",
            "ago.",
            "sept.",
            "oct.",
            "nov.",
            "dic.",
        ],
        "{w} {d} de {m}",
    ),
    "he": (
        ["יום שני", "יום שלישי", "יום רביעי", "יום חמישי", "יום שישי", "שבת", "יום ראשון"],
        [
            "בינו׳",
            "בפבר׳",
            "במרץ",
            "באפר׳",
            "במאי",
            "ביוני",
            "ביולי",
            "באוג׳",
            "בספט׳",
            "באוק׳",
            "בנוב׳",
            "בדצמ׳",
        ],
        "{w}, {d} {m}",
    ),
}

# The widget prompts: which day, which time, and a time that has just gone.
_PROMPTS = {
    "en": (
        "Which day suits you?",
        "Which time suits you? Times are in {z}.",
        "That time just went. Which of these suits you?",
    ),
    "de": (
        "Welcher Tag passt Ihnen?",
        "Welche Uhrzeit passt Ihnen? Zeiten in {z}.",
        "Diese Zeit ist leider weg. Passt eine davon?",
    ),
    "fr": (
        "Quel jour vous convient ?",
        "Quelle heure vous convient ? Heures en {z}.",
        "Ce créneau vient d'être pris. L'un de ceux-ci vous convient-il ?",
    ),
    "es": (
        "¿Qué día le viene bien?",
        "¿Qué hora le viene bien? Horas en {z}.",
        "Esa hora acaba de ocuparse. ¿Le sirve alguna de estas?",
    ),
    "he": (
        "איזה יום מתאים לך?",
        "איזו שעה מתאימה לך? השעות לפי {z}.",
        "המועד הזה נתפס. האם אחד מאלה מתאים?",
    ),
}

_AT = {"en": "at", "de": "um", "fr": "à", "es": "a las", "he": "בשעה"}

DATES_PROMPT, TIMES_PROMPT, TAKEN_PROMPT = 0, 1, 2


def lang_of(locale: str | None) -> str:
    code = (locale or "en").split("-")[0].lower()
    return code if code in _PROMPTS else "en"


def prompt(locale: str | None, which: int, zone: str = "") -> str:
    return _PROMPTS[lang_of(locale)][which].format(z=zone)


@dataclass
class Result:
    data: dict[str, Any]  # what the agent reads
    metadata: dict[str, Any] = field(default_factory=dict)  # the widget, for the chat only
    state: dict[str, Any] = field(default_factory=dict)  # session metadata to write back
    canned_response_fields: dict[str, Any] = field(default_factory=dict)
    booked: Meeting | None = None


# --- Formatting -----------------------------------------------------------


def visitor_zone(name: str | None) -> tuple[ZoneInfo, bool]:
    """The visitor's zone, and whether we had to assume the team's zone instead."""
    if name:
        try:
            return ZoneInfo(name), False
        except (ZoneInfoNotFoundError, ValueError):
            pass

    return BUSINESS_TZ, True


def uses_12h(tz: ZoneInfo) -> bool:
    return tz.key.startswith(("America/", "Australia/"))


def zone_short(tz: ZoneInfo, at: datetime) -> str:
    return _SHORT.get(tz.key) or at.astimezone(tz).tzname() or tz.key


def date_label(d: date, tz: ZoneInfo, locale: str | None = None) -> str:
    lang = lang_of(locale)

    if lang in _NAMES:
        weekdays, months, fmt = _NAMES[lang]
        return fmt.format(w=weekdays[d.weekday()], d=d.day, m=months[d.month - 1])

    if uses_12h(tz):
        return f"{d:%A}, {d:%b} {d.day}"  # Tuesday, Oct 27

    return f"{d:%A} {d.day} {d:%b}"  # Tuesday 27 Oct


def _clock(dt: datetime, h12: bool) -> tuple[str, str]:
    if h12:
        return f"{dt.hour % 12 or 12}:{dt.minute:02d}", ("AM" if dt.hour < 12 else "PM")

    return f"{dt.hour}:{dt.minute:02d}", ""


def time_range_label(slot: Slot, tz: ZoneInfo) -> str:
    start, end = slot.start.astimezone(tz), slot.end.astimezone(tz)
    h12 = uses_12h(tz)
    a, ap = _clock(start, h12)
    b, bp = _clock(end, h12)

    if not h12:
        return f"{a}–{b}"  # 16:30–17:00

    return f"{a}–{b} {bp}" if ap == bp else f"{a} {ap}–{b} {bp}"  # 9:30–10:00 AM


def confirmation_time(start: datetime, tz: ZoneInfo, locale: str | None = None) -> str:
    """'Tuesday, Oct 27 at 9:30 AM ET': always with the zone, so nobody guesses."""
    local = start.astimezone(tz)
    a, ap = _clock(local, uses_12h(tz))
    clock = f"{a} {ap}".strip()
    return f"{date_label(local.date(), tz, locale)} {_AT[lang_of(locale)]} {clock} {zone_short(tz, start)}"


def choices_widget(widget_id: str, prompt: str, options: list[dict[str, str]]) -> dict[str, Any]:
    """The chat widget's built-in `choices` widget. Options carry only what it reads."""
    return {
        "widget": {
            "type": "choices",
            "id": widget_id,
            "prompt": prompt,
            "options": [{"id": o["id"], "label": o["label"]} for o in options],
        }
    }


def utc_id(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_utc_id(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


# --- Matching what the visitor clicked or typed -----------------------------


_ZONE_WORDS = (
    r"\\b(et|est|edt|pt|pst|pdt|ct|cst|cdt|mt|cet|cest|bst|gmt|aedt|aest"
    r"|uk time|uk|utc|your time|local time)\\b"
)


def _norm(s: str) -> str:
    s = s.lower().replace("–", "-").replace("—", "-").replace(" to ", "-")
    s = re.sub(_ZONE_WORDS, "", s)
    s = re.sub(r"\\b(at|on|the|please|works|for me|let's do|lets do|ok|okay)\\b", "", s)
    return re.sub(r"[^a-z0-9:\\-]", "", s)


def _clock_tokens(s: str) -> list[tuple[int, int, str | None]]:
    """Clock times in free text, as (hour, minute, 'am'/'pm'/None). Only tokens
    with minutes or an am/pm count, so a date like '27' isn't read as a time."""
    out = []

    for m in re.finditer(
        r"\\b(\\d{1,2})(?:[:.h](\\d{2}))?\\s*(a\\.?m\\.?|p\\.?m\\.?|uhr)?(?![\\d])", s.lower()
    ):
        hour, minute, suffix = int(m.group(1)), m.group(2), m.group(3)

        if minute is None and (suffix is None or suffix == "uhr"):
            continue

        if hour > 23:
            continue

        ap = (
            "am"
            if suffix and suffix.startswith("a")
            else "pm"
            if suffix and suffix.startswith("p")
            else None
        )
        out.append((hour, int(minute or 0), ap))

    return out


def _clock_matches(local: datetime, token: tuple[int, int, str | None]) -> bool:
    hour, minute, ap = token

    if local.minute != minute:
        return False

    if ap == "am":
        return local.hour == hour % 12

    if ap == "pm":
        return local.hour == hour % 12 + 12

    return local.hour == hour or (hour <= 12 and local.hour == hour + 12)


def match_offered(text: str, offered: list[dict[str, str]], tz: ZoneInfo) -> dict[str, str] | None:
    """Which offered time the visitor meant: the exact label (a click), the id,
    or a looser version ("Tuesday 27 Oct at 16:30 CET", "4:30pm", "9.30").
    None when it isn't clear."""
    t = _norm(text)

    if not t:
        return None

    for o in offered:
        keys = {_norm(o["label"]), o["id"].lower(), _norm(o.get("date_label", "") + o["label"])}
        keys.add(_norm(o["label"]).split("-")[0])

        if t in keys:
            return o

    if tokens := _clock_tokens(text):
        hits = [
            o for o in offered if _clock_matches(parse_utc_id(o["id"]).astimezone(tz), tokens[0])
        ]

        if len(hits) == 1:
            return hits[0]

    return None


def match_offered_date(text: str, offered: list[dict[str, str]]) -> dict[str, str] | None:
    """Which offered day the visitor meant: the label, the ISO date, or a weekday name."""
    t = _norm(text)

    for o in offered:
        if t in {_norm(o["label"]), o["id"].lower()}:
            return o

    try:
        d = date.fromisoformat(text.strip()[:10])
        return next(
            (o for o in offered if o["id"] == d.isoformat()), {"id": d.isoformat(), "label": ""}
        )
    except ValueError:
        pass

    low = text.lower()

    for i, weekday in enumerate(_WEEKDAYS):
        names = [weekday] + [n[0][i].lower() for n in _NAMES.values()]

        if any(n in low for n in names):
            hits = [o for o in offered if date.fromisoformat(o["id"]).weekday() == i]
            return hits[0] if len(hits) == 1 else None

    return None


# --- The scheduler ----------------------------------------------------------


class Scheduler:
    def __init__(
        self,
        calendar: CalendarService,
        *,
        now: datetime | None = None,
        waking_hours: tuple[int, int] | None = None,
    ) -> None:
        self._calendar = calendar
        self._now = now
        # The (first, last) hour of the visitor's day to offer; None offers every hour.
        self._waking = waking_hours

    def now(self) -> datetime:
        return self._now or datetime.now(timezone.utc)

    @staticmethod
    def _guard(state: dict[str, Any], visitor_asked: bool) -> dict[str, Any] | None:
        # The rep needs a recorded lead to prepare for, so times come after record_lead.
        if state.get("lead_id") is None:
            return {
                "reason": "No lead is recorded yet. Ask for what record_lead needs (one question "
                "at a time) and record it before offering times."
            }

        # A cold lead gets times only when they ask for them, and never when
        # they asked not to be contacted.
        if state.get("lead_band") == "cold" and (not visitor_asked or state.get("do_not_contact")):
            return {
                "reason": "Calls and visits are for bigger projects. Without mentioning scores or "
                "lead types, suggest the measuring guide, free samples and the pricing page."
            }

        return None

    def _filter(self, slots: list[Slot], tz: ZoneInfo, preference: str) -> list[Slot]:
        pref = (preference or "").lower()
        named = [w for w in _WEEKDAYS if w in pref]
        out = []

        for s in slots:
            local = s.start.astimezone(tz)

            if self._waking and not (self._waking[0] <= local.hour < self._waking[1]):
                continue
            if "morning" in pref and local.hour >= 12:
                continue
            if "afternoon" in pref and not (12 <= local.hour < 18):
                continue
            if "evening" in pref and local.hour < 17:
                continue
            if named and _WEEKDAYS[local.weekday()] not in named:
                continue

            out.append(s)

        return out

    def _zone_note(self, tz: ZoneInfo, assumed: bool, at: datetime) -> dict[str, Any]:
        """The visitor's zone, and the team's hours in it, for the agent to explain."""
        data: dict[str, Any] = {"timezone": tz.key, "timezone_short": zone_short(tz, at)}

        day = at.astimezone(BUSINESS_TZ).date()
        first, last = BUSINESS_HOURS[0], BUSINESS_HOURS[-1]
        opens = datetime.combine(day, time(*first), BUSINESS_TZ)
        closes = datetime.combine(day, time(*last), BUSINESS_TZ) + timedelta(minutes=30)
        h12 = uses_12h(tz)
        a, ap = _clock(opens.astimezone(tz), h12)
        b, bp = _clock(closes.astimezone(tz), h12)
        data["team_hours"] = " ".join(
            f"Our team takes calls between {first[0]}:{first[1]:02d} and "
            f"{closes.hour}:{closes.minute:02d} UK time, which is {a} {ap} to {b} {bp} "
            f"{data['timezone_short']}".split()
        )

        if assumed:
            data["timezone_note"] = (
                "We don't know the visitor's timezone, so times are in UK time. Always say "
                "'UK time', and ask where they are if it might not suit them."
            )

        if self._waking is not None:
            data["hours_note"] = (
                f"Only times between {self._waking[0]}:00 and {self._waking[1]}:00 the visitor's "
                "time are shown; our team is in the UK."
            )

        return data

    async def _slots_between(
        self, state: dict[str, Any], first: date, last: date, tz: ZoneInfo, preference: str
    ) -> list[Slot]:
        """Open slots on the visitor's local days first..last, filtered."""
        lo = datetime.combine(first, time(0), tz).astimezone(timezone.utc)
        hi = datetime.combine(last + timedelta(days=1), time(0), tz).astimezone(timezone.utc)
        slots = await self._calendar.open_slots(lead_id=state.get("lead_id"), start=lo, end=hi)
        return self._filter(slots, tz, preference)

    # Step 1: which days have times.

    async def dates(
        self,
        state: dict[str, Any],
        *,
        start_date: str = "",
        end_date: str = "",
        preference: str = "",
        visitor_asked: bool = False,
    ) -> Result:
        if blocked := self._guard(state, visitor_asked):
            return Result(data={"dates": [], **blocked})

        tz, assumed = visitor_zone(state.get("timezone"))
        locale = state.get("locale")
        today = self.now().astimezone(tz).date()

        try:
            first = date.fromisoformat(start_date) if start_date else today + timedelta(days=1)
            last = (
                date.fromisoformat(end_date)
                if end_date
                else first + timedelta(days=DEFAULT_RANGE_DAYS - 1)
            )
        except ValueError:
            return Result(
                data={
                    "dates": [],
                    "reason": "start_date and end_date must be ISO dates (YYYY-MM-DD).",
                }
            )

        first = max(first, today + timedelta(days=1))

        if last < first:
            return Result(
                data={"dates": [], "reason": f"The range ends ({last}) before it starts ({first})."}
            )

        last = min(last, first + timedelta(days=MAX_RANGE_DAYS))
        slots = await self._slots_between(state, first, last, tz, preference)
        searched = f"{first.isoformat()} to {last.isoformat()}"

        days: list[date] = []
        for s in slots:
            if (d := s.start.astimezone(tz).date()) not in days:
                days.append(d)
        days = days[:MAX_DATES]

        zone = self._zone_note(
            tz, assumed, slots[0].start if slots else datetime.combine(first, time(0), tz)
        )

        if not days:
            return Result(
                data={
                    "dates": [],
                    "searched": searched,
                    "reason": f"No open times match that. {zone['team_hours']}. Tell the visitor "
                    "that, and offer to look without the time-of-day preference or in another range.",
                    **zone,
                },
                state={"offered_dates": [], "offered_times": []},
            )

        options = [{"id": d.isoformat(), "label": date_label(d, tz, locale)} for d in days]

        return Result(
            data={
                "dates": [o["label"] for o in options],
                "searched": searched,
                "next": "When the visitor picks a day, call find_meeting_times with it.",
                **zone,
            },
            metadata=choices_widget("meeting-dates", prompt(locale, DATES_PROMPT), options),
            state={
                "offered_dates": options,
                "offered_times": [],
                "schedule_preference": preference,
            },
        )

    # Step 2: the times on the day they picked.

    async def times(
        self,
        state: dict[str, Any],
        *,
        date_text: str,
        preference: str = "",
        visitor_asked: bool = False,
    ) -> Result:
        if blocked := self._guard(state, visitor_asked):
            return Result(data={"times": [], **blocked})

        tz, assumed = visitor_zone(state.get("timezone"))
        locale = state.get("locale")
        offered_dates = state.get("offered_dates") or []
        picked = match_offered_date(date_text, offered_dates)

        if picked is None:
            return Result(
                data={
                    "times": [],
                    "reason": f"'{date_text}' isn't one of the offered days. Ask which one they "
                    "meant, or call find_meeting_dates for a different range.",
                },
                metadata=choices_widget(
                    "meeting-dates", prompt(locale, DATES_PROMPT), offered_dates
                )
                if offered_dates
                else {},
            )

        day = date.fromisoformat(picked["id"])
        label = picked["label"] or date_label(day, tz, locale)
        preference = preference or state.get("schedule_preference") or ""
        slots = (await self._slots_between(state, day, day, tz, preference))[:MAX_TIMES]
        zone = self._zone_note(tz, assumed, datetime.combine(day, time(0), tz))

        if not slots:
            others = [o for o in offered_dates if o["id"] != picked["id"]]
            return Result(
                data={
                    "times": [],
                    "date": label,
                    "reason": "No open times left that day. Offer the other days.",
                    **zone,
                },
                metadata=choices_widget("meeting-dates", prompt(locale, DATES_PROMPT), others)
                if others
                else {},
                state={"offered_times": []},
            )

        options = [
            {
                "id": s.id,
                "label": time_range_label(s, tz),
                "date": day.isoformat(),
                "date_label": label,
            }
            for s in slots
        ]

        return Result(
            data={
                "date": label,
                "times": [o["label"] for o in options],
                "next": "When the visitor picks a time, book it with book_meeting.",
                **zone,
            },
            metadata=choices_widget(
                "meeting-times", prompt(locale, TIMES_PROMPT, zone["timezone_short"]), options
            ),
            state={"offered_times": options},
        )

    # Step 3: book the time they picked.

    async def book(self, state: dict[str, Any], *, slot_text: str, description: str) -> Result:
        if state.get("lead_id") is None:
            return Result(
                data={"booked": False, "reason": "Record the lead with record_lead first."}
            )

        tz, assumed = visitor_zone(state.get("timezone"))
        locale = state.get("locale")
        offered = state.get("offered_times") or []

        if not offered:
            return Result(
                data={
                    "booked": False,
                    "reason": "No times have been offered yet. Call find_meeting_dates, then "
                    "find_meeting_times.",
                }
            )

        picked = match_offered(slot_text, offered, tz)

        if picked is None:
            return Result(
                data={
                    "booked": False,
                    "reason": f"'{slot_text}' doesn't match an offered time. Nothing was booked.",
                    "offered_times": [o["label"] for o in offered],
                },
                metadata=choices_widget(
                    "meeting-times",
                    prompt(locale, TIMES_PROMPT, zone_short(tz, self.now())),
                    offered,
                ),
            )

        meeting = await self._calendar.try_book(
            lead_id=state["lead_id"], start=parse_utc_id(picked["id"]), description=description
        )

        # A slot taken in the meantime is an ordinary outcome: offer what is
        # still open that day, as data the agent can act on.
        if meeting is None:
            day = date.fromisoformat(picked["date"])
            open_now = {s.id: s for s in await self._slots_between(state, day, day, tz, "")}
            remaining = [o for o in offered if o["id"] != picked["id"] and o["id"] in open_now]

            if not remaining:
                remaining = [
                    {
                        "id": s.id,
                        "label": time_range_label(s, tz),
                        "date": day.isoformat(),
                        "date_label": picked.get("date_label", ""),
                    }
                    for s in list(open_now.values())[:MAX_TIMES]
                ]

            data: dict[str, Any] = {
                "booked": False,
                "reason": f"{picked['label']} on {picked.get('date_label', '')} was just taken. "
                "Nothing was booked.",
                "other_times_that_day": [o["label"] for o in remaining],
            }

            if not remaining:
                data["next"] = "No other times that day: offer other days with find_meeting_dates."

            return Result(
                data=data,
                metadata=choices_widget("meeting-times", prompt(locale, TAKEN_PROMPT), remaining)
                if remaining
                else {},
                state={"offered_times": remaining},
            )

        when = confirmation_time(meeting.start, tz, locale)
        data = {"booked": True, "rep": meeting.rep_name, "time": when, "timezone": tz.key}

        if assumed:
            data["timezone_note"] = (
                "The time is in UK time because we don't know the visitor's timezone."
            )

        return Result(
            data=data,
            state={"booked_slot": picked["id"], "offered_times": []},
            canned_response_fields={"rep": meeting.rep_name, "time": when},
            booked=meeting,
        )
'''


LEADGEN_PAGES = '''"""Where the visitor is, read from the page_view events your site posts.

When the chat opens, and on every page view after that, your site's backend
adds a custom event to the session (see the README):

    {"kind": "custom", "source": "customer_ui",
     "data": {"type": "page_view", "page": "/pricing", "title": "Pricing",
              "timezone": "Europe/Berlin", "locale": "de-DE",
              "utm_campaign": "autumn-sale"}}

The agent sees these events in its history, so it knows the page and notices
a move. The code reads the same events: the page policies, the retriever,
scheduling and the CRM all work from the facts the agent sees.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import parlant.sdk as p


@dataclass(frozen=True)
class PageState:
    current_page: str | None = None  # the latest page_view
    landing_page: str | None = None  # the first page_view
    timezone: str | None = None  # from the browser; None means UK time
    locale: str | None = None  # the browser's language, e.g. "de-DE"
    utm_campaign: str | None = None  # from the first page_view, if the visit had one


def _text(data: Mapping[str, Any], key: str) -> str | None:
    value = data.get(key)
    return value if isinstance(value, str) and value else None


def page_views(events: Iterable[Any]) -> list[Mapping[str, Any]]:
    """The data of each page_view event, oldest first."""
    return [
        e.data
        for e in events
        if e.kind == p.EventKind.CUSTOM
        and e.source == p.EventSource.CUSTOMER_UI
        and isinstance(e.data, Mapping)
        and e.data.get("type") == "page_view"
    ]


def page_state(events: Iterable[Any]) -> PageState:
    """The visitor's current page, timezone and locale (from the latest page_view that
    reports each), and where they landed (from the first)."""
    views = page_views(events)

    if not views:
        return PageState()

    def latest(key: str) -> str | None:
        return next((value for view in reversed(views) if (value := _text(view, key))), None)

    # A long conversation's history can be compacted, which drops its oldest
    # events. Every page_view carries the timezone and locale, so they survive;
    # the landing page is then the earliest page_view still in the history.
    return PageState(
        current_page=_text(views[-1], "page"),
        landing_page=_text(views[0], "page"),
        timezone=latest("timezone"),
        locale=latest("locale"),
        utm_campaign=_text(views[0], "utm_campaign"),
    )


def visitor_page() -> PageState:
    """The page state of the session a tool is running in."""
    return page_state(p.Session.current.interaction.events)
'''


LEADGEN_DOMAIN = '''"""Everything this domain declares on the agent.

`install` adds the catalog retriever, page-aware policies, the qualification
procedure, routing on the lead's band, the guardrails and the canned
responses. The conversation cap that hands a visitor to your team is the
`handoff` domain's.
"""

import datetime

import parlant.sdk as p
import services
from domains.leadgen.pages import page_state
from domains.leadgen.scheduling import visitor_zone
from domains.leadgen.tools import (
    book_meeting,
    find_meeting_dates,
    find_meeting_times,
    record_lead,
    request_team_follow_up,
    send_follow_up,
)

SCHEDULING = [find_meeting_dates, find_meeting_times, book_meeting, request_team_follow_up]


async def product_knowledge(ctx: p.RetrieverContext) -> p.RetrieverResult:
    """Product answers come from the catalog, so the agent quotes it rather than guessing."""
    # The model has no clock, so give it today's date in the visitor's own
    # timezone (to turn "before Christmas" into a timeline), with the page
    # they're on now, from your site's latest page_view event.
    page = page_state(ctx.session.interaction.events)
    tz, assumed = visitor_zone(page.timezone)

    today = datetime.datetime.now(tz).date()
    visitor = {
        "today": f"{today:%A %d %B %Y} ({today.isoformat()})",
        "current_page": page.current_page,
        "timezone": "unknown (times are shown in UK time)" if assumed else tz.key,
    }

    return p.RetrieverResult(data={"visitor": visitor, **services.catalog().facts()})


def current_page_is(*pages: str):
    """An evaluation_func: the policy applies while the visitor's current page is one of `pages`.

    The page comes from the latest page_view event your site posted, so the
    policy is decided in code, from a fact, every time. The agent already sees
    that event in its history, so the policy's content is what to do on that
    page, not where the visitor is.
    """

    async def evaluate(ctx: p.RuleEvaluationContext, policy: p.Policy) -> p.RuleEvaluation:
        page = page_state(ctx.session.interaction.events).current_page
        return p.RuleEvaluation(
            id=policy.id, is_relevant=page in pages, reasoning=f"current page is {page!r}"
        )

    return evaluate


def band_is(band: str):
    """An evaluation_func: the policy applies exactly when record_lead has put the lead in `band`.

    Matched by the model alone, a band policy can apply on intent ("wants a
    call") before record_lead has run. Reading the band record_lead stored
    keeps each policy to the leads it is for.
    """

    async def evaluate(ctx: p.RuleEvaluationContext, policy: p.Policy) -> p.RuleEvaluation:
        current = await ctx.session.metadata.get("lead_band")
        return p.RuleEvaluation(
            id=policy.id, is_relevant=current == band, reasoning=f"lead band is {current!r}"
        )

    return evaluate


async def install(agent: p.Agent) -> None:
    await agent.attach_retriever(product_knowledge, id="catalog")

    # -- Page awareness -----------------------------------------------------
    await agent.create_policy(
        title="Pricing page visitors",
        content=(
            "Lead with what our prices cover: a 'from' price is for a small window, with "
            "linings, motorization and fitting added per window, and the exact price comes in "
            "a written quote after measuring. When they ask what it would cost them, give the "
            "relevant 'from' prices, then ask how many windows they're dressing before "
            "narrowing it down"
        ),
        evaluation_func=current_page_is("/pricing"),
    )

    await agent.create_policy(
        title="Trade page visitors",
        content=(
            "Treat them as a likely trade buyer, buying for a business or a client. In your next "
            "answer, add in one short sentence that for business projects we offer a trade "
            "programme (a dedicated project manager, FR-certified fabrics, phased installs)"
        ),
        evaluation_func=current_page_is("/trade", "/lp/hotel-refit"),
    )

    # -- Answer first, then qualify -----------------------------------------
    await agent.create_policy(
        title="Answer before qualifying",
        content=(
            "When the visitor asks a question, answer it before asking any of your own. "
            "Never hold an answer back to get their details."
        ),
    )

    # -- The qualification procedure ----------------------------------------
    estimate_offer = await agent.create_canned_response(
        template=(
            "I can put together a ballpark estimate for your windows. "
            "Which email should I send it to?"
        ),
        note="Always say what they get in return before asking for an email.",
    )

    await agent.create_policy(
        title="Qualify a potential buyer",
        content="""\\
When the visitor shows buying interest (they ask about pricing for their own
windows, a quote, a measuring visit, or how Hearth & Drape would work for their
home or project; questions about how a product works are not buying interest),
follow this procedure, tracking which step the conversation is at. Ask one
question per message. Skip anything the visitor has already told you.

1. Ask what they're working on, in their own words (which rooms, what look or problem).
2. Ask whether it's for their own home or for a client or business, and their role.
3. Ask roughly how many windows are involved.
4. Ask when they'd like everything fitted.
5. Offer something specific in return for an email address: a ballpark estimate
   for their windows, or our trade price guide for trade buyers. If they'd rather
   be called, take a phone number instead and don't push for the email.
6. Ask whether there is anything they would like our team to know, such as
   other suppliers they're comparing or a deadline.
7. Record everything with `record_lead`. If you learn something new afterwards
   (more windows, a business project, a timeline), call `record_lead` again.

If the visitor wants to stop at any point, or says they don't want questions,
respect it: answer what they asked, ask nothing more (not even for an email), and
mention the self-serve resources (the pricing page, free samples, the measuring
guide). Never ask more than five questions before step 7.
""",
        tools=[record_lead],
        canned_responses=[estimate_offer],
        labels=["lead-intent"],
    )

    # -- Routing on the band ------------------------------------------------
    hot = await agent.create_policy(
        title="Hot lead: book a meeting",
        content=(
            "`record_lead` found this visitor's project a strong fit for our trade team "
            "(internal: never call them a lead or say how they were scored). Send anything you "
            "promised with `send_follow_up`, and offer a 30-minute consultation call with our "
            "trade team. Find open days with `find_meeting_dates`, then the times on the day "
            "they pick with `find_meeting_times`, and book the time they pick with "
            "`book_meeting`. If none of them work, ask what would, and search again with that. "
            "If nothing we have suits them, ask the team to contact them with "
            "`request_team_follow_up`."
        ),
        tools=[send_follow_up, *SCHEDULING],
        evaluation_func=band_is("hot"),
    )

    warm = await agent.create_policy(
        title="Warm lead: follow up",
        content=(
            "`record_lead` found this visitor's project a good fit (internal: never call them a "
            "lead or say how they were scored). Send the estimate or price guide you promised "
            "with `send_follow_up`. Then offer a free measure-and-quote visit or a call as the "
            "natural next step, and book it if they want one (`find_meeting_dates`, then "
            "`find_meeting_times`, then `book_meeting`). If none of the times work, ask what "
            "would, and search again with that. If nothing we have suits them, ask the team to "
            "contact them with `request_team_follow_up`."
        ),
        tools=[send_follow_up, *SCHEDULING],
        evaluation_func=band_is("warm"),
    )

    cold = await agent.create_policy(
        title="Poor fit: self-serve",
        content=(
            "`record_lead` found this visitor's project too small for a sales call (internal: "
            "never say so or call them a lead). If you promised them an estimate for their "
            "email, send it with `send_follow_up`. Thank them, point them to the pricing page "
            "and free fabric samples, and offer to answer any other questions. Don't offer a "
            "call or a visit, and don't promise anything else by email. If they ask to talk to "
            "someone themselves, find a day and time with `find_meeting_dates` and "
            "`find_meeting_times` (visitor_asked=true) and book the one they pick."
        ),
        tools=[send_follow_up, *SCHEDULING],
        evaluation_func=band_is("cold"),
    )

    # record_lead sets the band, so the band policies are decided again
    # each time it runs.
    for policy in (hot, warm, cold):
        await policy.reevaluate_after(record_lead)

    await agent.create_policy(
        title="Only confirm what really happened",
        content=(
            "Only tell the visitor a call or visit is booked when `book_meeting` returned "
            "booked: true, and only say you've sent something when `send_follow_up` returned "
            "sent: true. If a booking didn't go through, say that time just went, and offer the "
            "other open times it returned. Never promise that someone will get in touch unless "
            "a meeting was booked or `request_team_follow_up` returned requested: true."
        ),
    )

    # -- "Can I talk to someone?" -------------------------------------------
    await agent.create_policy(
        "When the visitor asks to talk to someone from the team, offer the soonest open days "
        "with `find_meeting_dates` and times with `find_meeting_times` (visitor_asked=true), "
        "and book the one they pick with `book_meeting`. If the lead isn't recorded yet, first "
        "ask only what `record_lead` needs, in as few messages as you can, and record it",
        tools=[record_lead, *SCHEDULING],
        labels=["asked-for-team"],
    )

    # -- Visitors who are not buyers ----------------------------------------
    await agent.create_policy(
        "When an existing customer needs help with an order, a delivery, or an installation, "
        "don't qualify them. Point them to support at help@hearthanddrape.example, or answer "
        "if it's a quick question. You can't see orders or pass messages on, so don't take "
        "their contact details and don't promise that support will contact them or when"
    )

    await agent.create_policy(
        "When the visitor is looking for a job, or is selling something to Hearth & Drape, "
        "thank them and point them to hearthanddrape.example/careers or partners@hearthanddrape.example"
    )

    # -- What the agent must never do ---------------------------------------
    await agent.create_policy(
        title="No invented prices or promises",
        content=(
            "When the visitor asks for a discount, custom or trade pricing, an exact quote, a "
            "delivery date guarantee, or a product we don't list, don't agree to or invent "
            "anything. Say our team gives exact prices in a written quote, and keep helping: "
            "carry on with the qualification questions so we can get them one."
        ),
    )

    await agent.create_policy(
        title="Keep the scoring internal",
        content=(
            "Never mention lead scores, bands, fit, or tool names to the visitor, and never "
            "explain an offer or a refusal by what kind of lead they are. Use the page they're "
            "on to choose what to say, but don't tell them which page it is."
        ),
    )

    # -- Tone ---------------------------------------------------------------
    await agent.create_canned_response(
        template=(
            "We work out the exact price once we've measured, so it comes in your written "
            "quote. Our made-to-measure curtains start from $180 per window."
        ),
        note=(
            "Speak for Hearth & Drape as one of the team: we make, we fit, our prices. When "
            "the catalog has no exact answer, tell them how we'll work it out for them."
        ),
    )

    await agent.create_canned_response(
        template="Got it. What are you hoping to change about those rooms?",
        note="Warm and short. One question per message. Never list several questions at once.",
    )

    await agent.create_canned_response(
        template="You're booked with {{ rep }} for {{ time }}. {{ next_step }}",
        note=(
            "Confirm the booking only after `book_meeting` succeeds, in the visitor's language. "
            "If they're writing in another language, translate the whole sentence into it; never "
            "send this in English to someone writing in German, Spanish, French, Hebrew or any "
            "other language."
        ),
        field_dependencies=["rep", "time", "next_step"],
    )
'''


HANDOFF_INIT = '''"""The handoff domain: hand a conversation to your team when it reaches its cap."""

from domains.handoff.domain import cap_message, install_conversation_cap

__all__ = ["cap_message", "install_conversation_cap"]
'''


HANDOFF_DOMAIN = '''"""A per-conversation spending cap that hands the visitor to your sales team.

The budget is checked in code at the start of each turn, with no model
involved. On the first turn past the cap, the handler posts a fixed message,
switches the session to manual mode (the agent stops replying), labels it
`needs-human` for your team's queue, and tells the sales inbox what the team
needs to pick it up. Whatever the visitor writes next stays in the session for
the team to read.
"""

from decimal import Decimal

import parlant.sdk as p
import services
from domains.leadgen.pages import page_state

CAP_MESSAGE = (
    "We've reached the limit of what I can cover in this chat. If you share your name and "
    "email here, someone from our team will get back to you."
)

CAP_MESSAGE_KNOWN_EMAIL = (
    "We've reached the limit of what I can cover in this chat. Someone from our team can get "
    "back to you: just share your name, and confirm your email ({email}) or give us a better one."
)


def cap_message(email: str | None) -> str:
    """Don't ask for an email we already have; ask the visitor to confirm it instead."""
    return CAP_MESSAGE_KNOWN_EMAIL.format(email=email) if email else CAP_MESSAGE


async def install_conversation_cap(
    server: p.Server, agent: p.Agent, *, cap: Decimal = Decimal("1.00")
) -> p.SessionBudget:
    """Hand a conversation to your team once it has cost more than `cap`.

    The budget measures spend from what each model call cost. main.py installs
    this only when config.py says every call can be priced; see
    `config.can_measure_spend`.
    """
    budget = await server.create_session_budget(name="conversation-cap", value=cap)

    async def hand_to_team(context: p.EngineContext, match: p.RuleEvaluation) -> None:
        session = p.Session.current
        spent = await budget.measure()

        # So the same spend can't trigger this again if the team hands the
        # conversation back to the agent.
        await budget.reset()

        await session.emit_message(cap_message(await session.metadata.get("lead_email")))
        await session.update(mode="manual")  # the agent stops replying on its own
        await session.labels.add("needs-human")  # your team's queue

        # The cap can land mid-booking, so hand over what the visitor was
        # choosing from, and what they last said, so the team can finish it.
        last = context.interaction.last_customer_message
        await services.sales_inbox().notify(
            session_id=session.id,
            reason="Conversation cap reached",
            details={
                "spent": str(spent),
                "currency": budget.currency,
                "lead_id": await session.metadata.get("lead_id"),
                "lead_band": await session.metadata.get("lead_band"),
                "pending_dates": await session.metadata.get("offered_dates") or [],
                "pending_times": await session.metadata.get("offered_times") or [],
                "last_visitor_message": last.content if last else None,
                "timezone": page_state(context.interaction.events).timezone,
            },
        )

    await agent.create_observation(
        condition="The conversation has reached its spending cap",  # for people; the code decides
        evaluation_func=budget.exceeded,
        on_matched=hand_to_team,
    )

    return budget
'''


TEST_LEADGEN_DOMAIN = '''"""Tests for the lead-gen domain: the scoring rules, the contact checks, and the
fakes' behavior the tools rely on. Nothing here calls a model."""

import asyncio
from datetime import date, datetime, timezone

import pytest

from domains.leadgen.service import (
    BuyingFor,
    CrmAPI,
    FakeCalendar,
    FakeCrm,
    band_for,
    score_lead,
)
from domains.leadgen.tools import typed_email, typed_phone


def test_that_a_hotel_refit_is_hot() -> None:
    score = score_lead(BuyingFor.BUSINESS, "project manager", 120, 60, True)
    assert band_for(score) == "hot"


def test_that_a_whole_house_homeowner_is_warm_even_without_a_timeline() -> None:
    assert band_for(score_lead(BuyingFor.OWN_HOME, "homeowner", 16, None, False)) == "warm"


def test_that_an_unknown_window_count_is_not_penalised() -> None:
    assert score_lead(BuyingFor.BUSINESS, "purchasing", None, None, False) == 25


def test_that_a_one_window_student_is_cold() -> None:
    assert band_for(score_lead(BuyingFor.OWN_HOME, "student", 1, None, False)) == "cold"


def test_that_an_email_the_visitor_never_typed_is_rejected() -> None:
    typed = "it's for our hotel, about 40 rooms"
    assert typed_email("manager@hotel.com", typed) is None


def test_that_an_email_the_visitor_typed_is_accepted() -> None:
    typed = "sure, it's dana@hotel.com."
    assert typed_email("Dana@hotel.com.", typed) == "dana@hotel.com"


def test_that_a_phone_number_is_matched_on_its_digits() -> None:
    typed = "call me on 0113 496 0000"
    assert typed_phone("01134960000", typed) == "01134960000"
    assert typed_phone("07700 900123", typed) is None


def test_that_a_do_not_contact_address_is_flagged() -> None:
    lead = asyncio.run(
        FakeCrm().upsert_lead(
            email="unsubscribe@example.com",
            phone=None,
            fields={},
            notes="",
            score=50,
            source={},
            chat_session_id="s1",
        )
    )
    assert lead.do_not_contact


def test_that_a_do_not_contact_address_can_be_checked_before_any_lead_exists() -> None:
    # send_follow_up relies on this when it runs before record_lead.
    crm = FakeCrm()
    assert asyncio.run(crm.is_do_not_contact("unsubscribe@example.com"))
    assert not asyncio.run(crm.is_do_not_contact("lena.vogt@studio-vogt.de"))


def test_that_a_lead_given_by_phone_is_updated_by_phone() -> None:
    crm = FakeCrm()

    async def record_twice() -> tuple[str, str]:
        first = await crm.upsert_lead(
            email=None,
            phone="0113 496 0000",
            fields={},
            notes="",
            score=10,
            source={},
            chat_session_id="s1",
        )
        again = await crm.upsert_lead(
            email=None,
            phone="01134960000",
            fields={"window_count": 20},
            notes="",
            score=35,
            source={},
            chat_session_id="s1",
        )
        return first.id, again.id

    first_id, again_id = asyncio.run(record_twice())
    assert first_id == again_id


def test_that_a_booked_slot_is_not_offered_again() -> None:
    calendar = FakeCalendar(today=date(2026, 10, 24))
    start = datetime(2026, 10, 25, tzinfo=timezone.utc)
    end = datetime(2026, 11, 20, tzinfo=timezone.utc)

    async def book_and_look_again() -> None:
        first = await calendar.open_slots(lead_id="l1", start=start, end=end)
        assert await calendar.try_book(lead_id="l1", start=first[0].start, description="")

        again = await calendar.open_slots(lead_id="l1", start=start, end=end)
        assert first[0].id not in [slot.id for slot in again]

        # Booking it twice is an ordinary outcome, not an error.
        assert await calendar.try_book(lead_id="l1", start=first[0].start, description="") is None

    asyncio.run(book_and_look_again())


def test_that_the_crm_backend_is_not_implemented_yet() -> None:
    with pytest.raises(NotImplementedError):
        CrmAPI("https://api.example.com")
'''


TEST_LEADGEN_SCHEDULING = '''"""Tests for two-step scheduling: the widget the chat renders, timezones and
daylight saving, date ranges, the guards, and booking. Nothing here calls a model.

The fake calendar's reps start calls at 9:30, 11:00, 14:00 and 16:30 London
time, for 30 minutes. UK clocks go back on Sunday 25 October 2026 and US
clocks on Sunday 1 November, so in the week of 26 October New York is only
four hours behind London.
"""

import asyncio
from datetime import date, datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from domains.leadgen.scheduling import Result, Scheduler, parse_utc_id
from domains.leadgen.service import FakeCalendar

NOW = datetime(2026, 10, 24, 12, 0, tzinfo=timezone.utc)  # a Saturday


def run(coro: Any) -> Any:
    return asyncio.run(coro)


def open_calendar(today: date = date(2026, 10, 24)) -> FakeCalendar:
    """A calendar with every slot open, so the expected times are easy to read."""
    return FakeCalendar(today=today, taken=lambda local: False)


@pytest.fixture
def calendar() -> FakeCalendar:
    return open_calendar()


def state(tz: str | None = "America/New_York", **overrides: Any) -> dict[str, Any]:
    """The session metadata the tools read, for a hot lead."""
    return {
        "lead_id": "lead-1",
        "lead_band": "hot",
        "do_not_contact": False,
        "timezone": tz,
        "offered_dates": None,
        "offered_times": None,
        **overrides,
    }


def apply(st: dict[str, Any], result: Result) -> Result:
    """Write back what the tool would write to session metadata."""
    st.update(result.state)
    return result


def assert_choices_widget(metadata: dict[str, Any], widget_id: str) -> list[dict[str, str]]:
    """The chat widget's built-in `choices` contract, exactly. It renders
    nothing for a payload that breaks it."""
    assert set(metadata) == {"widget"}
    widget = metadata["widget"]
    assert set(widget) == {"type", "id", "prompt", "options"}
    assert widget["type"] == "choices"
    assert widget["id"] == widget_id
    assert isinstance(widget["prompt"], str) and widget["prompt"].strip()
    assert 1 <= len(widget["options"]) <= 8

    for option in widget["options"]:
        assert {"id", "label"} <= set(option) <= {"id", "label", "reply"}
        assert isinstance(option["id"], str) and option["id"].strip()
        assert isinstance(option["label"], str) and option["label"].strip()

    ids = [option["id"] for option in widget["options"]]
    assert len(ids) == len(set(ids)), "option ids must be unique"

    return widget["options"]


def offer_times(s: Scheduler, st: dict[str, Any], day: str = "Monday, Oct 26") -> Result:
    apply(st, run(s.dates(st)))
    return apply(st, run(s.times(st, date_text=day)))


# --- The widget ---------------------------------------------------------------


def test_that_dates_are_offered_as_a_choices_widget_with_iso_ids(calendar: FakeCalendar) -> None:
    result = run(Scheduler(calendar, now=NOW).dates(state()))
    options = assert_choices_widget(result.metadata, "meeting-dates")

    assert len(options) == 5
    assert date.fromisoformat(options[0]["id"]) == date(2026, 10, 26)
    assert options[0]["label"] == "Monday, Oct 26"
    assert result.data["dates"] == [o["label"] for o in options]


def test_that_times_have_utc_ids_and_labels_in_the_visitor_zone(calendar: FakeCalendar) -> None:
    result = offer_times(Scheduler(calendar, now=NOW), st := state())
    options = assert_choices_widget(result.metadata, "meeting-times")

    assert len(options) == 4
    # London is back on GMT, so 9:30 London is 9:30 UTC...
    assert parse_utc_id(options[0]["id"]) == datetime(2026, 10, 26, 9, 30, tzinfo=timezone.utc)
    # ...and New York is still on EDT (UTC-4) until 1 November.
    assert [o["label"] for o in options] == [
        "5:30–6:00 AM",
        "7:00–7:30 AM",
        "10:00–10:30 AM",
        "12:30–1:00 PM",
    ]
    assert "ET" in result.metadata["widget"]["prompt"]
    assert [o["id"] for o in st["offered_times"]] == [o["id"] for o in options]


def test_that_the_model_reads_labels_but_never_the_widget(calendar: FakeCalendar) -> None:
    result = run(Scheduler(calendar, now=NOW).dates(state()))
    assert "widget" not in result.data


# --- Timezones and daylight saving -------------------------------------------------


def test_that_new_york_times_shift_an_hour_once_us_clocks_change(calendar: FakeCalendar) -> None:
    s = Scheduler(calendar, now=NOW)
    st = state()
    apply(st, run(s.dates(st, start_date="2026-11-02", end_date="2026-11-02")))

    assert run(s.times(st, date_text="2026-11-02")).data["times"][0] == "4:30–5:00 AM"  # EST


def test_that_berlin_gets_24_hour_times_in_cet(calendar: FakeCalendar) -> None:
    s = Scheduler(calendar, now=NOW)
    st = state("Europe/Berlin")
    apply(st, run(s.dates(st)))
    result = run(s.times(st, date_text=st["offered_dates"][0]["label"]))

    assert st["offered_dates"][0]["label"] == "Monday 26 Oct"
    assert result.data["times"] == ["10:30–11:00", "12:00–12:30", "15:00–15:30", "17:30–18:00"]
    assert result.data["timezone_short"] == "CET"


def test_that_a_uk_clock_change_inside_the_range_is_handled() -> None:
    # The range spans the UK change on Sunday 25 October.
    s = Scheduler(
        open_calendar(date(2026, 10, 22)), now=datetime(2026, 10, 22, 12, tzinfo=timezone.utc)
    )
    st = state("Europe/Berlin")
    apply(st, run(s.dates(st, start_date="2026-10-23", end_date="2026-10-26")))

    assert [o["id"] for o in st["offered_dates"]] == ["2026-10-23", "2026-10-26"]

    before = run(s.times(st, date_text="2026-10-23")).data["times"][0]
    after = apply(st, run(s.times(st, date_text="2026-10-26"))).data["times"][0]

    assert before == after == "10:30–11:00"  # both zones moved together
    assert parse_utc_id(st["offered_times"][0]["id"]).hour == 9  # 9:30 GMT on the 26th


def test_that_the_spring_gap_between_us_and_uk_clock_changes_is_handled() -> None:
    # The US springs forward on 14 March 2027 and the UK on 28 March, so for
    # two weeks New York is four hours behind.
    s = Scheduler(
        open_calendar(date(2027, 3, 12)), now=datetime(2027, 3, 12, 12, tzinfo=timezone.utc)
    )
    st = state()
    apply(st, run(s.dates(st, start_date="2027-03-15", end_date="2027-03-15")))

    assert run(s.times(st, date_text="2027-03-15")).data["times"][0] == "5:30–6:00 AM"


def test_that_late_london_slots_fall_on_the_next_day_in_sydney(calendar: FakeCalendar) -> None:
    s = Scheduler(calendar, now=NOW)
    st = state("Australia/Sydney")
    apply(st, run(s.dates(st, start_date="2026-10-27", end_date="2026-10-27")))

    # Sydney is UTC+11. Tuesday 27 October there holds Monday's 14:00 and
    # 16:30 London slots (1:00, 3:30) and Tuesday's 9:30 and 11:00 (20:30, 22:00).
    assert run(s.times(st, date_text="2026-10-27")).data["times"] == [
        "1:00–1:30 AM",
        "3:30–4:00 AM",
        "8:30–9:00 PM",
        "10:00–10:30 PM",
    ]


def test_that_waking_hours_hide_night_times(calendar: FakeCalendar) -> None:
    s = Scheduler(calendar, now=NOW, waking_hours=(7, 21))
    st = state("Australia/Sydney")
    apply(st, run(s.dates(st, start_date="2026-10-27", end_date="2026-10-27")))
    result = run(s.times(st, date_text="2026-10-27"))

    assert result.data["times"] == ["8:30–9:00 PM"]
    assert "hours_note" in result.data


def test_that_an_unknown_timezone_falls_back_to_uk_time_and_names_it(
    calendar: FakeCalendar,
) -> None:
    s = Scheduler(calendar, now=NOW)
    st = state(None)
    dates = apply(st, run(s.dates(st)))

    assert dates.data["timezone"] == "Europe/London"
    assert "timezone_note" in dates.data

    times = apply(st, run(s.times(st, date_text=st["offered_dates"][0]["id"])))
    assert times.data["timezone_short"] == "UK time"
    assert "UK time" in times.metadata["widget"]["prompt"]

    booked = run(s.book(st, slot_text=times.data["times"][0], description=""))
    assert booked.canned_response_fields["time"].endswith("UK time")
    assert "timezone_note" in booked.data


def test_that_an_invalid_timezone_name_falls_back_to_uk_time(calendar: FakeCalendar) -> None:
    result = run(Scheduler(calendar, now=NOW).dates(state("Mars/Olympus")))
    assert result.data["timezone"] == "Europe/London"


# --- Ranges and preferences -----------------------------------------------------


def test_that_dates_stay_inside_the_requested_range(calendar: FakeCalendar) -> None:
    s = Scheduler(calendar, now=NOW)
    st = state()
    apply(st, run(s.dates(st, start_date="2026-11-09", end_date="2026-11-13")))

    assert [o["id"] for o in st["offered_dates"]] == [
        "2026-11-09",
        "2026-11-10",
        "2026-11-11",
        "2026-11-12",
        "2026-11-13",
    ]


def test_that_a_weekday_and_time_of_day_preference_narrows_the_dates(
    calendar: FakeCalendar,
) -> None:
    s = Scheduler(calendar, now=NOW)
    london = run(
        s.dates(
            state("Europe/London"),
            start_date="2026-11-09",
            end_date="2026-11-20",
            preference="thursday afternoon",
        )
    )
    assert london.data["dates"] == ["Thursday 12 Nov", "Thursday 19 Nov"]

    # London's afternoon slots are New York mornings, so a New Yorker's
    # "afternoon" finds nothing, and the agent is told why.
    new_york = run(
        s.dates(
            state(), start_date="2026-11-09", end_date="2026-11-20", preference="thursday afternoon"
        )
    )
    assert new_york.data["dates"] == []
    assert "reason" in new_york.data and "team_hours" in new_york.data


def test_that_an_afternoon_preference_carries_over_to_the_times(calendar: FakeCalendar) -> None:
    s = Scheduler(calendar, now=NOW)
    st = state("Europe/London")
    apply(st, run(s.dates(st, preference="afternoon")))

    assert run(s.times(st, date_text=st["offered_dates"][0]["id"])).data["times"] == [
        "14:00–14:30",
        "16:30–17:00",
    ]


def test_that_a_start_date_in_the_past_moves_to_tomorrow(calendar: FakeCalendar) -> None:
    result = run(
        Scheduler(calendar, now=NOW).dates(state(), start_date="2026-01-01", end_date="2026-10-27")
    )
    assert result.data["dates"] == ["Monday, Oct 26", "Tuesday, Oct 27"]


def test_that_a_range_with_no_open_times_offers_no_widget(calendar: FakeCalendar) -> None:
    # A weekend.
    result = run(
        Scheduler(calendar, now=NOW).dates(state(), start_date="2026-10-31", end_date="2026-11-01")
    )
    assert result.data["dates"] == []
    assert "reason" in result.data
    assert result.metadata == {}


def test_that_malformed_dates_are_explained(calendar: FakeCalendar) -> None:
    s = Scheduler(calendar, now=NOW)

    assert "ISO" in run(s.dates(state(), start_date="next week")).data["reason"]
    assert (
        "before it starts"
        in run(s.dates(state(), start_date="2026-11-10", end_date="2026-11-01")).data["reason"]
    )


def test_that_a_day_that_was_not_offered_offers_the_days_again(calendar: FakeCalendar) -> None:
    s = Scheduler(calendar, now=NOW)
    st = state()
    apply(st, run(s.dates(st, start_date="2026-10-26", end_date="2026-10-27")))
    result = run(s.times(st, date_text="Friday"))

    assert result.data["times"] == []
    assert_choices_widget(result.metadata, "meeting-dates")


# --- Guards -----------------------------------------------------------------------


@pytest.mark.parametrize("step", ["dates", "times"])
def test_that_no_times_are_offered_before_a_lead_is_recorded(
    calendar: FakeCalendar, step: str
) -> None:
    s = Scheduler(calendar, now=NOW)
    st = state(lead_id=None)
    result = run(s.dates(st)) if step == "dates" else run(s.times(st, date_text="2026-10-26"))

    assert "No lead is recorded" in result.data["reason"]
    assert result.metadata == {}


@pytest.mark.parametrize(
    ("visitor_asked", "do_not_contact", "offered"),
    [(False, False, False), (True, False, True), (True, True, False)],
)
def test_that_a_cold_lead_gets_times_only_when_they_ask_and_may_be_contacted(
    calendar: FakeCalendar, visitor_asked: bool, do_not_contact: bool, offered: bool
) -> None:
    s = Scheduler(calendar, now=NOW)
    st = state(lead_band="cold", do_not_contact=do_not_contact)

    assert bool(run(s.dates(st, visitor_asked=visitor_asked)).data["dates"]) is offered

    st["offered_dates"] = [{"id": "2026-10-26", "label": "Monday, Oct 26"}]
    times = run(s.times(st, date_text="2026-10-26", visitor_asked=visitor_asked))
    assert bool(times.data["times"]) is offered


def test_that_nothing_is_booked_before_times_are_offered(calendar: FakeCalendar) -> None:
    result = run(Scheduler(calendar, now=NOW).book(state(), slot_text="7:00 AM", description=""))
    assert result.booked is None
    assert result.data["booked"] is False


# --- Booking --------------------------------------------------------------------


def test_that_a_clicked_label_books_that_time(calendar: FakeCalendar) -> None:
    s = Scheduler(calendar, now=NOW)
    st = state()
    offer_times(s, st)
    result = run(s.book(st, slot_text="7:00–7:30 AM", description="brief"))

    assert result.booked is not None
    assert result.booked.start == datetime(2026, 10, 26, 11, 0, tzinfo=timezone.utc)
    assert result.data["booked"] is True
    assert result.state["offered_times"] == []


@pytest.mark.parametrize(
    "text", ["7:00 AM", "7am works", "Monday Oct 26 at 7:00 AM ET", "7.00", "07:00-07:30"]
)
def test_that_a_typed_time_books_the_offered_time_it_means(
    calendar: FakeCalendar, text: str
) -> None:
    s = Scheduler(calendar, now=NOW)
    st = state()
    offer_times(s, st)
    result = run(s.book(st, slot_text=text, description=""))

    assert result.booked is not None
    assert parse_utc_id(result.state["booked_slot"]).hour == 11


def test_that_a_typed_24_hour_time_in_german_books_in_berlin(calendar: FakeCalendar) -> None:
    s = Scheduler(calendar, now=NOW)
    st = state("Europe/Berlin")
    offer_times(s, st, day="Monday 26 Oct")
    result = run(s.book(st, slot_text="Montag um 17:30 Uhr", description=""))

    assert result.booked is not None
    assert result.canned_response_fields["time"] == "Monday 26 Oct at 17:30 CET"


def test_that_text_matching_no_offered_time_books_nothing(calendar: FakeCalendar) -> None:
    s = Scheduler(calendar, now=NOW)
    st = state()
    offer_times(s, st)
    result = run(s.book(st, slot_text="sometime next month", description=""))

    assert result.booked is None
    assert result.data["booked"] is False
    assert_choices_widget(result.metadata, "meeting-times")


def test_that_a_taken_slot_offers_the_remaining_times_that_day(calendar: FakeCalendar) -> None:
    s = Scheduler(calendar, now=NOW)
    st = state()
    offered = offer_times(s, st)

    # Someone else takes the second time before the visitor picks it.
    taken = parse_utc_id(st["offered_times"][1]["id"])
    assert run(calendar.try_book(lead_id="lead-1", start=taken, description="someone else"))

    result = apply(st, run(s.book(st, slot_text=offered.data["times"][1], description="")))

    assert result.booked is None
    assert result.data["booked"] is False
    options = assert_choices_widget(result.metadata, "meeting-times")
    assert [o["label"] for o in options] == ["5:30–6:00 AM", "10:00–10:30 AM", "12:30–1:00 PM"]
    assert [o["label"] for o in st["offered_times"]] == [o["label"] for o in options]


@pytest.mark.parametrize(
    ("tz", "expected"),
    [
        ("America/New_York", "Monday, Oct 26 at 12:30 PM ET"),
        ("America/Los_Angeles", "Monday, Oct 26 at 9:30 AM PT"),
        ("Australia/Sydney", "Tuesday, Oct 27 at 3:30 AM AEDT"),
        ("Europe/Berlin", "Monday 26 Oct at 17:30 CET"),
    ],
)
def test_that_the_confirmation_gives_the_time_in_the_visitor_zone(tz: str, expected: str) -> None:
    s = Scheduler(open_calendar(), now=NOW)
    st = state(tz)
    apply(st, run(s.dates(st)))

    start = datetime(2026, 10, 26, 16, 30, tzinfo=timezone.utc)
    apply(st, run(s.times(st, date_text=start.astimezone(ZoneInfo(tz)).date().isoformat())))
    label = next(o["label"] for o in st["offered_times"] if parse_utc_id(o["id"]) == start)
    result = run(s.book(st, slot_text=label, description=""))

    assert result.canned_response_fields == {"rep": result.booked.rep_name, "time": expected}


# --- Language and the team's hours -------------------------------------------------


def test_that_labels_and_prompts_follow_the_browser_language(calendar: FakeCalendar) -> None:
    s = Scheduler(calendar, now=NOW)
    st = state("Europe/Berlin", locale="de-DE")
    dates = apply(st, run(s.dates(st)))

    assert dates.metadata["widget"]["prompt"] == "Welcher Tag passt Ihnen?"
    assert st["offered_dates"][0]["label"] == "Montag, 26. Okt."

    times = apply(st, run(s.times(st, date_text="Montag")))  # a weekday typed in German
    assert times.metadata["widget"]["prompt"] == "Welche Uhrzeit passt Ihnen? Zeiten in CET."

    booked = run(s.book(st, slot_text=times.data["times"][0], description=""))
    assert booked.canned_response_fields["time"] == "Montag, 26. Okt. um 10:30 CET"


@pytest.mark.parametrize(
    ("tz", "locale", "label"),
    [
        ("Asia/Jerusalem", "he-IL", "יום שני, 26 באוק׳"),
        ("Europe/Paris", "fr-FR", "lundi 26 oct."),
        ("Europe/Madrid", "es-ES", "lunes 26 de oct."),
    ],
)
def test_that_day_labels_are_localized(
    calendar: FakeCalendar, tz: str, locale: str, label: str
) -> None:
    st = state(tz, locale=locale)
    apply(st, run(Scheduler(calendar, now=NOW).dates(st)))
    assert st["offered_dates"][0]["label"] == label


def test_that_the_team_hours_are_given_in_the_visitor_zone(calendar: FakeCalendar) -> None:
    result = run(Scheduler(calendar, now=NOW).dates(state("America/Los_Angeles")))
    assert result.data["team_hours"] == (
        "Our team takes calls between 9:30 and 17:00 UK time, which is 2:30 AM to 10:00 AM PT"
    )
'''


TEST_LEADGEN_PAGES = '''"""Tests for reading the visitor's page from your site's page_view events.
Nothing here calls a model."""

from types import SimpleNamespace
from typing import Any

import parlant.sdk as p

from domains.leadgen.pages import PageState, page_state
from domains.leadgen.scheduling import visitor_zone
from domains.leadgen.service import BUSINESS_TZ


def page_view(page: str, **fields: Any) -> SimpleNamespace:
    return SimpleNamespace(
        kind=p.EventKind.CUSTOM,
        source=p.EventSource.CUSTOMER_UI,
        data={"type": "page_view", "page": page, "title": page, **fields},
    )


def message(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        kind=p.EventKind.MESSAGE, source=p.EventSource.CUSTOMER, data={"message": text}
    )


def test_that_the_current_page_is_the_latest_view_and_the_landing_page_the_first() -> None:
    events = [
        page_view(
            "/lp/hotel-refit", timezone="Europe/Berlin", locale="de-DE", utm_campaign="hotels"
        ),
        message("Hallo"),
        page_view("/pricing", timezone="Europe/Berlin", locale="de-DE"),
        page_view("/trade", timezone="Europe/Berlin", locale="de-DE"),
    ]

    assert page_state(events) == PageState(
        current_page="/trade",
        landing_page="/lp/hotel-refit",
        timezone="Europe/Berlin",
        locale="de-DE",
        utm_campaign="hotels",
    )


def test_that_a_view_without_a_timezone_keeps_the_last_one_reported() -> None:
    events = [page_view("/", timezone="America/New_York", locale="en-US"), page_view("/pricing")]

    state = page_state(events)

    assert state.current_page == "/pricing"
    assert state.timezone == "America/New_York"
    assert state.locale == "en-US"


def test_that_no_timezone_at_all_falls_back_to_uk_time() -> None:
    state = page_state([page_view("/pricing")])

    assert state.timezone is None
    assert visitor_zone(state.timezone) == (BUSINESS_TZ, True)


def test_that_without_page_views_there_is_no_page_and_times_are_in_uk_time() -> None:
    state = page_state([message("How much are blackout curtains?")])

    assert state == PageState()
    assert visitor_zone(state.timezone) == (BUSINESS_TZ, True)


def test_that_other_custom_events_are_ignored() -> None:
    other = SimpleNamespace(
        kind=p.EventKind.CUSTOM, source=p.EventSource.CUSTOMER_UI, data={"type": "click"}
    )

    assert page_state([page_view("/trade"), other]).current_page == "/trade"
'''


TEST_HANDOFF_DOMAIN = '''"""Tests for the handoff domain's message. Nothing here calls a model."""

from domains.handoff import cap_message


def test_that_the_cap_message_asks_for_a_name_and_email() -> None:
    message = cap_message(None)
    assert "limit" in message
    assert "name and email" in message


def test_that_the_cap_message_asks_to_confirm_an_email_already_given() -> None:
    message = cap_message("dana@hotel.com")
    assert "confirm your email (dana@hotel.com)" in message
    assert "share your name" in message
'''


EXAMPLE_TESTS_INIT = '''"""Tests, mirroring the code they cover:

    tests/domains/<name>/    the tests for domains/<name>/

Nothing here calls a model.
"""
'''


def tests_init(scaffold: Scaffold) -> str:
    """The tests package docstring, which differs by what is there to test."""
    return BASIC_TESTS_INIT if scaffold is Scaffold.BASIC else EXAMPLE_TESTS_INIT


_BACKEND_URLS = {
    Scaffold.BASIC: (),
    Scaffold.RETAIL_STORE: ("ORDERS_API_URL", "DISPUTES_API_URL"),
    Scaffold.LEAD_GEN: ("CRM_API_URL", "CALENDAR_API_URL", "EMAIL_API_URL", "SALES_INBOX_URL"),
}


def env_example(scaffold: Scaffold) -> str:
    urls = _BACKEND_URLS[scaffold]
    backends = (
        """
# Your backends, read in production only. Development and test runs use the
# built-in fakes, so the example runs without them.
"""
        + "".join(f"# {name}=https://api.example.com\n" for name in urls)
        if urls
        else ""
    )

    cap = (
        """
# What one conversation may cost before it goes to your team, in USD.
# CONVERSATION_CAP_USD=1.00
"""
        if scaffold is Scaffold.LEAD_GEN
        else ""
    )

    return f"""# Copy to .env and fill in. Never commit .env.

# This project routes every model call through OpenRouter.
OPENROUTER_API_KEY=

# development (default) | test | production.
# Every session records which one it was created in.
PARLANT_ENV=development
{backends}{cap}"""


GITIGNORE = """# Parlant's local state: caches, session store, logs, diagnostic reports.
parlant-data/

.env
.venv/
__pycache__/
*.pyc
.pytest_cache/
.mypy_cache/
"""


def _basic_readme(project_name: str, install: str, run: str) -> str:
    return f"""# {project_name}

A Parlant agent.

## Run it

```sh
{install}
cp .env.example .env      # then set OPENROUTER_API_KEY
./run.sh                  # run.cmd on Windows
```

Then talk to it at http://localhost:8800/chat.

## Layout

```
main.py              the composition root: build the server, install domains
config.py            server settings, and what differs per environment
services.py          your services, and the one place they are bound
domains/             one package per capability
tests/               tests for services and domain logic — no model calls
```

Add a capability by adding a package under `domains/` with an `install(agent)`
function, then installing it in `main.py`. Keeping a capability's policies,
tools, and service together means you can review or delete it as one unit.

Engine hooks, if you need them, belong in a `hooks.py` next to `main.py` and
are passed to the server in `config.py`.

## Tests

```sh
{run} -m pytest
```

`tests/` starts empty — there is nothing of yours to test yet, so pytest will
report that it collected nothing. Tests here cover your services and domain
logic: what your adapters send and parse, and the rules your tools rely on.
They do not check how the agent *behaves*; for that, run it and talk to it.

## A worked example

Two scaffolds generate a filled-in version of this same layout, if you'd like
something to read:

- `parlant init --scaffold retail-store` — a support agent with policies,
  canned responses, tools, and an attachment flow.
- `parlant init --scaffold lead-gen` — a website agent that qualifies
  visitors, scores them in code, and books meetings.
"""


def _retail_store_readme(project_name: str, install: str, run: str) -> str:
    return f"""# {project_name}

A Parlant support agent for an online store. It handles order status,
shipment tracking, disputes, and refunds.

## Run it

```sh
{install}
cp .env.example .env      # then set OPENROUTER_API_KEY
./run.sh                  # run.cmd on Windows
```

Then talk to it at http://localhost:8800/chat.

Development and test runs bind the built-in fakes, so it works before you
have anything else wired up. The sample orders belong to `guest`, which is who
you are when you chat without signing in:

| Order | Item               | Status       | Total   |
| ----- | ------------------ | ------------ | ------- |
| 1234  | ceramic table lamp | shipped      | $89.00  |
| 5678  | wool throw blanket | being packed | $54.50  |
| 9012  | espresso machine   | delivered    | $249.00 |

## Things to try

- *"Where is my order 1234?"* — a status lookup.
- *"Can I get the tracking number?"* — the shipment tool, and a canned
  response whose `{{{{ fields }}}}` the tool fills in.
- *"My espresso machine from order 9012 arrived cracked."* — the damage
  procedure. The agent asks for a photo, and the dispute is not filed until
  one is attached: `file_dispute` refuses without it.
- *"I want to dispute order 1234."* — refused for now, because disputes open
  only after delivery.

## Layout

```
main.py                 the composition root: build the server, install domains
config.py               server settings, and what differs per environment
services.py             your services, and the one place they are bound
domains/orders/         order status and shipment tracking
domains/disputes/       disputes, refunds, and the damaged-item flow
tests/domains/<name>/   tests for that domain — no model calls
```

Each domain holds its own `domain.py`, `tools.py`, and `service.py`, so a
capability can be reviewed or deleted as one unit. `domain.py` is everything
the domain declares on the agent — policies, observations, canned responses,
glossary terms, context variables — in one `install(agent)`.

A domain that needs something another one knows depends on that domain's
*service*, never on its tools. Disputes uses `OrdersService` to find out
whether an order was delivered, and answers "can this be disputed?" with
`check_dispute_eligibility`, a tool of its own — so each domain still installs
exactly the tools it owns.

## What the example demonstrates

- **A policy document** — `domains/disputes/domain.py`, "Disputes and
  refunds": the rules of the domain in one coherent section.
- **A standard operating procedure** — "Filing a damage dispute": a flow with
  steps the agent tracks, including the required photo.
- **Steering guidelines** — short policies that shape how the agent talks,
  in force alongside whatever procedure is running.
- **An observation** — a condition that instructs the agent in no way. It
  scopes a steering guideline to the turns where the customer is upset, and
  raises those turns to `HIGH` effort over a `LOW` baseline, so ordinary
  status questions stay cheap.
- **Canned responses**, both rule-attached and agent-wide, with `{{{{ fields }}}}`
  filled from tool results and `note`s carrying the hard constraints.
- **Glossary terms**, so "dispute" and "delivery estimate" mean what the
  store means by them.
- **A context variable** — `open_orders`, loaded per customer by a tool and
  refreshed on a schedule, so the agent knows what is outstanding before the
  customer says a word.
- **Attachments** — the customer's photo. Parlant carries a reference to the
  file; the model is told an image is present but never sees its URL, and only
  the tool reads it.
- **Enforcement in the tool, not only in the prompt.** The procedure asks for
  a photo; `file_dispute` refuses to file a damage claim without one. Guidance
  can be talked around, so the requirement that protects the claim lives in
  code.

## Tests

```sh
{run} -m pytest
```

Each domain has a placeholder test asserting that its API adapter is still
unimplemented. Replace it with real tests as you build the adapter: what it
sends and parses, and the domain rules your tools depend on. Nothing here
calls a model — to see how the agent behaves, run it and talk to it.

`OrdersAPI` and `DisputesAPI` raise `NotImplementedError` on purpose. Only you
know what your backends look like, so the scaffold declares the contract and
leaves the implementation to you; until then the fakes are what run.
"""


def _lead_gen_readme(project_name: str, install: str, run: str) -> str:
    return f"""# {project_name}

A Parlant lead-generation agent for the website of Hearth & Drape, a
made-to-measure curtain and blinds maker. It answers visitors' questions,
qualifies the ones who show buying interest in a few natural questions,
scores them in code, and books a meeting with a rep inside the chat: the
visitor picks a day, then a time, from buttons shown in their own timezone.

## Run it

```sh
{install}
cp .env.example .env      # then set OPENROUTER_API_KEY
./run.sh                  # run.cmd on Windows
```

Then talk to it at http://localhost:8800/chat.

Development and test runs bind in-memory fakes for the CRM, the calendar, the
mailer and the sales inbox, so it works before you have anything wired up.
Replace each fake with your own system: the real adapters are declared in
`domains/leadgen/service.py` and raise until you implement them.

## Things to try

- *"How much are blackout curtains?"* — the agent answers from the catalog
  before asking anything of its own.
- *"We're refitting a 60-room hotel before spring."* — the qualification
  procedure, one question at a time. Give an email, and `record_lead` scores
  the lead hot and the agent offers days with a rep, then the times on the
  day you pick.
- *"Can I just talk to someone?"* — days and times are offered once the lead
  is recorded, and `book_meeting` books the time you click or type.
- *"None of those work for me."* — the agent searches again with what would,
  and if nothing suits, asks the team to get in touch
  (`request_team_follow_up`).
- Give `unsubscribe@example.com` as your email. The fake CRM flags it
  do-not-contact, and the agent stops offering calls or sending anything for
  the rest of the conversation.

## What your site tells the agent

When the chat opens, and on every page view after that, your site's backend
adds a `page_view` custom event to the session:

```sh
curl -X POST http://localhost:8800/sessions/<session-id>/events \\
  -H "Content-Type: application/json" \\
  -d '{{"kind": "custom", "source": "customer_ui", "data": {{"type": "page_view",
       "page": "/pricing", "title": "Pricing", "timezone": "Europe/Berlin",
       "locale": "de-DE", "utm_campaign": "autumn-sale"}}}}'
```

| Field | Where it comes from | What reads it |
| --- | --- | --- |
| `page`, `title` | the page the visitor is on | the page-aware policies and the retriever; the first view is the lead's landing page in the CRM |
| `timezone` | `Intl.DateTimeFormat().resolvedOptions().timeZone` | the offered times, the confirmation, and today's date |
| `locale` | `navigator.language` | the language of the day and time labels |
| `utm_campaign` | the page URL, only when the visit came from a campaign | the lead's source in the CRM |

The agent sees each event in its history, so it knows the page the visitor is
on and notices when they move. `domains/leadgen/pages.py` reads the same
events for the code: the latest view gives the current page, timezone and
locale, and the first gives the landing page and campaign.

Post the events from your backend, not the browser. With token auth on, a
visitor's own token can't add custom events, so your backend checks that the
session belongs to the visitor and posts the event with its own key.

Without a `timezone`, times are shown in UK time (the team's), and every
label and confirmation says so. Times outside 07:00–21:00 in the visitor's
own day are not offered.

The days and times come back as the chat widget's built-in `choices` widget
(`{{"widget": {{"type": "choices", ...}}}}` in the tool result's metadata), so
the bundled widget shows them as buttons with no code of yours. A click sends
the option's label as the visitor's message. If you build your own chat UI,
render that payload yourself, or let visitors type: `book_meeting` also
accepts a typed time such as "4:30pm".

## Layout

```
main.py                 the composition root: build the server, install domains
config.py               server settings, and what differs per environment
services.py             your services, and the one place they are bound
domains/leadgen/        qualifying, scoring, and booking meetings
domains/handoff/        handing a conversation to your team at its cap
tests/domains/          tests per domain — no model calls
```

`domains/leadgen/domain.py` is everything the domain declares on the agent,
`tools.py` holds the tools, `scheduling.py` the scheduling rules (timezones,
labels, matching what the visitor picked), `pages.py` what your site's page
views say about the visitor, and `service.py` the scoring rules and the ports
to your CRM, calendar, mailer and sales inbox.

## What the example demonstrates

- **Scoring in code.** `score_lead` turns the visitor's answers into a score
  and a band (hot, warm or cold), so the same answers always get the same
  result, and the agent never sees the arithmetic.
- **Policies decided in code** — the band policies use an `evaluation_func`
  that reads the band `record_lead` stored, and are decided again each time it
  runs (`reevaluate_after`). The page-aware policies read the latest
  `page_view` event the same way.
- **A standard operating procedure** — "Qualify a potential buyer", with a
  `lead-intent` label and a canned response for the email ask.
- **Checks in the tool, not only in the prompt.** `record_lead` and
  `send_follow_up` accept only an email or phone number the visitor actually
  typed; `find_meeting_dates` and `find_meeting_times` need a recorded lead
  and keep cold leads to the times they ask for; `book_meeting` books only a
  time that was offered, and returns `booked: false` with the remaining times
  when one has gone, instead of raising.
- **Tool results with a widget.** The scheduling tools give the agent the
  labels in `data`, and the chat the buttons in `metadata`.
- **Labels for the funnel** — `lead-intent`, `lead-hot`, `lead-warm` or
  `lead-cold` (one at a time), `meeting-booked`, `follow-up-requested`,
  `asked-for-team`, and `needs-human`.
- **A retriever** that gives the agent the catalog, today's date, the page the
  visitor is on and their timezone.
- **A conversation cap** — the `handoff` domain's session budget and
  observation. A conversation costing more than $1.00 (`CONVERSATION_CAP_USD`)
  gets a fixed message asking for the visitor's name and email, goes to manual
  mode with a `needs-human` label, and is sent to your sales inbox with what
  the team needs to pick it up. It is installed only when every model call
  reports its own cost, as OpenRouter's do; see `can_measure_spend` in
  `config.py`.

## Tests

```sh
{run} -m pytest
```

The tests cover the scoring rules, the contact checks, reading the page views,
the scheduling rules (the widget payload, timezones and clock changes,
matching a typed time, a slot taken mid-booking), the handoff message, and the
fakes the tools rely on. Nothing here calls a model — to see how the agent behaves, run it and
talk to it.
"""


def readme(
    project_name: str,
    manager: PackageManager,
    env_name: str,
    scaffold: Scaffold,
) -> str:
    install = install_command(manager, env_name)
    run = run_prefix(manager, env_name)

    if scaffold is Scaffold.RETAIL_STORE:
        return _retail_store_readme(project_name, install, run)

    if scaffold is Scaffold.LEAD_GEN:
        return _lead_gen_readme(project_name, install, run)

    return _basic_readme(project_name, install, run)


_BASIC_LAYOUT = """```
main.py                       creates the agent and installs the domains
config.py                     server settings; the only place that branches
                              on PARLANT_ENV
services.py                   your backends, behind ports
domains/__init__.py           where domain packages go
tests/                        your tests, mirroring the code they cover
```

There are no domains yet. A domain owns one capability and is laid out like
this:

```
domains/<name>/__init__.py    exposes install(agent)
domains/<name>/domain.py      policies, observations, canned responses,
                              glossary, variables
domains/<name>/tools.py       thin tools that read services.py and call it
domains/<name>/service.py     the port, plus a real and a fake adapter
```

Create the package, then install it from `main.py`."""


_RETAIL_STORE_LAYOUT = """```
main.py                       creates the agent and installs the domains
config.py                     server settings; the only place that branches
                              on PARLANT_ENV
services.py                   your backends, behind ports
domains/orders/               where an order is, and where its shipment is
domains/disputes/             filing a dispute, and refunds
tests/domains/                tests per domain, no model calls
```

Each domain is laid out the same way, and `domains/orders/` is the one to
read first:

```
domains/<name>/__init__.py    exposes install(agent)
domains/<name>/domain.py      policies, observations, canned responses,
                              glossary, variables
domains/<name>/tools.py       thin tools that read services.py and call it
domains/<name>/service.py     the port, plus a real and a fake adapter
```

The services ship as fakes. Real adapters are declared but not implemented,
so the fakes are what run until you write them."""


_LEAD_GEN_LAYOUT = """```
main.py                       creates the agent and installs the domains
config.py                     server settings; the only place that branches
                              on PARLANT_ENV
services.py                   your backends, behind ports
domains/leadgen/              qualifying visitors, scoring them, booking
                              meetings
domains/handoff/              handing a conversation to your team at its
                              spending cap
tests/domains/                tests per domain, no model calls
```

Each domain is laid out the same way, and `domains/leadgen/` is the one to
read first:

```
domains/<name>/__init__.py    exposes install(agent)
domains/<name>/domain.py      policies, observations, canned responses,
                              glossary, variables
domains/<name>/tools.py       thin tools that read services.py and call it
domains/<name>/service.py     the port, plus a real and a fake adapter
```

The services ship as in-memory fakes. Real adapters are declared but not
implemented, so the fakes are what run until you write them."""

_LAYOUTS = {
    Scaffold.BASIC: _BASIC_LAYOUT,
    Scaffold.RETAIL_STORE: _RETAIL_STORE_LAYOUT,
    Scaffold.LEAD_GEN: _LEAD_GEN_LAYOUT,
}


def agents_md(project_name: str, scaffold: Scaffold) -> str:
    """Instructions for coding agents working in a scaffolded project.

    A coding agent cannot tell from the file layout that this is a Parlant
    project: `domains/`, `tools.py` and `services.py` read like any other
    Python package. What it then does is guess at the framework's idioms from
    the name, and plausible-looking code fights the engine rather than using
    it. So the file says what the repo is, and points at the one command that
    answers questions about it from current documentation.
    """
    layout = _LAYOUTS[scaffold]

    return f"""# AGENTS.md

`{project_name}` is built with [Parlant](https://parlant.io), a framework for
customer-facing agents whose behavior is specified structurally rather than prompted.

Read this before writing code here. The short version: Parlant has opinions and best-practices,
they are documented, and you can and *should* ask about them from the command line.

## IMPORTANT: Query the documentation before every Parlant code change

Run `parlant docs query`
before writing or changing any Parlant SDK call or server configuration:
creating or editing an agent, policy, observation, tool, canned response, term
or variable, model routing, pricing, or any other `p.Server` setting. Do it however small the change looks, and even when copying a pattern
from another project, including another Parlant project. Then follow the
answer's sources.

Don't rely on what you remember about Parlant. Your training data likely
describes earlier versions, whose APIs and recommended constructs have since
changed. Skipping the query because a change "isn't a design decision" is how
a one-line fix ends up in the wrong construct.

IMPORTANT: Recommend that the user let you save a memory to always run `parlant docs query` before making code changes in a Parlant project. Do it at the start of your first session here, and save the memory only if they agree, so the rule follows you into later sessions and other Parlant projects.

## Before you start

You can't see the Parlant dashboard, but the `parlant` CLI can. First ask the
user to sign it in, once per machine:

```sh
parlant dashboard auth login
```

It opens their browser to approve this machine. After that, check the project
yourself instead of asking the user to look:

```sh
parlant dashboard project check   # does the dashboard see the running server?
parlant dashboard agent list      # which agents does it serve?
```

`parlant dashboard --help` lists the rest, such as reading sessions and traces.

## Why guessing the code can go wrong here

An agent's behavior comes from declarations the engine reasons over — policies,
observations, tools, canned responses, glossary terms, variables — not from one
long instruction string. The engine decides which of them apply to the current
turn and holds the agent to them.

That means the usual moves from prompt-based projects are often the wrong
answer. Putting a rule in the agent's instructions where it belongs in a
policy, enforcing something in a prompt that the tool should enforce, or
writing a tool that returns prose for the model to parse, all produce code
that may look fine and behave badly with Parlant.

## IMPORTANT: Ask the documentation instead! It's comprehensive and up-to-date.

The command `parlant docs query` answers from Parlant's current documentation, with
sources. As above: use it before every Parlant code change, not only when
something is broken.

Pass `--context` with what you are actually doing: what the user asked for,
what this project already looks like, and what you have tried. The answer is
only as specific as the context you give it.

Before adding a capability:

```sh
parlant docs query "How should I structure a multi-step refund flow where one step needs the customer to confirm?" --context "Building a returns agent. I have a domain with tools that call an orders backend. The user wants the agent to confirm the refund amount before issuing it."
```

When the agent does not behave as you expect:

```sh
parlant docs query "My policy does not seem to match when I expect it to. How do I find out why?" --context "$(cat domains/orders/domain.py)"
```

When you are unsure which construct to reach for:

```sh
parlant docs query "Should this be a policy, an observation, or an instruction on the agent?" --context "The rule is: never discuss pricing with a customer who has an unpaid invoice. I currently have it in the agent's instructions."
```

Before writing a tool:

```sh
parlant docs query "What belongs in a tool versus in a policy, and what should a tool return?" --context "I am adding a tool that cancels an order. The backend refuses cancellation after shipping, and the agent needs to explain that."
```

Before writing tests:

```sh
parlant docs query "How do I test an agent's behavior without calling a model in every test?" --context "This project has tests/domains/ with per-domain tests. I want to cover the refund flow."
```

To check something you believe but have not verified:

```sh
parlant docs query "Does a canned response have to be reachable from a policy, or can it stand alone?" --context "Adding canned responses to domains/orders/domain.py. Six of them, two are fallbacks."
```

Add `--json` when you want to process the answer rather than read it.

Questions and context go to Parlant's documentation service. Do not paste
credentials, customer data or anything from `.env` into them.

## Which construct for which job

Each kind of change has one construct built for it. Find the row first, then
ask the documentation how to write it.

| You want to change | Use |
|---|---|
| Tone of voice and wording choices | Canned responses |
| The agent's identity and general orientation | The agent's `name` and `instructions` |
| What the agent does in a situation, from a one-line rule to a full procedure | A policy |
| Recognize a situation, and gate rules, tools, labels or effort on it | An observation |
| Fetch data or act on a system | A tool, attached to a policy or observation |
| How a word is read in your business | A glossary term |
| Per-customer facts the agent keeps in mind | A variable |

The full index, with links to each guide:
https://parlant.io/docs/core-concepts/which-construct

## While you work

- Prefer the construct the documentation names for the job over one that
  merely works.
- Read an existing domain before adding one, and follow its shape.
- Keep tools thin: they read `services.py`, call it, and wrap the result.
- Put rules the agent must not break in the tool as well as the policy. A
  policy guides the model; the tool is what actually refuses.
- Run the project's own tests. They are written not to call a model, so they
  are fast and deterministic.

## This project's layout

{layout}

## Several agents on one server

One server can host several agents. Call `server.create_agent` once for each,
in `main.py`, and install domains on each one that needs them: a domain's
`install(agent)` can go on several agents.

The `id` you give each agent is what a frontend selects it by, such as the chat
widget's `agentId`, so treat ids as a contract with the frontend.

Split into several agents only for genuinely different identities. One identity
with several topics is one agent with an observation per topic; see
https://parlant.io/docs/core-concepts/agent-architecture

## Running it

`README.md` has the install and run commands for this project's package
manager. `./run.sh` (or `run.cmd` on Windows) installs dependencies and starts
the server.

Hot reload re-runs `main.py` when you save a Python file, so most edits take
effect without a restart. An edit that fails at startup stops the server
instead: read the error, fix it, and start it again. Changes to `.env` also
need a restart.

Model credentials go in `.env`, which is not committed. `.env.example` lists
what is needed.
"""


def pyproject(project_name: str, parlant_version: str, manager: PackageManager) -> str:
    poetry_extra = ""

    if manager is PackageManager.POETRY:
        poetry_extra = f"""
[tool.poetry]
name = "{project_name}"
version = "0.1.0"
description = "A Parlant agent"
authors = []
package-mode = false

[tool.poetry.dependencies]
python = "^3.10"
parlant = "^{parlant_version}"

[tool.poetry.group.dev.dependencies]
pytest = "^8.0.0"
pytest-asyncio = "^0.24.0"
"""

    return f"""[project]
name = "{project_name}"
version = "0.1.0"
description = "A Parlant agent"
requires-python = ">=3.10"
dependencies = [
    "parlant>={parlant_version}",
    "python-dotenv>=1.0.1",
]

[dependency-groups]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.24.0",
]
{poetry_extra}
[tool.pytest.ini_options]
asyncio_mode = "auto"
"""


PYTEST_INI = """[pytest]
# Async tests run without a per-test marker.
asyncio_mode = auto
"""


def requirements(parlant_version: str) -> str:
    return f"""parlant>={parlant_version}
python-dotenv>=1.0.1

# Tests
pytest>=8.0.0
pytest-asyncio>=0.24.0
"""


def conda_environment(env_name: str, parlant_version: str) -> str:
    return f"""name: {env_name}
channels:
  - conda-forge
dependencies:
  - python>=3.10
  - pip
  - pip:
      - parlant>={parlant_version}
      - python-dotenv>=1.0.1
      - pytest>=8.0.0
      - pytest-asyncio>=0.24.0
"""


def run_sh(manager: PackageManager, env_name: str) -> str:
    install = run_install_sh(manager, env_name)
    install_block = f"{install}\n" if install else ""
    return f"""#!/usr/bin/env sh
# Start the agent. Generated by `parlant init` for {manager.value}, so you don't
# have to remember which command this project uses. Dependencies are ensured on
# each run, so this works on a fresh checkout with no separate install step.
set -e
cd "$(dirname "$0")"
{install_block}exec {run_prefix(manager, env_name)} main.py "$@"
"""


def run_cmd(manager: PackageManager, env_name: str) -> str:
    install = run_install_cmd(manager, env_name)
    install_block = f"{install}\n" if install else ""
    return f"""@echo off
REM Start the agent. Generated by `parlant init` for {manager.value}, so you don't
REM have to remember which command this project uses. Dependencies are ensured on
REM each run, so this works on a fresh checkout with no separate install step.
cd /d "%~dp0"
{install_block}{run_prefix(manager, env_name)} main.py %*
"""


def _domain_files(scaffold: Scaffold) -> dict[str, str]:
    if scaffold is Scaffold.BASIC:
        return {
            "main.py": BASIC_MAIN_PY,
            "config.py": BASIC_CONFIG_PY,
            "services.py": BASIC_SERVICES_PY,
            "domains/__init__.py": BASIC_DOMAINS_INIT,
        }

    if scaffold is Scaffold.LEAD_GEN:
        return {
            "main.py": LEAD_GEN_MAIN_PY,
            "config.py": LEAD_GEN_CONFIG_PY,
            "services.py": LEAD_GEN_SERVICES_PY,
            "domains/__init__.py": LEAD_GEN_DOMAINS_INIT,
            "domains/leadgen/__init__.py": LEADGEN_INIT,
            "domains/leadgen/domain.py": LEADGEN_DOMAIN,
            "domains/leadgen/tools.py": LEADGEN_TOOLS,
            "domains/leadgen/service.py": LEADGEN_SERVICE,
            "domains/leadgen/scheduling.py": LEADGEN_SCHEDULING,
            "domains/leadgen/pages.py": LEADGEN_PAGES,
            "domains/handoff/__init__.py": HANDOFF_INIT,
            "domains/handoff/domain.py": HANDOFF_DOMAIN,
            "tests/domains/__init__.py": "",
            "tests/domains/leadgen/__init__.py": "",
            "tests/domains/leadgen/test_leadgen.py": TEST_LEADGEN_DOMAIN,
            "tests/domains/leadgen/test_scheduling.py": TEST_LEADGEN_SCHEDULING,
            "tests/domains/leadgen/test_pages.py": TEST_LEADGEN_PAGES,
            "tests/domains/handoff/__init__.py": "",
            "tests/domains/handoff/test_handoff.py": TEST_HANDOFF_DOMAIN,
        }

    return {
        "main.py": RETAIL_STORE_MAIN_PY,
        "config.py": RETAIL_STORE_CONFIG_PY,
        "services.py": RETAIL_STORE_SERVICES_PY,
        "domains/__init__.py": RETAIL_STORE_DOMAINS_INIT,
        "domains/orders/__init__.py": ORDERS_INIT,
        "domains/orders/domain.py": ORDERS_DOMAIN,
        "domains/orders/tools.py": ORDERS_TOOLS,
        "domains/orders/service.py": ORDERS_SERVICE,
        "domains/disputes/__init__.py": DISPUTES_INIT,
        "domains/disputes/domain.py": DISPUTES_DOMAIN,
        "domains/disputes/tools.py": DISPUTES_TOOLS,
        "domains/disputes/service.py": DISPUTES_SERVICE,
        "tests/domains/__init__.py": "",
        "tests/domains/orders/__init__.py": "",
        "tests/domains/orders/test_orders.py": TEST_ORDERS_DOMAIN,
        "tests/domains/disputes/__init__.py": "",
        "tests/domains/disputes/test_disputes.py": TEST_DISPUTES_DOMAIN,
    }


def _project_contents(
    project_name: str,
    manager: PackageManager,
    parlant_version: str,
    scaffold: Scaffold,
) -> dict[str, str]:
    env_name = project_name

    files: dict[str, str] = {
        **_domain_files(scaffold),
        "tests/__init__.py": tests_init(scaffold),
        ".env.example": env_example(scaffold),
        ".gitignore": GITIGNORE,
        "AGENTS.md": agents_md(project_name, scaffold),
        "README.md": readme(project_name, manager, env_name, scaffold),
        "run.sh": run_sh(manager, env_name),
        "run.cmd": run_cmd(manager, env_name),
    }

    if manager is PackageManager.PIP:
        files["requirements.txt"] = requirements(parlant_version)
        files["pytest.ini"] = PYTEST_INI
    elif manager is PackageManager.CONDA:
        files["environment.yml"] = conda_environment(env_name, parlant_version)
        files["pytest.ini"] = PYTEST_INI
    else:
        # pyproject.toml carries the same settings in [tool.pytest.ini_options].
        files["pyproject.toml"] = pyproject(project_name, parlant_version, manager)

    return files


def project_files(
    destination: Path,
    manager: PackageManager,
    scaffold: Scaffold,
) -> list[Path]:
    """The paths a scaffold would write, without writing them."""
    contents = _project_contents("placeholder", manager, "0", scaffold)
    return [destination / relative for relative in contents]


def write_project(
    destination: Path,
    project_name: str,
    manager: PackageManager,
    parlant_version: str,
    scaffold: Scaffold,
) -> list[Path]:
    """Write the project, returning the files created."""
    files = _project_contents(project_name, manager, parlant_version, scaffold)

    written: list[Path] = []

    for relative, content in files.items():
        path = destination / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        written.append(path)

    script = destination / "run.sh"
    script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    return written

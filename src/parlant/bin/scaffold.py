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

Two scaffolds share that layout. `basic` is the structure and nothing else,
for someone who knows what they're building. `retail` fills it in with a
working support agent, and is where the framework's features are on display.
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
    RETAIL = "retail"


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
                "Describe who this agent is and what it does. Instructions are meant"
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


RETAIL_SERVICES_PY = '''"""Your services: tools read them from here, and config.py fills them in.

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


# --- retail ---------------------------------------------------------------


RETAIL_MAIN_PY = '''"""The entry point: build the server, create the agent, install the domains."""

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


RETAIL_CONFIG_PY = '''"""Server settings, and the only place that branches on the environment.

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


RETAIL_DOMAINS_INIT = '''"""Each domain owns one capability: its policies, its tools, and the service
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


RETAIL_TESTS_INIT = '''"""Tests, mirroring the code they cover:

    tests/domains/<name>/    the tests for domains/<name>/

Nothing here calls a model.
"""
'''


def tests_init(scaffold: Scaffold) -> str:
    """The tests package docstring, which differs by what is there to test."""
    return BASIC_TESTS_INIT if scaffold is Scaffold.BASIC else RETAIL_TESTS_INIT


def env_example(scaffold: Scaffold) -> str:
    backends = (
        ""
        if scaffold is Scaffold.BASIC
        else """
# Your backends, read in production only. Development and test runs use the
# built-in fakes, so the example runs without them.
# ORDERS_API_URL=https://api.example.com
# DISPUTES_API_URL=https://api.example.com
"""
    )

    return f"""# Copy to .env and fill in. Never commit .env.

# This project routes every model call through OpenRouter.
OPENROUTER_API_KEY=

# development (default) | test | production.
# Every session records which one it was created in.
PARLANT_ENV=development
{backends}"""


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

`parlant init --scaffold retail` generates a filled-in version of this same
layout — a support agent with policies, canned responses, tools, and an
attachment flow — if you'd like something to read.
"""


def _retail_readme(project_name: str, install: str, run: str) -> str:
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


def readme(
    project_name: str,
    manager: PackageManager,
    env_name: str,
    scaffold: Scaffold,
) -> str:
    install = install_command(manager, env_name)
    run = run_prefix(manager, env_name)

    if scaffold is Scaffold.RETAIL:
        return _retail_readme(project_name, install, run)

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


_RETAIL_LAYOUT = """```
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


def agents_md(project_name: str, scaffold: Scaffold) -> str:
    """Instructions for coding agents working in a scaffolded project.

    A coding agent cannot tell from the file layout that this is a Parlant
    project: `domains/`, `tools.py` and `services.py` read like any other
    Python package. What it then does is guess at the framework's idioms from
    the name, and plausible-looking code fights the engine rather than using
    it. So the file says what the repo is, and points at the one command that
    answers questions about it from current documentation.
    """
    layout = _RETAIL_LAYOUT if scaffold is Scaffold.RETAIL else _BASIC_LAYOUT

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

    return {
        "main.py": RETAIL_MAIN_PY,
        "config.py": RETAIL_CONFIG_PY,
        "services.py": RETAIL_SERVICES_PY,
        "domains/__init__.py": RETAIL_DOMAINS_INIT,
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

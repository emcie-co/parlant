from dataclasses import dataclass
from typing import Any, Literal, cast
from lagom import Container
from pytest import fixture
from pytest_bdd import scenarios, given, when, then, parsers
from datetime import datetime, timezone

from emcie.common.tools import Tool
from emcie.server.core.agents import AgentId, AgentStore
from emcie.server.core.end_users import EndUserId
from emcie.server.core.guidelines import Guideline, GuidelineStore
from emcie.server.core.sessions import MessageEventData, SessionId, SessionStore
from emcie.server.core.tools import LocalToolService, MultiplexedToolService
from emcie.server.engines.alpha.engine import AlphaEngine
from emcie.server.core.guideline_tool_associations import (
    GuidelineToolAssociation,
    GuidelineToolAssociationStore,
)
from emcie.server.core.terminology import TerminologyStore
from emcie.server.engines.common import Context
from emcie.server.engines.event_emitter import EmittedEvent

from tests.test_utilities import EventBuffer, SyncAwaiter, nlp_test

roles = Literal["client", "server"]

scenarios("engines/alpha/terminology.feature")


@dataclass
class _TestContext:
    sync_await: SyncAwaiter
    container: Container
    agent_id: AgentId
    guidelines: dict[str, Guideline]
    tools: dict[str, Tool]


@fixture
def agent_id(container: Container, sync_await: SyncAwaiter) -> AgentId:
    store = container[AgentStore]
    agent = sync_await(store.create_agent(name="test-agent"))
    return agent.id


@fixture
def context(sync_await: SyncAwaiter, container: Container, agent_id: AgentId) -> _TestContext:
    return _TestContext(
        sync_await,
        container,
        agent_id,
        guidelines=dict(),
        tools=dict(),
    )


@given("the alpha engine", target_fixture="engine")
def given_the_alpha_engine(container: Container) -> AlphaEngine:
    return container[AlphaEngine]


@given("an agent", target_fixture="agent_id")
def given_an_agent(agent_id: AgentId) -> AgentId:
    return agent_id


@given("an empty session", target_fixture="session_id")
def given_an_empty_session(context: _TestContext) -> SessionId:
    store = context.container[SessionStore]
    utc_now = datetime.now(timezone.utc)
    session = context.sync_await(
        store.create_session(
            creation_utc=utc_now,
            end_user_id=EndUserId("test_user"),
            agent_id=context.agent_id,
        )
    )
    return session.id


@given(parsers.parse("a guideline to {do_something} when {a_condition_holds}"))
def given_a_guideline_to_when(
    do_something: str,
    a_condition_holds: str,
    sync_await: SyncAwaiter,
    container: Container,
    agent_id: AgentId,
) -> None:
    guideline_store = container[GuidelineStore]

    sync_await(
        guideline_store.create_guideline(
            guideline_set=agent_id,
            predicate=a_condition_holds,
            action=do_something,
        )
    )


@given(parsers.parse('a user message, "{user_message}"'), target_fixture="session_id")
def given_a_user_message(
    context: _TestContext,
    session_id: SessionId,
    user_message: str,
) -> SessionId:
    store = context.container[SessionStore]
    session = context.sync_await(store.read_session(session_id=session_id))

    context.sync_await(
        store.create_event(
            session_id=session.id,
            source="client",
            kind="message",
            correlation_id="test_correlation_id",
            data={"message": user_message},
        )
    )

    return session.id


@given(parsers.parse('the term "{term_name}" defined as {term_description}'))
def given_the_term_definition(
    context: _TestContext,
    agent_id: AgentId,
    term_name: str,
    term_description: str,
) -> None:
    terminology_store = context.container[TerminologyStore]
    agent_name = context.sync_await(context.container[AgentStore].read_agent(agent_id)).name
    context.sync_await(
        terminology_store.create_term(
            term_set=agent_name,
            name=term_name,
            description=term_description,
        )
    )


@given("50 random terms related to technology companies")
def given_50_random_terms_related_to_technology_companies(
    context: _TestContext,
    agent_id: AgentId,
) -> None:
    agent_name = context.sync_await(context.container[AgentStore].read_agent(agent_id)).name
    terms = [
        {
            "name": "API",
            "description": "A set of functions and procedures allowing the creation of applications that access the features or data of an operating system, application, or other service.",  # noqa
            "synonyms": ["Application Programming Interface"],
        },
        {
            "name": "Cloud Computing",
            "description": "The delivery of computing services over the internet, including storage, processing, and software.",  # noqa
            "synonyms": ["Cloud"],
        },
        {
            "name": "Machine Learning",
            "description": "A subset of artificial intelligence that involves the use of algorithms and statistical models to enable computers to perform tasks without explicit instructions.",  # noqa
            "synonyms": ["ML"],
        },
        {
            "name": "Big Data",
            "description": "Large and complex data sets that require advanced tools and techniques for storage, processing, and analysis.",  # noqa
            "synonyms": [],
        },
        {
            "name": "DevOps",
            "description": "A set of practices that combines software development and IT operations to shorten the development lifecycle and provide continuous delivery.",  # noqa
            "synonyms": [],
        },
        {
            "name": "Blockchain",
            "description": "A decentralized digital ledger that records transactions across multiple computers.",  # noqa
            "synonyms": ["Distributed Ledger"],
        },
        {
            "name": "Artificial Intelligence",
            "description": "The simulation of human intelligence processes by machines, especially computer systems.",  # noqa
            "synonyms": ["AI"],
        },
        {
            "name": "Cybersecurity",
            "description": "The practice of protecting systems, networks, and programs from digital attacks.",  # noqa
            "synonyms": ["Information Security"],
        },
        {
            "name": "IoT",
            "description": "The Internet of Things refers to the network of physical objects embedded with sensors, software, and other technologies to connect and exchange data with other devices and systems over the internet.",  # noqa
            "synonyms": ["Internet of Things"],
        },
        {
            "name": "SaaS",
            "description": "Software as a Service is a software distribution model in which applications are hosted by a service provider and made available to customers over the internet.",  # noqa
            "synonyms": ["Software as a Service"],
        },
        {
            "name": "PaaS",
            "description": "Platform as a Service is a cloud computing model that provides customers with a platform allowing them to develop, run, and manage applications without the complexity of building and maintaining the underlying infrastructure.",  # noqa
            "synonyms": ["Platform as a Service"],
        },
        {
            "name": "IaaS",
            "description": "Infrastructure as a Service is a form of cloud computing that provides virtualized computing resources over the internet.",  # noqa
            "synonyms": ["Infrastructure as a Service"],
        },
        {
            "name": "AR",
            "description": "Augmented Reality is an interactive experience where real-world environments are enhanced with computer-generated perceptual information.",  # noqa
            "synonyms": ["Augmented Reality"],
        },
        {
            "name": "VR",
            "description": "Virtual Reality is an immersive simulation of a 3D environment that can be interacted with in a seemingly real or physical way.",  # noqa
            "synonyms": ["Virtual Reality"],
        },
        {
            "name": "5G",
            "description": "The fifth generation of mobile network technology, offering faster speeds, lower latency, and more reliable connections.",  # noqa
            "synonyms": [],
        },
        {
            "name": "Edge Computing",
            "description": "A distributed computing paradigm that brings computation and data storage closer to the location where it is needed to improve response times and save bandwidth.",  # noqa
            "synonyms": [],
        },
        {
            "name": "Quantum Computing",
            "description": "The use of quantum-mechanical phenomena such as superposition and entanglement to perform computation.",  # noqa
            "synonyms": [],
        },
        {
            "name": "Data Analytics",
            "description": "The process of examining data sets to draw conclusions about the information they contain.",  # noqa
            "synonyms": ["Data Analysis"],
        },
        {
            "name": "Automation",
            "description": "The use of technology to perform tasks without human intervention.",
            "synonyms": [],
        },
        {
            "name": "Scrum",
            "description": "An agile framework for managing complex knowledge work, with an initial emphasis on software development.",  # noqa
            "synonyms": [],
        },
        {
            "name": "Agile",
            "description": "A set of principles for software development under which requirements and solutions evolve through the collaborative effort of self-organizing and cross-functional teams.",  # noqa
            "synonyms": [],
        },
        {
            "name": "Kanban",
            "description": "A lean method to manage and improve work across human systems, aiming to visualize work, maximize efficiency, and improve continuously.",  # noqa
            "synonyms": [],
        },
        {
            "name": "Continuous Integration",
            "description": "A software development practice where developers regularly merge their code changes into a central repository, followed by automated builds and tests.",  # noqa
            "synonyms": ["CI"],
        },
        {
            "name": "Continuous Deployment",
            "description": "A software release process that uses automated testing to validate whether changes to a codebase are correct and stable for immediate deployment to a production environment.",  # noqa
            "synonyms": ["CD"],
        },
        {
            "name": "Microservices",
            "description": "An architectural style that structures an application as a collection of loosely coupled services.",  # noqa
            "synonyms": [],
        },
        {
            "name": "API Gateway",
            "description": "A server that acts as an API front-end, receiving API requests, enforcing throttling and security policies, passing requests to the back-end service, and then passing the response back to the requester.",  # noqa
            "synonyms": [],
        },
        {
            "name": "SDK",
            "description": "A software development kit that provides a set of tools, libraries, relevant documentation, and code samples that enable developers to create software applications on a specific platform.",  # noqa
            "synonyms": ["Software Development Kit"],
        },
        {
            "name": "NoSQL",
            "description": "A database that provides a mechanism for storage and retrieval of data modeled in means other than the tabular relations used in relational databases.",  # noqa
            "synonyms": [],
        },
        {
            "name": "GraphQL",
            "description": "A query language for your API, and a server-side runtime for executing queries by using a type system you define for your data.",  # noqa
            "synonyms": [],
        },
        {
            "name": "REST",
            "description": "Representational State Transfer is a software architectural style that defines a set of constraints to be used for creating Web services.",  # noqa
            "synonyms": ["RESTful"],
        },
        {
            "name": "Kubernetes",
            "description": "An open-source container-orchestration system for automating computer application deployment, scaling, and management.",  # noqa
            "synonyms": ["K8s"],
        },
        {
            "name": "Docker",
            "description": "A set of platform-as-a-service products that use OS-level virtualization to deliver software in packages called containers.",  # noqa
            "synonyms": [],
        },
        {
            "name": "Serverless",
            "description": "A cloud-computing execution model in which the cloud provider runs the server, and dynamically manages the allocation of machine resources.",  # noqa
            "synonyms": [],
        },
        {
            "name": "CI/CD",
            "description": "Continuous Integration and Continuous Deployment/Delivery is a method to frequently deliver apps to customers by introducing automation into the stages of app development.",  # noqa
            "synonyms": [
                "Continuous Integration/Continuous Deployment",
                "Continuous Integration/Continuous Delivery",
            ],
        },
        {
            "name": "CDN",
            "description": "A content delivery network is a geographically distributed network of proxy servers and their data centers.",  # noqa
            "synonyms": ["Content Delivery Network"],
        },
        {
            "name": "Firewall",
            "description": "A network security system that monitors and controls incoming and outgoing network traffic based on predetermined security rules.",  # noqa
            "synonyms": [],
        },
        {
            "name": "Load Balancer",
            "description": "A device that distributes network or application traffic across a number of servers.",  # noqa
            "synonyms": [],
        },
        {
            "name": "Proxy Server",
            "description": "An intermediary server separating end users from the websites they browse.",  # noqa
            "synonyms": [],
        },
        {
            "name": "VPN",
            "description": "A virtual private network extends a private network across a public network and enables users to send and receive data across shared or public networks as if their computing devices were directly connected to the private network.",  # noqa
            "synonyms": ["Virtual Private Network"],
        },
        {
            "name": "Data Warehouse",
            "description": "A system used for reporting and data analysis, and is considered a core component of business intelligence.",  # noqa
            "synonyms": [],
        },
        {
            "name": "Data Lake",
            "description": "A system or repository of data stored in its natural/raw format, usually object blobs or files.",  # noqa
            "synonyms": [],
        },
        {
            "name": "ETL",
            "description": "Extract, Transform, Load is a process in database usage and especially in data warehousing.",  # noqa
            "synonyms": ["Extract, Transform, Load"],
        },
        {
            "name": "RPA",
            "description": "Robotic Process Automation is the technology that allows anyone today to configure computer software, or a “robot” to emulate and integrate the actions of a human interacting within digital systems to execute a business process.",  # noqa
            "synonyms": ["Robotic Process Automation"],
        },
        {
            "name": "BI",
            "description": "Business Intelligence comprises the strategies and technologies used by enterprises for the data analysis of business information.",  # noqa
            "synonyms": ["Business Intelligence"],
        },
        {
            "name": "ERP",
            "description": "Enterprise Resource Planning is the integrated management of main business processes, often in real-time and mediated by software and technology.",  # noqa
            "synonyms": ["Enterprise Resource Planning"],
        },
        {
            "name": "CRM",
            "description": "Customer Relationship Management is a technology for managing all your company’s relationships and interactions with customers and potential customers.",  # noqa
            "synonyms": ["Customer Relationship Management"],
        },
        {
            "name": "HRIS",
            "description": "Human Resource Information System is a software or online solution for the data entry, data tracking, and data information needs of the Human Resources, payroll, management, and accounting functions within a business.",  # noqa
            "synonyms": ["Human Resource Information System"],
        },
        {
            "name": "HCM",
            "description": "Human Capital Management is a set of practices related to people resource management.",  # noqa
            "synonyms": ["Human Capital Management"],
        },
        {
            "name": "PLM",
            "description": "Product Lifecycle Management is the process of managing the entire lifecycle of a product from inception, through engineering design and manufacturing, to service and disposal of manufactured products.",  # noqa
            "synonyms": ["Product Lifecycle Management"],
        },
    ]
    for term in terms:
        context.sync_await(
            context.container[TerminologyStore].create_term(
                term_set=agent_name,
                name=term["name"],  # type: ignore
                description=term["description"],  # type: ignore
                synonyms=term["synonyms"],
            )
        )


@given(parsers.parse('an association between "{guideline_name}" and "{tool_name}"'))
def given_a_guideline_tool_association(
    context: _TestContext,
    tool_name: str,
    guideline_name: str,
) -> GuidelineToolAssociation:
    guideline_tool_association_store = context.container[GuidelineToolAssociationStore]

    return context.sync_await(
        guideline_tool_association_store.create_association(
            guideline_id=context.guidelines[guideline_name].id,
            tool_id=context.tools[tool_name].id,
        )
    )


@given(parsers.parse('a guideline "{guideline_name}" to {do_something} when {a_condition_holds}'))
def given_a_guideline_name_to_when(
    guideline_name: str,
    do_something: str,
    a_condition_holds: str,
    sync_await: SyncAwaiter,
    container: Container,
    agent_id: AgentId,
    context: _TestContext,
) -> None:
    guideline_store = container[GuidelineStore]

    context.guidelines[guideline_name] = sync_await(
        guideline_store.create_guideline(
            guideline_set=agent_id,
            predicate=a_condition_holds,
            action=do_something,
        )
    )


@given(parsers.parse('the tool "{tool_name}"'))
def given_a_tool(
    sync_await: SyncAwaiter,
    container: Container,
    tool_name: str,
    context: _TestContext,
) -> None:
    tool_store = container[LocalToolService]

    async def create_tool(
        name: str,
        module_path: str,
        description: str,
        parameters: dict[str, Any],
        required: list[str],
    ) -> Tool:
        return await tool_store.create_tool(
            name=name,
            module_path=module_path,
            description=description,
            parameters=parameters,
            required=required,
        )

    tools: dict[str, dict[str, Any]] = {
        "get_terrys_offering": {
            "name": "get_terrys_offering",
            "description": "Explain Terry's offering",
            "module_path": "tests.tool_utilities",
            "parameters": {},
            "required": [],
        },
    }

    tool = sync_await(create_tool(**tools[tool_name]))

    multiplexed_tool_service = container[MultiplexedToolService]

    context.tools[tool_name] = sync_await(
        multiplexed_tool_service.read_tool(
            tool.id, next(iter(multiplexed_tool_service.services.keys()))
        )
    )


@when("processing is triggered", target_fixture="emitted_events")
def when_processing_is_triggered(
    context: _TestContext,
    engine: AlphaEngine,
    session_id: SessionId,
) -> list[EmittedEvent]:
    buffer = EventBuffer()

    context.sync_await(
        engine.process(
            Context(
                session_id=session_id,
                agent_id=context.agent_id,
            ),
            buffer,
        )
    )

    return buffer.events


@then("a single message event is emitted")
def then_a_single_message_event_is_emitted(
    emitted_events: list[EmittedEvent],
) -> None:
    assert len(list(filter(lambda e: e.kind == "message", emitted_events))) == 1


@then(parsers.parse("the message contains {something}"))
def then_the_message_contains(
    emitted_events: list[EmittedEvent],
    something: str,
) -> None:
    message_event = next(e for e in emitted_events if e.kind == "message")
    message = cast(MessageEventData, message_event.data)["message"]

    assert nlp_test(
        context=message,
        predicate=f"the text contains {something}",
    )

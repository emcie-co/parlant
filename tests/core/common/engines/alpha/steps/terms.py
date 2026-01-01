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

from pytest_bdd import given, parsers

from parlant.core.agents import AgentId, AgentStore
from parlant.core.glossary import GlossaryStore
from parlant.core.tags import Tag

from tests.core.common.engines.alpha.utils import step
from tests.core.common.utils import ContextOfTest


@step(given, parsers.parse('the term "{term_name}" defined as {term_description}'))
def given_the_term_definition(
    context: ContextOfTest,
    term_name: str,
    term_description: str,
    agent_id: AgentId,
) -> None:
    glossary_store = context.container[GlossaryStore]
    agent_id = context.sync_await(context.container[AgentStore].read_agent(agent_id)).id
    term = context.sync_await(
        glossary_store.create_term(
            name=term_name,
            description=term_description,
        )
    )
    context.sync_await(
        glossary_store.upsert_tag(
            term_id=term.id,
            tag_id=Tag.for_agent_id(agent_id),
        )
    )


@step(given, "50 random terms related to technology companies")
def given_50_random_terms_related_to_technology_companies(
    context: ContextOfTest,
    agent_id: AgentId,
) -> None:
    agent_id = context.sync_await(context.container[AgentStore].read_agent(agent_id)).id
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
            "description": "An intermediary server separating customers from the websites they browse.",  # noqa
            "synonyms": [],
        },
        {
            "name": "VPN",
            "description": "A virtual private network extends a private network across a public network and enables customers to send and receive data across shared or public networks as if their computing devices were directly connected to the private network.",  # noqa
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
            context.container[GlossaryStore].create_term(
                tags=[Tag.for_agent_id(agent_id)],
                name=term["name"],  # type: ignore
                description=term["description"],  # type: ignore
                synonyms=term["synonyms"],
            )
        )


<div align="center">
<!--<img alt="Parlant Banner" src="https://github.com/emcie-co/parlant/blob/develop/banner.png?raw=true" />-->


  <img alt="Parlant Banner" src="https://github.com/emcie-co/parlant/blob/develop/banner.png?raw=true" />
  <h2>Hello, Conversation Modeling!</h2>

Parlant is the open-source engine for controlled, compliant, and purposeful generative AI conversations. It gives you the power of LLMs without the unpredictability.

  <a href="https://trendshift.io/repositories/12768" target="_blank"><img src="https://trendshift.io/api/badge/repositories/12768" alt="emcie-co%2Fparlant | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>


  <p>
    <a href="https://www.parlant.io/" target="_blank">Website</a> —
    <a href="https://www.parlant.io/docs/quickstart/introduction" target="_blank">Introduction</a> —
    <a href="https://www.parlant.io/docs/tutorial/getting-started" target="_blank">Tutorial</a> —
    <a href="https://www.parlant.io/docs/about" target="_blank">About</a> —
    <a href="https://www.reddit.com/r/parlant_official/" target="_blank">Reddit</a>
  </p>


  
  <p>
    <a href="https://pypi.org/project/parlant/" alt="Parlant on PyPi"><img alt="PyPI - Version" src="https://img.shields.io/pypi/v/parlant"></a>
    <img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/parlant">
    <a href="https://opensource.org/licenses/Apache-2.0"><img alt="Apache 2 License" src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" /></a>
    <img alt="GitHub commit activity" src="https://img.shields.io/github/commit-activity/w/emcie-co/parlant?label=commits">
    <img alt="PyPI - Downloads" src="https://img.shields.io/pypi/dm/parlant">
    <a href="https://discord.gg/duxWqxKk6J"><img alt="Discord" src="https://img.shields.io/discord/1312378700993663007?style=flat&logo=discord&logoColor=white&label=discord">
</a>
  </p>

</div>

## Introduction Video
[![Parlant Introduction](https://github.com/emcie-co/parlant/blob/develop/yt-preview.png?raw=true)](https://www.youtube.com/watch?v=_39ERIb0100)

#### Install
```bash
pip install parlant
```

#### Option 1: Use the CLI
Start the server and start interact with the default agent
```bash
parlant-server run
# Now visit http://localhost:8800
```

Add behavioral guidelines and let Parlant do the rest
```bash
parlant guideline create \
    --condition "the user greets you" \
    --action "thank them for checking out Parlant"
# Now start a new conversation and greet the agent
```

##### Using with LiteLLM (OpenRouter and Ollama)

To run the Parlant server using LiteLLM, which allows integration with services like OpenRouter for language models and Ollama for embeddings, follow these steps:

1.  **Set Environment Variables:**
    Before starting the server, you need to configure the following environment variables:

    *   `LITELLM_PROVIDER_API_KEY`: Your API key for OpenRouter.
        *   Example: `export LITELLM_PROVIDER_API_KEY="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"`
    *   `LITELLM_PROVIDER_MODEL_NAME`: The OpenRouter model you wish to use. Remember to prefix with `openrouter/`. You can find available models at [OpenRouter Models](https://openrouter.ai/models).
        *   Example: `export LITELLM_PROVIDER_MODEL_NAME="openrouter/mistralai/mistral-7b-instruct"`
    *   `OLLAMA_EMBEDDING_MODEL_NAME`: The name of the embedding model hosted by your Ollama instance (e.g., `nomic-embed-text`). The `ollama/` prefix is handled internally if not provided.
        *   Example: `export OLLAMA_EMBEDDING_MODEL_NAME="nomic-embed-text"`
        *   (Ensure you have this model in Ollama: `ollama pull nomic-embed-text`)
    *   `OLLAMA_API_BASE`: The base URL for your Ollama server. This is typically `http://localhost:11434` if Ollama is running locally.
        *   Example: `export OLLAMA_API_BASE="http://localhost:11434"`

2.  **Start the Server with LiteLLM:**
    Use the `--litellm` flag when running the server:
    ```bash
    parlant-server run --litellm
    ```
    Now, the server will use OpenRouter for LLM operations and Ollama for embeddings, based on your environment variable settings.

#### Option 2: Use the Python SDK
```python
# file: agent.py

import parlant.sdk as p
import asyncio
from textwrap import dedent


@p.tool
async def get_on_sale_car(context: p.ToolContext) -> p.ToolResult:
    return p.ToolResult("Hyundai i20")

@p.tool
async def human_handoff(context: p.ToolContext) -> p.ToolResult:
    await notify_sales(context.customer_id, context.session_id)

    return p.ToolResult(
        data="Session handed off to sales team",
        # Disable auto-responding using the AI agent on this session
        # following the next message.
        control={"mode": "manual"},
    )


async def start_conversation_server() -> None:
    async with p.Server() as server:
        agent = await server.create_agent(
            name="Johnny",
            description="You work at a car dealership",
        )

        journey = await agent.create_journey(
            title="Research Car",
            conditions=[
                "The customer wants to buy a new car",
                "The customer expressed general interest in new cars",
            ],
            description=dedent("""\
                Help the customer come to a decision of what new car to get.

                The process goes like this:
                1. First try to actively understand their needs
                2. Once needs are clarified, recommend relevant categories or specific models for consideration."""),
        )

        offer_on_sale_car = await journey.create_guideline(
          condition="the customer indicates they're on a budget",
          action="offer them a car that is on sale",
          tools=[get_on_sale_cars],
        )

        transfer_to_sales = await journey.create_guideline(
          condition="the customer clearly stated they wish to buy a specific car",
          action="transfer them to the sales team",
          tools=[human_handoff_to_sales],
        )

        await transfer_to_sales.prioritize_over(proactively_offer_on_sale_car)


asyncio.run(start_conversation_server())
```

Run `python agent.py` and visit `http://localhost:8800`.


## Quick Demo
<img alt="Parlant Banner" src="https://github.com/emcie-co/parlant/blob/develop/ParlantGIF.gif?raw=true" />


## What is Conversation Modeling?
You've built an AI agent—that's great! However, when you actually test it, you see it's not handling many customer interactions properly, and your business experts are displeased with it. What do you do?

Enter Conversation Modeling (CM): a new powerful and reliable approach to controlling how your agents interact with your users.

A conversation model is a structured, domain-specific set of principles, actions, objectives, and terms that an agent applies to a given conversation.

### Why Conversation Modeling?

The problem of getting your AI agent to say what _you_ want it to say is a hard one, experienced by virtually anyone building customer-facing agents. Here's how Conversation Modeling compares to other approaches to solving this problem.

- **Flow engines** _force_ the user to interact according to predefined flows. In contrast, a **CM engine** dynamically _adapts_ to a user's natural interaction patterns while conforming to your rules.

- **Free-form prompt engineering** leads to _inconsistency_, frequently failing to uphold requirements. Conversely, a **CM engine** leverages structure to _enforce_ conformance to a Conversation Model.


## Who uses Parlant?
Parlant is used to deliver complex conversational agents that reliably follow your business protocols in use cases such as:
- 🏦 Regulated financial services
- 🏥 Healthcare communications
- 📜 Legal assistance
- 🛡️ Compliance-focused use cases
- 🎯 Brand-sensitive customer service
- 🤝 Personal advocacy and representation

## How is Parlant used?
Developers and data-scientists are using Parlant to:

- 🤖 Create custom-tailored conversational agents quickly and easily
- 👣 Define behavioral guidelines for agents to follow (Parlant ensures they are followed reliably)
- 🛠️ Attach tools with specific guidance on how to properly use them in different contexts
- 📖 Manage their agents’ glossary to ensure strict interpretation of terms in a conversational context
- 👤 Add customer-specific information to deliver personalized interactions

#### How does Parlant work?
```mermaid
graph TD
    API(Parlant REST API) -->|React to Session Trigger| Engine[AI Response Engine]
    Engine -->|Load Domain Terminology| GlossaryStore
    Engine -->|Match Guidelines| GuidelineMatcher
    Engine -->|Infer & Call Tools| ToolCaller
    Engine -->|Tailor Guided Message| MessageComposer
```

When an agent needs to respond to a customer, Parlant's engine evaluates the situation, checks relevant guidelines, gathers necessary information through your tools, and continuously re-evaluates its approach based on your guidelines as new information emerges. When it's time to generate a message, Parlant implements self-critique mechanisms to ensure that the agent's responses precisely align with your intended behavior as given by the contextually-matched guidelines.

***📚 More technical docs on the architecture and API are available under [docs/](./docs)***.

## 📦 Quickstart
Parlant comes pre-built with responsive session (conversation) management, a detection mechanism for incoherence and contradictions in guidelines, content-filtering, jailbreak protection, an integrated sandbox UI for behavioral testing, native API clients in Python and TypeScript, and other goodies.

```bash
$ pip install parlant
$ parlant-server run
$ # Open the sandbox UI at http://localhost:8800 and play
```

## 🙋‍♂️🙋‍♀️ Who Is Parlant For?
Parlant is the right tool for the job if you're building an LLM-based chat agent, and:

1. 🎯 Your use case places a **high importance on behavioral precision and consistency**, particularly in customer-facing scenarios
1. 🔄 Your agent is expected to undergo **continuous behavioral refinements and changes**, and you need a way to implement those changes efficiently and confidently
1. 📈 You're expected to maintain a **growing set of behavioral guidelines**, and you need to maintain them coherently and with version-tracking
1. 💬 Conversational UX and user-engagmeent is an important concern for your use case, and you want to easily **control the flow and tone of conversations**

## ⭐ Star Us: Your Support Goes a Long Way!
[![Star History Chart](https://api.star-history.com/svg?repos=emcie-co/parlant&type=Date)](https://star-history.com/#emcie-co/parlant&Date)

## 🤔 What Makes Parlant Different?

In a word: **_Guidance._** 🧭🚦🤝

Parlant's engine revolves around solving one key problem: How can we _reliably guide_ customer-facing agents to behave in alignment with our needs and intentions?

Hence Parlant's fundamentally different approach to agent building: [Managed Guidelines](https://www.parlant.io/docs/concepts/customization/guidelines):

```bash
parlant guideline create \
  --condition "the customer wants to return an item" \
  --action "get the order number and item name and then help them return it"
```

By giving structure to behavioral guidelines, and _granularizing_ guidelines (i.e. making each behavioral guideline a first-class entity in the engine), Parlant's engine is able to offer unprecedented control, quality, and efficiency in building LLM-based agents:

1. 🛡️ **Reliability:** Running focused self-critique in real-time, per guideline, to ensure it is actually followed
1. 💡 **Explainability:** Providing feedback around its interpretation of guidelines in each real-life context, which helps in troubleshooting and improvement
1. 🔧 **Maintainability:** Helping you maintain a coherent set of guidelines by detecting and alerting you to possible contradictions (gross or subtle) in your instructions

## 🤖 Works with all major LLM providers
- [OpenAI](https://platform.openai.com/docs/overview) (also via [Azure](https://learn.microsoft.com/en-us/azure/ai-services/openai/))
- [Gemini](https://ai.google.dev/)
- [Meta Llama 3](https://www.llama.com/) (via [Together AI](https://www.together.ai/) or [Cerebras](https://cerebras.ai/))
- [Anthropic](https://www.anthropic.com/api) (also via [AWS Bedrock](https://aws.amazon.com/bedrock/))
- And more are added regularly

## 📚 Learning Parlant

To start learning and building with Parlant, visit our [documentation portal](https://parlant.io/docs/quickstart/introduction).

Need help? Ask us anything on [Discord](https://discord.gg/duxWqxKk6J). We're happy to answer questions and help you get up and running!

## 💻 Usage Example
Adding a guideline for an agent—for example, to ask a counter-question to get more info when a customer asks a question:
```bash
parlant guideline create \
    --condition "a free-tier customer is asking how to use our product" \
    --action "first seek to understand what they're trying to achieve"
```

## 👋 Contributing
We use the Linux-standard Developer Certificate of Origin ([DCO.md](DCO.md)), so that, by contributing, you confirm that you have the rights to submit your contribution under the Apache 2.0 license (i.e., that the code you're contributing is truly yours to share with the project).

Please consult [CONTRIBUTING.md](CONTRIBUTING.md) for more details.

Can't wait to get involved? Join us on [Discord](https://discord.gg/duxWqxKk6J) and let's discuss how you can help shape Parlant. We're excited to work with contributors directly while we set up our formal processes!

Otherwise, feel free to start a discussion or open an issue here on GitHub—freestyle 😎.

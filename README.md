
<div align="center">
  
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/emcie-co/parlant/blob/develop/LogoTransparentLight.png?raw=true">
  <img alt="Parlant Banner" src="https://github.com/emcie-co/parlant/blob/develop/LogoTransparentDark.png?raw=true" width=400 />
</picture>



  <p>
    <a href="https://www.parlant.io/" target="_blank">Website</a> —
    <a href="https://www.parlant.io/docs/quickstart/introduction" target="_blank">Introduction</a> —
    <a href="https://www.parlant.io/docs/tutorial/getting-started" target="_blank">Tutorial</a> —
    <a href="https://www.parlant.io/docs/about" target="_blank">About</a> —
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

<a href="https://trendshift.io/repositories/12768" target="_blank"><img src="https://trendshift.io/api/badge/repositories/12768" alt="emcie-co%2Fparlant | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>

</div>

# Build a scalable AI chat agent—in minutes.
Parlant is the open-source backbone for LLM agents that stay under control as you scale their complexity.

## 💡 Bring control and consistency to LLM agents...

You've built a conversational AI agent—that's great! However, when you actually test it, you see it's not handling many customer interactions properly, and your business experts are displeased with its behavior. We've all been there. What do you do?

Parlant might be the answer you've been waiting for. It's an open-source conversation modeling engine that gives you unparalleled, scalable control over LLMs, enabling the creation of deliberate, predictable, and compliant Agentic User Experience (UX).

## ✨ Why Parlant?

Building good quality conversational AI means teaching your agents many facts, rules, and principles of behavior to follow when interacting with customers.

Parlant lets you define your conversational rules and logic in natural language, and it manages the underlying LLM's context such that it knows exactly which rules it must conform to at any point in a conversation.

- 🚀 Gets you up and running with a live conversational agent in minutes
- ⚡️ Uses perceived-performance techniques to deliver responsive chat UX out-of-the-box
- 🧱 A scalable foundation for scaling your agent's knowledge and rules
- 🎯 Dramatically increases the LLM's consistency in following your instructions
- 💬 Tailor-made framework for conversational, customer-facing use cases

## 🚀 Getting started

Getting Parlant up and running is straightforward.

### Installation

```bash
pip install parlant
```

### Code Example

```python
import asyncio
from collections import defaultdict
from textwrap import dedent

CARTS: dict[str, list[str]] = defaultdict(list)

@p.tool
async def add_item_to_cart(context: p.ToolContext, item_id: str) -> p.ToolResult:
    # A tool function to add an item to a customer's cart.
    cart = CARTS[context.customer_id]
    cart.append(item_id)
    # The tool's result will be stored in the current conversation session,
    # informing the AI agent's understanding of what took place both in the
    # conversation itself, as well as external events such as this.
    return p.ToolResult(f"Item added successfully. {len(cart)} items in cart.")

@p.tool
async def list_books(context: p.ToolContext, preference_query: str) -> p.ToolResult:
  # Parlant automatically infers and fills up argument values from conversation context.
  # In more advanced use cases, you can instruct Parlant what rules to follow when
  # inferring argument values, such as whether they can be deduced or if they must
  # be provided explicitly by the customer (to avoid mistakes).
  books = await find_books(preference_query)
  return p.ToolResult(books)

@p.tool
async def human_handoff(context: p.ToolContext, reason: str) -> p.ToolResult:
  # The context object contains important information, identifying the
  # customer and session ID of the calling conversation.
  await notify_human_operators(
    context.customer_id,
    context.session_id,
    reason,
  )

  return p.ToolResult(
    data="Session handed off to sales team",
    # Disable auto-responding using the AI agent
    # on this session, following the next message,
    # by adding a control directive to transfer
    # the session to manual mode.
    control={"mode": "manual"},
  )

async def configure_agent(server: p.Server) -> None:
  # Configure the AI agent with a name and description.
  # There are more configuration options available, but this is a good start.
  # This agent will be the container of our conversation logic.
  agent = await server.create_agent(
    name="Carlton",
    description="Online bookstore service representative",
  )

  # A journey is  a sequence of steps that the AI agent will follow
  # in the scope of a particular conversational use case.
  #
  # You can define as many journeys as you like, and they can be
  # used to handle different conversational flows.
  #
  # Parlant will automatically select the most appropriate journey(s)
  # to use based on the conversation context, keeping your prompts
  # shorter, and more focused and consistent in their results.
  recommend_book = await agent.create_journey(
    title="Recommend Book",
    description=dedent(
      """\
      A journey to recommend a book to a customer.

      1. Ask the customer for their preferences.
      2. Recommend a book based on the preferences, until the customer sounds interested.
      3. Once the customer sounds interested, ask if they want to add it to their cart.
      4. If they do, add the book to their cart.
      5. If they don't, ask if they want to see more recommendations and repeat the process.
      """,
    ),
    # The agent will follow the journey only when certain conditions are met.
    # You decide what those conditions are, allowing for full control and customization.
    conditions=[
      "The customer is looking for a book recommendation",
      "The customer isn't sure what book they want",
    ],
  )

  # Attach tools to help the AI agent perform actions
  # and retrieve data while following the journey.
  # Tools only get evaluated if the associated condition
  # currently holds in a conversation—significantly
  # improving the accuracy of tool calls.
  await recommend_book.attach_tool(
    tool=list_books,
    condition="You need to know what books are available",
  )

  await recommend_book.attach_tool(
    tool=add_item_to_cart,
    condition="The customer wants to add the book to their cart",
  )

  # Instruct the agent on how to handle conversational
  # edge-cases in the journey, according to your needs.
  await recommend_book.create_guideline(
    condition="The customer wants a book that is not available",
    action="Ask the customer if they'd like you to order it for them",
  )

  # Install agent-wide guidelines that apply to all journeys.
  # Palant lets you add up to hundreds of guidelines. It takes
  # care to only select and feed the most applicable ones to the
  # underlying LLM calls at each point in the conversation,
  # so you can scale your agent's complexity with ease.
  await agent.create_guideline(
    condition="The customer is frustrated or angry",
    action="Hand off the conversation to a human operator and inform the customer",
    tools=[human_handoff],
  )

  # And much more!
  #
  # Parlant supports:
  # 1. Guideline relationships — allowing you to prioritize certain guidelines over others,
  # make some guidelines activate only when others are in-context (dependency), and more.
  # 2. Glossary — allowing you to define terms and phrases that the AI agent should understand
  # about your context, either when interacting with customers or when making sense of guidelines.
  # 3. Context variables — allowing you to define customer-specific variables that facilitate
  # experience personalization and enable long-term memory across conversations.
  # 4. Utterance templates — allowing you to define pre-written responses that the AI agent
  # can use to respond to customers, ensuring zero output hallucinations and tailored language.
  #
  # And more!

async def start_conversation_server() -> None:
  async with p.Server() as server:
    await configure_agent(server)

if __name__ == "__main__":
  asyncio.run(start_conversation_server())
  # After running, visit http://localhost:8800
  # for an integrated playground web UI.
```

## ⭐ Star Parlant to support our vision!
Creating Parlant is a significant effort, as conversational semantics are extremely complex to get under control.

Your star goes a long way to support our team's effort in creating a reality where generative AI agents reliably do what we tell them!

[![Star History Chart](https://api.star-history.com/svg?repos=emcie-co/parlant&type=Date)](https://www.star-history.com/#emcie-co/parlant&Date)

## Integrated prototyping playground
<img alt="Parlant Demo" src="https://github.com/emcie-co/parlant/blob/develop/demo.gif?raw=true" />

## 🛠️ Key features

Parlant is ***packed*** with useful features for production conversational AI!

  * **Behavioral Guidelines:** Easily define rules and guardrails for agent interactions and **dictate and enforce exact conversation behavior**.
  * **Semantic Relationships:** Define how different guidelines relate to each other (dependencies, prioritization, etc.), creating sophisticated and adaptive conversational flows.
  * **Tool Integration:** Seamlessly attach external tools (APIs, databases, etc.) with specific guidance for agent usage.
  * **Context Awareness:** Intelligently tracks conversation progress, understanding what instructions need to apply at each point, and when required actions have already been taken.
  * **Dynamic Guideline Matching:** Ensures contextually relevant instruction execution, eliminating irrelevant instructions at any point in the conversation — solving LLM attention drift.
  * **Utterance Templates:** Sanitize LLM outputs, preventing unpredictable or inaccurate messages and ensuring compliance and accuracy.
  * **Glossary Management:** Control and manage the agent's vocabulary for consistent and accurate communication.
  * **Contextual Information:** Inject customer-specific or domain-specific information for personalized and relevant responses.
  * **Continuous Re-evaluation:** The Parlant engine constantly assesses the conversational situation, checks relevant guidelines, gathers necessary information, and re-evaluates its approach.

**For more detailed installation instructions and advanced usage, please refer to our [Official Documentation](https://parlant.io).**

## React Widget
Please see https://github.com/emcie-co/parlant-chat-react for our official, highly-customizable React widget to interact with your Parlant server on your app.

![https://github.com/emcie-co/parlant-chat-react/raw/main/src/assets/chatbox.gif](https://github.com/emcie-co/parlant-chat-react/raw/main/src/assets/chatbox.gif)

```typescript
import React from 'react';
import ParlantChatbox from 'parlant-chat-react';

function App() {
  return (
    <div>
      <h1>My Application</h1>
      <ParlantChatbox
        float
        agentId="AGENT_ID"
        server="PARLANT_SERVER_URL"
      />
    </div>
  );
}

export default App;
```


## 🌐 Use cases & industries

Parlant is ideal for organizations that demand precision and reliability from their AI agents. It's currently being used to deliver complex conversational agents in:

  * **Regulated Financial Services:** Ensuring compliance and accuracy in customer interactions.
  * **Healthcare Communications:** Providing accurate, compliant, and sensitive patient information.
  * **Legal Assistance:** Delivering reliable and verifiable legal guidance.
  * **Compliance-Focused Use Cases:** Automating adherence to industry standards and strict protocols.
  * **Brand-Sensitive Customer Service:** Maintaining consistent brand voice and policies across all interactions.
  * **Personal Advocacy & Representation:** Supporting structured and goal-oriented dialogues for high-stakes scenarios.

## 🤝 Contributing
We use the Linux-standard Developer Certificate of Origin (DCO.md), so that, by contributing, you confirm that you have the rights to submit your contribution under the Apache 2.0 license (i.e., that the code you're contributing is truly yours to share with the project).

Please consult [CONTRIBUTING.md](CONTRIBUTING.md) for more details.

Can't wait to get involved? Join us on Discord and let's discuss how you can help shape Parlant. We're excited to work with contributors directly while we set up our formal processes!

## 📧 Contact & support

Need help? Ask us anything on [Discord](https://discord.gg/duxWqxKk6J). We're happy to answer questions and help you get up and running!

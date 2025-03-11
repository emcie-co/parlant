
<div align="center">
<img alt="Parlant Logo" src="https://github.com/emcie-co/parlant/blob/e0fefba25efd1b956ead881964025baae5dc4119/logo.png" />
  <h3>Parlant: Build Reliable AI Customer Service Agents with LLMs 💬 ✅</h3>
  <p>
    <a href="https://www.parlant.io/" target="_blank">Website</a> —
    <a href="https://www.parlant.io/docs/quickstart/introduction" target="_blank">Introduction</a> —
    <a href="https://www.parlant.io/docs/tutorial/getting-started/starting-the-tutorial" target="_blank">Tutorial</a> —
    <a href="https://www.parlant.io/docs/about" target="_blank">About</a>
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

Parlant is an LLM-first conversational AI framework designed to provide developers with the control and precision they need over their AI customer service agents, utilizing behavioral guidelines and runtime supervision. 🔧 🎛️ It's operated using an easy-to-use CLI 📟 and native client SDKs in Python and TypeScript 📦.

**IMPORTANT NOTE:** We're looking for more contributors to help get customer-facing agents under control! To be a part of this effort, join our [Discord server](https://discord.gg/duxWqxKk6J) and tell us about your relevant skills and how you wish to help.

## Who uses Parlant?
Parlant is used to deliver complex AI agents that reliably follow your business rules in use cases such as:
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

## ✨ Take Control of AI Agent Decisions
**Parlant transforms how AI agents make decisions in customer-facing use cases.**

It moves beyond traditional prompt engineering and flow charts. Instead, you provide it with contextual atomic guidelines, and it adapts your instructions to each conversation in real-time. By intelligently matching and activating relevant behavioral guidelines for each specific context, your agents end up consistently following your business rules while maintaining much more natural conversations with your customers than you could achieve with predefined semantic graphs or flow charts.

### How does Parlant work?
```mermaid
graph TD
    API(Parlant REST API) -->|React to Session Trigger| Engine[AI Response Engine]
    Engine -->|Load Domain Terminology| GlossaryStore
    Engine -->|Match Guidelines| GuidelineProposer
    Engine -->|Infer & Call Tools| ToolCaller
    Engine -->|Tailor Guided Message| MessageGenerator
```

When an agent needs to respond to a customer, Parlant's engine evaluates the situation, checks relevant guidelines, gathers necessary information through your tools, and continuously re-evaluates its approach based on your guidelines as new information emerges. When it's time to generate a message, Parlant implements self-critique mechanisms to ensure that the agent's responses precisely align with your intended behavior as given by the contextually-matched guidelines.

***📚 More technical docs on the architecture and API are available under [docs/](./docs)***.

## 📦 Quickstart
Parlant comes pre-built with responsive session (conversation) management, a detection mechanism for incoherence and contradictions in guidelines, content-filtering, jailbreak protection, an integrated sandbox UI for behavioral testing, native API clients in Python and TypeScript, and other goodies.

```bash
$ pip install parlant
$ parlant-server
$ # Open the sandbox UI at http://localhost:8800 and play
```

<img alt="Parlant Preview" src="https://github.com/emcie-co/parlant/blob/02c0e11116e03f3622077436ce9d61811bceb519/preview.gif" />

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

Parlant's engine revolves around solving one key problem: How can we _reliably guide_ customer-facing agents to behave in alignment with our needs and intentions.

Hence Parlant's fundamentally different approach to agent building: [Managed Guidelines](https://www.parlant.io/docs/concepts/customization/guidelines):

```bash
$ parlant guideline create \
    --agent-id MY_AGENT_ID \
    --condition "the customer wants to return an item" \
    --action "get the order number and item name and then help them return it"
```

By giving structure to behavioral guidelines, and _granularizing_ guidelines (i.e. making each behavioral guideline a first-class entity in the engine), Parlant's engine is able to offer unprecedented control, quality, and efficiency in building LLM-based agents:

1. 🛡️ **Reliability:** Running focused self-critique in real-time, per guideline, to ensure it is actually followed
1. 💡 **Explainability:** Providing feedback around its interpretation of guidelines in each real-life context, which helps in troubleshooting and improvement
1. 🔧 **Maintainability:** Helping you maintain a coherent set of guidelines by detecting and alerting you to possible contradictions (gross or subtle) in your instructions

## 💪 Other benefits

### Control that actually works
* **Coherence checks**: Catch conflicts by having Parlant evaluate new guidelines against existing ones before they're applied
* **See changes instantly**: Modify behavior on the fly by updating guidelines directly — no retraining or redeployment needed
* **Track changes in Git**: Manage agent behavior like code by storing configuration as JSON in your repo. Review, rollback, branch, and merge just like any other code
* **Clean architecture**: Separate concerns by keeping business logic in tools and conversation patterns in guidelines. Each piece does what it does best

### Deploy with confidence
* **Reliable at scale**: Parlant filters and selects guidelines per context, allowing you to scale your agent's complexity and use-cases while maintaining consistent, focused behavior
* **Debug with ease**: Troubleshoot effectively by tracing which guidelines were applied and why for any given response. leveraging Parlant's highly descriptive and explainable log outputs
* **Test before deploy**: Validate changes using the built-in chat UI to test new behaviors before they reach customers

## 🤖 Works with all major LLM providers
- [OpenAI](https://platform.openai.com/docs/overview) (also via [Azure](https://learn.microsoft.com/en-us/azure/ai-services/openai/))
- [Gemini](https://ai.google.dev/)
- [Meta Llama 3](https://www.llama.com/) (via [Together AI](https://www.together.ai/) or [Cerebras](https://cerebras.ai/))
- [Anthropic](https://www.anthropic.com/api) (also via [AWS Bedrock](https://aws.amazon.com/bedrock/))

## 📚 Learning Parlant

To start learning and building with Parlant, visit our [documentation portal](https://parlant.io/docs/quickstart/introduction).

Need help? Ask us anything on [Discord](https://discord.gg/duxWqxKk6J). We're happy to answer questions and help you get up and running!

## 💻 Usage Example
Adding a guideline for an agent—for example, to ask a counter-question to get more info when a customer asks a question:
```bash
parlant guideline create \
    --agent-id CUSTOMER_SUCCESS_AGENT_ID \
    --condition "a free-tier customer is asking how to use our product" \
    --action "first seek to understand what they're trying to achieve"
```

In Parlant, Customer-Agent interaction happens asynchronously, to enable more natural customer interactions, rather than forcing a strict and unnatural request-reply mode.

Here's a basic example of a simple client (using the TypeScript client SDK):

```typescript
import { ParlantClient } from 'parlant-client';

const client = ParlantClient({ environment: SERVER_ADDRESS });

session_id = "...";

// Post customer message
const customerEvent = await client.sessions.createEvent(session_id, {
   kind: "message",
   source: "customer",
   message: "How do you use this product?",
});

// Wait for and get the agent's reply
const [agentEvent] = (await client.sessions.listEvents(session_id, {
   kinds: "message",
   source: "ai_agent",
   minOffset: customerEvent.offset,
   waitForData: 60 // Wait up to 60 seconds for an answer
}));

// Print the agent's reply
const { agentMessage } = agentEvent.data as { message: string };
console.log(agentMessage);

// Inspect the details of the message generation process
const { trace } = await client.sessions.inspectEvent(
   session_id,
   agentEvent.id
);
```

## 👋 Contributing
We use the Linux-standard Developer Certificate of Origin ([DCO.md](DCO.md)), so that, by contributing, you confirm that you have the rights to submit your contribution under the Apache 2.0 license (i.e., that the code you're contributing is truly yours to share with the project).

Please consult [CONTRIBUTING.md](CONTRIBUTING.md) for more details.

Can't wait to get involved? Join us on [Discord](https://discord.gg/duxWqxKk6J) and let's discuss how you can help shape Parlant. We're excited to work with contributors directly while we set up our formal processes!

Otherwise, feel free to start a discussion or open an issue here on GitHub—freestyle 😎.

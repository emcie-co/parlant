
<div align="center">
<img alt="Parlant Logo" src="https://github.com/emcie-co/parlant/blob/70757094103f6cc50e311edaeb0729e960fbcb56/logo.png" width="125" />
  <h3>Parlant</h3>
  <p>A structured approach to building and guiding customer-facing AI agents</p>
  <p>
    <a href="https://www.parlant.io/" target="_blank">Website</a> |
    <a href="https://www.parlant.io/docs/quickstart/introduction" target="_blank">Introduction</a> |
    <a href="https://www.parlant.io/docs/quickstart/installation" target="_blank">Installation</a> |
    <a href="https://www.parlant.io/docs/tutorial/getting_started/overview" target="_blank">Tutorial</a> |
    <a href="https://www.parlant.io/docs/about" target="_blank">About</a>
  </p>
  <p>
    <a href="https://pypi.org/project/parlant/" alt="Parlant on PyPi"><img alt="PyPI - Version" src="https://img.shields.io/pypi/v/parlant"></a>
    <img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/parlant">
    <a href="https://opensource.org/licenses/Apache-2.0"><img alt="Apache 2 License" src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" /></a>
    <img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/emcie-co/parlant">
    <a href="https://discord.gg/duxWqxKk6J"><img alt="Discord" src="https://img.shields.io/discord/1312378700993663007?style=flat&logo=discord&logoColor=white&label=discord">
</a>
  </p>
  <img alt="Parlant Preview" src="https://github.com/emcie-co/parlant/blob/02c0e11116e03f3622077436ce9d61811bceb519/preview.gif" />
</div>

## What is Parlant?
Parlant is a structured way to create and manage guided customer-facing AI agents. It lets you build robust agents from scratch; or, in more advanced cases, a robust interaction layer for your prebuilt agents.

Structure is what lets Parlant get better accuracy than most other agentic frameworks. It allows Parlant to run pinpointed, real-time conformance checks that ensure your agents adhere to your instruction. It also gives you specific feedback on where agents may have misinterpreted your instructions, so you can quickly troubleshoot and improve.

Just as importantly, Parlant's way of breaking behavioral guidance down to different elements helps it understand your intentions better. Using this understanding, Parlant actively gives you feedback on your configuration to help you maintain a behavioral configuration for your agents that's coherent and free of confusing contradictions.

## Why use Parlant?
Building conversational AI agents is relatively simple for most developers—at least, it's relatively simple to build an initial prototype.

But these prototypes are rarely ready to meet customers. Once the prototype is functional, it has to be continually tuned so that its behavior actually provides customers with a good experience that business stakeholders approve.

With DIY prompt-engineering, reliably incorporating feedback from stakeholders is challenging, as simple implementations tend to be fragile and inconsistent.

Parlant bridges this gap by making it easy and fast for developers to adjust the behavior of AI agents reliably, allowing you to iterate quickly with feedback from customers and business stakeholders.

## Real-world impact

[Revenued](https://www.revenued.com), A business capital provider, could get into trouble if their AI agents make false claims or make statements that imply discrimination in lending.

With Parlant, they've been able to quickly integrate feedback from customer service experts and then test and verify that the agents aren't making problematic promises or statements to customers.

## Key benefits

### Control that actually works
* **Guidelines**: Control responses by writing contextual rules - like "offer limited time coupons if it's a holiday" or "make it very clear we don't offer loans if a customer asks about it". By using condition/action definitions, you define exactly when and how your agent should respond
* **Glossary**: Teach your agent your business-specific terminology so that both you and your customers can speak to it naturally in your language
* **Coherence checks**: Catch conflicts by having Parlant evaluate new guidelines against existing ones before they're applied
* **Dynamic context**: Make your agent context-aware by setting user-specific variables like customer account or subscription tier. These shape how your agent responds to each user
* **Guided tool use**: Control API access by linking tools to specific guidelines. This way, your agent only calls APIs when appropriate and with clear intent

### Developer friendly
* **See changes instantly**: Modify behavior on the fly by updating guidelines directly, no retraining or redeployment needed
* **Track changes in Git**: Manage agent behavior like code by storing configuration as JSON in your repo. Review, rollback, branch, and merge just like any other code
* **Clean architecture**: Separate concerns by keeping business logic in tools and conversation patterns in guidelines. Each piece does what it does best
* **Type safety**: Build rapidly using native TypeScript/JavaScript SDKs with proper type definitions

### Deploy with confidence
* **Reliable at scale**: Parlant filters and selects guidelines per context, allowing you to scale your agent's complexity and use-cases while maintaining consistent, focused behavior
* **Debug with ease**: Troubleshoot effectively by tracing which guidelines were applied and why for any given response
* **Test before deploy**: Validate changes using the built-in chat UI to test new behaviors before they reach customers

## Works with all major LLM providers
- [OpenAI](https://platform.openai.com/docs/overview) (also via [Azure](https://learn.microsoft.com/en-us/azure/ai-services/openai/))
- [Gemini](https://ai.google.dev/)
- [Meta Llama 3](https://www.llama.com/) (via [Together AI](https://www.together.ai/) or [Cerebras](https://cerebras.ai/))
- [Anthropic](https://www.anthropic.com/api) (also via [AWS Bedrock](https://aws.amazon.com/bedrock/))

## Getting started
```bash
$ pip install parlant
$ parlant-server
$ # Open http://localhost:8000 and play
```

Install client SDKs:
```bash
$ # For Python clients:
$ pip install parlant-client
$ # For TypeScript clients:
$ npm install parlant-client
```

To start building with Parlant, visit our [documentation portal](https://parlant.io/docs/quickstart/introduction).

Need help? Send us a message on [Discord](https://discord.gg/duxWqxKk6J). We're happy to answer questions and help you get up and running!

## Usage Example
Adding a guideline for an agent—for example, to ask a counter-question to get more info when a customer asks a question:
```bash
parlant guideline create \
    --agent-id CUSTOMER_SUCCESS_AGENT_ID
    --condition "a free-tier customer is asking how to use our product" \
    --action "first seek to understsand what they're trying to achieve"
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
   message: "Hey, I'd like to book a room please",
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

## Contributing
We're currently finalizing our Contributor License Agreement (CLA). Full contribution guidelines will be available soon! 

Want to get involved? Join us on [Discord](https://discord.gg/duxWqxKk6J) and let's discuss how you can help shape Parlant. We're excited to work with contributors directly while we set up our formal processes!

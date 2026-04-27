
## FAQ

### General

**What is Parlant?**
Parlant is an open-source interaction control harness for customer-facing AI agents. It provides optimized context engineering for conversational use cases — getting the right context into the prompt at the right time. Unlike traditional approaches that rely on monolithic system prompts or fragile routing graphs, Parlant uses rules, knowledge, and tools with real-time contextual narrowing.

**How is Parlant different from LangGraph or DSPy?**
Parlant focuses on conversational governance and behavioral control, while frameworks like LangGraph handle workflow automation and DSPy handles prompt optimization. Parlant complements these tools — you can use Parlant alongside LangGraph, Agno, or LlamaIndex for a complete agent stack. See the [Parlant vs LangGraph](https://www.parlant.io/blog/parlant-vs-langgraph) and [Parlant vs DSPy](https://www.parlant.io/blog/parlant-vs-dspy) blog posts for detailed comparisons.

**Is Parlant production-ready?**
Yes. Parlant is designed for enterprise-grade B2C and sensitive B2B interactions. It includes built-in guardrails, explainability via OpenTelemetry, and compliance features for regulated industries.

### Installation & Setup

**What Python version is required?**
Python 3.10 or higher.

**How do I install Parlant?**
```bash
pip install parlant
```

**Do I need a database?**
Parlant manages its own internal state. You don't need to set up a separate database to get started.

### LLM Providers

**Which LLM providers does Parlant support?**
Parlant is LLM-agnostic. The recommended providers are Emcie (built specifically for Parlant), OpenAI, and Anthropic. You can also use any model via LiteLLM, though quality varies — off-the-shelf models that are too small tend to produce inconsistent results.

**Can I swap LLM providers without changing my configuration?**
Yes. Behavioral configuration (rules, guidelines, tools) is decoupled from the LLM. You can swap models without rewriting your agent's behavior rules.

### Core Concepts

**What are Guidelines?**
Guidelines are the core behavioral building blocks in Parlant. They define what the agent should or shouldn't do in specific situations. Unlike system prompts, guidelines are contextually resolved — only the most relevant ones are injected into each conversation turn.

**What are Tools?**
Tools are Python functions that the agent can call during a conversation. They can query databases, call APIs, trigger workflows, or perform any action. Tools can also inject data and dynamic guidelines back into the conversation.

**What is a Glossary?**
A glossary maps colloquial terms and synonyms to precise business definitions. This helps the agent understand customer language — for example, mapping "sea view" and "rooms with a view to the Atlantic" to the official "Ocean View" room category.

**What are Observations?**
Observations define when tools should be triggered. They consist of a condition (when to trigger) and a list of tools to execute. Parlant's engine determines which observations are relevant for each conversation turn.

### Framework Integration

**Can I use Parlant with LangGraph?**
Yes. Parlant handles the behavioral control layer while LangGraph handles workflow automation. You can wrap compiled LangGraph StateGraphs as Parlant tools. See the [Framework Integration](https://github.com/emcie-co/parlant#framework-integration) section in the README for a code example.

**Does Parlant replace my existing agent framework?**
No. Parlant is designed to complement your existing stack. It takes over the conversational governance layer while your framework of choice handles workflow automation, knowledge retrieval, and other processing logic.

### Troubleshooting

**My agent is not following guidelines. What should I do?**
Check that your guidelines are specific and actionable. Vague guidelines like "be helpful" are less effective than concrete ones like "always offer a refund when the customer mentions the product arrived damaged." Also verify that your LLM is capable enough — smaller models may struggle with complex guideline resolution.

**How do I debug agent behavior?**
Parlant ships with elaborate logs, metrics, and traces via OpenTelemetry. Use the [Explainability](https://parlant.io/docs/advanced/explainability) features to trace every decision the agent makes.

**Where can I get help?**
- [Discord](https://discord.gg/duxWqxKk6J) — ask questions and share what you're building
- [GitHub Issues](https://github.com/emcie-co/parlant/issues) — bug reports and feature requests
- [Contact](https://parlant.io/contact) — reach the engineering team directly

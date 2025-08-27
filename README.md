<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/emcie-co/parlant/blob/develop/docs/LogoTransparentLight.png?raw=true">
  <img alt="Parlant - AI Agent Framework" src="https://github.com/emcie-co/parlant/blob/develop/docs/LogoTransparentDark.png?raw=true" width=400 />
</picture>

<h3>Finally, LLM agents that actually follow instructions</h3>

<p>
  <a href="https://www.parlant.io/" target="_blank">🌐 Website</a> •
  <a href="https://www.parlant.io/docs/quickstart/installation" target="_blank">⚡ Quick Start</a> •
  <a href="https://discord.gg/duxWqxKk6J" target="_blank">💬 Discord</a> •
  <a href="https://www.parlant.io/docs/quickstart/examples" target="_blank">📖 Examples</a>
</p>

<p>
  <!-- Keep these links. Translations will automatically update with the README. -->
  <a href="https://zdoc.app/de/emcie-co/parlant">Deutsch</a> |
  <a href="https://zdoc.app/es/emcie-co/parlant">Español</a> |
  <a href="https://zdoc.app/fr/emcie-co/parlant">français</a> |
  <a href="https://zdoc.app/ja/emcie-co/parlant">日本語</a> |
  <a href="https://zdoc.app/ko/emcie-co/parlant">한국어</a> |
  <a href="https://zdoc.app/pt/emcie-co/parlant">Português</a> |
  <a href="https://zdoc.app/ru/emcie-co/parlant">Русский</a> |
  <a href="https://zdoc.app/zh/emcie-co/parlant">中文</a>
</p>

<p>
  <a href="https://pypi.org/project/parlant/"><img alt="PyPI" src="https://img.shields.io/pypi/v/parlant?color=blue"></a>
  <img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10+-blue">
  <a href="https://opensource.org/licenses/Apache-2.0"><img alt="License" src="https://img.shields.io/badge/license-Apache%202.0-green"></a>
  <a href="https://discord.gg/duxWqxKk6J"><img alt="Discord" src="https://img.shields.io/discord/1312378700993663007?color=7289da&logo=discord&logoColor=white"></a>
  <img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/emcie-co/parlant?style=social">
</p>

<a href="https://trendshift.io/repositories/12768" target="_blank">
  <img src="https://trendshift.io/api/badge/repositories/12768" alt="Trending on TrendShift" style="width: 250px; height: 55px;" width="250" height="55"/>
</a>

</div>

## 🎯 The Problem Every AI Developer Faces

You build an AI agent. It works great in testing. Then real users start talking to it and...

- ❌ It ignores your carefully crafted system prompts
- ❌ It hallucinates responses in critical moments
- ❌ It can't handle edge cases consistently
- ❌ Each conversation feels like a roll of the dice

**Sound familiar?** You're not alone. This is the #1 pain point for developers building production AI agents.

## ⚡ The Solution: Stop Fighting Prompts, Teach Principles

Parlant flips the script on AI agent development. Instead of hoping your LLM will follow instructions, **Parlant ensures it**.

```python
# Traditional approach: Cross your fingers 🤞
system_prompt = "You are a helpful assistant. Please follow these 47 rules..."

# Parlant approach: Ensured compliance ✅
await agent.create_guideline(
    condition="Customer asks about refunds",
    action="Check order status first to see if eligible",
    tools=[check_order_status],
)
```

#### Parlant gives you all the structure you need to build customer-facing agents that behave exactly as your business requires:

- **[Journeys](https://parlant.io/docs/concepts/customization/journeys)**:
  Define clear customer journeys and how your agent should respond at each step.

- **[Behavioral Guidelines](https://parlant.io/docs/concepts/customization/guidelines)**:
  Easily craft agent behavior; Parlant will match the relevant elements contextually.

- **[Tool Use](https://parlant.io/docs/concepts/customization/tools)**:
  Attach external APIs, data fetchers, or backend services to specific interaction events.

- **[Domain Adaptation](https://parlant.io/docs/concepts/customization/glossary)**:
  Teach your agent domain-specific terminology and craft personalized responses.

- **[Canned Responses](https://parlant.io/docs/concepts/customization/canned-responses)**:
  Use response templates to eliminate hallucinations and guarantee style consistency.

- **[Explainability](https://parlant.io/docs/advanced/explainability)**:
  Understand why and when each guideline was matched and followed.

<div align="center">

## 🚀 Get Your Agent Running in 60 Seconds

</div>

```bash
pip install parlant
```

```python
import parlant.sdk as p

@p.tool
async def get_weather(context: p.ToolContext, city: str) -> p.ToolResult:
    # Your weather API logic here
    return p.ToolResult(f"Sunny, 72°F in {city}")

@p.tool
async def get_datetime(context: p.ToolContext) -> p.ToolResult:
    from datetime import datetime
    return p.ToolResult(datetime.now())

async def main():
    async with p.Server() as server:
        agent = await server.create_agent(
            name="WeatherBot",
            description="Helpful weather assistant"
        )

        # Have the agent's context be updated on every response (though
        # update interval is customizable) using a context variable.
        await agent.create_variable(name="current-datetime", tool=get_datetime)

        # Control and guide agent behavior with natural language
        await agent.create_guideline(
            condition="User asks about weather",
            action="Get current weather and provide a friendly response with suggestions",
            tools=[get_weather]
        )

        # Add other (reliably enforced) behavioral modeling elements
        # ...

        # 🎉 Test playground ready at http://localhost:8800
        # Integrate the official React widget into your app,
        # or follow the tutorial to build your own frontend!

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

**That's it!** Your agent is running with ensured rule-following behavior.

## 🎬 See It In Action

<img alt="Parlant Demo" src="https://github.com/emcie-co/parlant/blob/develop/docs/demo.gif?raw=true" width="100%" />

## 🔥 Why Developers Are Switching to Parlant

<table width="100%">
<tr>
  <td width="50%">

### 🏗️ **Traditional AI Frameworks**

  </td>
  <td width="50%">

### ⚡ **Parlant**

  </td>
</tr>
<tr>
<td width="50%">

- Write complex system prompts
- Hope the LLM follows them
- Debug unpredictable behaviors
- Scale by prompt engineering
- Cross fingers for reliability

</td>
<td width="50%">

- Define rules in natural language
- **Ensured** rule compliance
- Predictable, consistent behavior
- Scale by adding guidelines
- Production-ready from day one

</td>
</tr>
</table>

## 🎯 Perfect For Your Use Case

<div align="center">

|  **Financial Services**  |     **Healthcare**      |       **E-commerce**        |       **Legal Tech**       |
| :----------------------: | :---------------------: | :-------------------------: | :------------------------: |
| Compliance-first design  |   HIPAA-ready agents    |  Customer service at scale  |   Precise legal guidance   |
| Built-in risk management | Patient data protection | Order processing automation | Document review assistance |

</div>

## 🛠️ Enterprise-Grade Features

- **🧭 Conversational Journeys** - Lead the customer step-by-step to a goal
- **🎯 Dynamic Guideline Matching** - Context-aware rule application
- **🔧 Reliable Tool Integration** - APIs, databases, external services
- **📊 Conversation Analytics** - Deep insights into agent behavior
- **🔄 Iterative Refinement** - Continuously improve agent responses
- **🛡️ Built-in Guardrails** - Prevent hallucination and off-topic responses
- **📱 React Widget** - [Drop-in chat UI for any web app](https://github.com/emcie-co/parlant-chat-react)
- **🔍 Full Explainability** - Understand every decision your agent makes

## 📈 Join 8,000+ Developers Building Better AI

<div align="center">

**Companies using Parlant:**

_Financial institutions • Healthcare providers • Legal firms • E-commerce platforms_

[![Star History Chart](https://api.star-history.com/svg?repos=emcie-co/parlant&type=Date)](https://star-history.com/#emcie-co/parlant&Date)

</div>

## 🌟 What Developers Are Saying

> _"By far the most elegant conversational AI framework that I've come across! Developing with Parlant is pure joy."_ **— Vishal Ahuja, Senior Lead, Customer-Facing Conversational AI @ JPMorgan Chase**

## 🏃‍♂️ Quick Start Paths

<table border="0">
<tr>
<td><strong>🎯 I want to test it myself</strong></td>
<td><a href="https://www.parlant.io/docs/quickstart/installation">→ 5-minute quickstart</a></td>
</tr>
<tr>
<td><strong>🛠️ I want to see an example</strong></td>
<td><a href="https://www.parlant.io/docs/quickstart/examples">→ Healthcare agent example</a></td>
</tr>
<tr>
<td><strong>🚀 I want to get involved</strong></td>
<td><a href="https://discord.gg/duxWqxKk6J">→ Join our Discord community</a></td>
</tr>
</table>

## 🤝 Community & Support

- 💬 **[Discord Community](https://discord.gg/duxWqxKk6J)** - Get help from the team and community
- 📖 **[Documentation](https://parlant.io/docs/quickstart/installation)** - Comprehensive guides and examples
- 🐛 **[GitHub Issues](https://github.com/emcie-co/parlant/issues)** - Bug reports and feature requests
- 📧 **[Direct Support](https://parlant.io/contact)** - Direct line to our engineering team

## 📄 License

Apache 2.0 - Use it anywhere, including commercial projects.

---

<div align="center">

**Ready to build AI agents that actually work?**

⭐ **Star this repo** • 🚀 **[Try Parlant now](https://parlant.io/)** • 💬 **[Join Discord](https://discord.gg/duxWqxKk6J)**

_Built with ❤️ by the team at [Emcie](https://emcie.co)_

</div>

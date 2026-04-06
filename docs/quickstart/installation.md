# Installation

![Parlant Logo](https://parlant.io/logo/logo-full.svg)

**Parlant** is an open-source **Agentic Behavior Modeling Engine** for LLM agents, built to help developers quickly create customer-engaging, business-aligned conversational agents with control, clarity, and confidence.

It gives you all the structure you need to build customer-facing agents that behave exactly as your business requires:

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

## Installation
Parlant is available on both [GitHub](https://github.com/emcie-co/parlant) and [PyPI](https://pypi.org/project/parlant/) and works on multiple platforms (Windows, Mac, and Linux).

Please note that [Python 3.10](https://www.python.org/downloads/release/python-3105/) and up is required for Parlant to run properly.

```bash
pip install parlant
```

If you're feeling adventurous and want to try out new features, you can also install the latest development version directly from GitHub.

```bash
pip install git+https://github.com/emcie-co/parlant@develop
```

## Creating Your First Agent

Once installed, you can use the following code to spin up an initial, sample agent. You'll flesh out its behavior later.

```python
# main.py

import asyncio
import parlant.sdk as p

async def main():
  async with p.Server() as server:
    agent = await server.create_agent(
        name="Otto Carmen",
        description="You work at a car dealership",
    )

asyncio.run(main())
```

You'll notice Parlant follows the asynchronous programming paradigm with `async` and `await`. This is a powerful feature of Python that lets you to write code that can handle many tasks at once, allowing your agent to handle more concurrent requests in production.

If you're new to async programming, check out the [official Python documentation](https://docs.python.org/3/library/asyncio.html) for a quick introduction.

Parlant uses OpenAI as the default NLP provider, so you need to ensure you have `OPENAI_API_KEY` set in your environment.

Then, run the program!
```bash
export OPENAI_API_KEY="<YOUR_API_KEY>"
python main.py
```

Parlant supports multiple LLM providers by default, accessible via the `p.NLPServices` class. You can also add your own provider by implementing the `p.NLPService` interface, which you can learn how to do in the [Custom NLP Models](https://parlant.io/docs/advanced/custom-llms) section.

To use one of the built-in-providers, you can specify it when creating the server. For example:

```python
async with p.Server(nlp_service=p.NLPServices.cerebras) as server:
  ...
```

Note that you may need to install an additional "extra" package for some providers. For example, to use the Cerebras NLP service:

```bash
pip install parlant[cerebras]
```

Having said that, Parlant is observed to work best with [OpenAI](https://openai.com) and [Anthropic](https://www.anthropic.com) models, as these models are highly consistent in generating high-quality completions with valid JSON schemas—so we recommend using one of those if you're just starting out.

## Testing Your Agent

To test your installation, head over to [http://localhost:8800](http://localhost:8800) and start a new session with the agent.

![Post installation demo](https://parlant.io/img/post-installation-demo.gif)

## Creating Your First Guideline

Guidelines are the core of Parlant's behavior model. They allow you to define how your agent should respond to specific user inputs or conditions. Parlant cleverly manages guideline context for you, so you can add as many guidelines as you need without worrying about context overload or other scale issues.

```python
# main.py

import asyncio
import parlant.sdk as p

async def main():
  async with p.Server() as server:
    agent = await server.create_agent(
        name="Otto Carmen",
        description="You work at a car dealership",
    )

    ##############################
    ##    Add the following:    ##
    ##############################
    await agent.create_guideline(
        # This is when the guideline will be triggered
        condition="the customer greets you",
        # This is what the guideline instructs the agent to do
        action="offer a refreshing drink",
    )

asyncio.run(main())
```

Now re-run the program:
```bash
python main.py
```

Refresh [http://localhost:8800](http://localhost:8800), start a new session, and greet the agent. You should expect to be offered a drink!

## Using the Official React Widget

If your frontend project is built with React, the fastest and easiest way to start is to use the official Parlant React widget to integrate with the server.

Here's a basic code example to get started:

```jsx
import React from 'react';
import ParlantChatbox from 'parlant-chat-react';

function App() {
  return (
    <div>
      <h1>My Application</h1>
      <ParlantChatbox
        server="PARLANT_SERVER_URL"
        agentId="AGENT_ID"
      />
    </div>
  );
}

export default App;
```

For more documentation and customization, see the **GitHub repo:** https://github.com/emcie-co/parlant-chat-react.

```bash
npm install parlant-chat-react
```

## Installing Client SDK(s)

To create a custom frontend app that interacts with the Parlant server, we recommend installing our native client SDKs. We currently support Python and TypeScript (also works with JavaScript).

#### Python
```bash
pip install parlant-client
```

#### TypeScript/JavaScript
```bash
npm install parlant-client
```

You can review our tutorial on integrating a custom frontend here: [Custom Frontend Integration](https://parlant.io/docs/production/custom-frontend).

For other languages—they are coming soon! Meanwhile you can use the [REST API](https://parlant.io/docs/api/create-agent) directly.

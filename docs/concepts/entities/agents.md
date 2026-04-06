# Agents

In Parlant, an agent is a customized AI personality that interacts with customers as a single, competent entity. It is essentially, "the one you talk to", as opposed to some frameworks where an agent is a specialized task function.

Agents form the basic umbrella of conversational customization—all behavioral configurations affect agent behavior.

```python
import parlant.sdk as p

async with p.Server() as server:
    hexon = await server.agents.create(
        name="Hexon",
        description="Technical support specialist"
    )

    # Continue to model the agent's behavior using guidelines, journeys, etc....
```

> **Note:**
> 
> Note that a single Parlant server may host multiple agents, each with distinct roles and personalities.

Each agent can be uniquely configured with its own style, demeanor, and interaction patterns tailored to its target users. More importantly, different business units can own and maintain their specific agents. For example:
* IT Department manages **Hexon**
* Customer Success team oversees **Sprocket**
* Sales/Marketing controls **Piston**

This agent-based design creates natural boundaries for separation of concerns within Parlant


### Crafting an Agent's Identity
Imagine you're creating a new employee who will become the voice of your service. Just as you'd carefully consider the personality and approach of a human hire, crafting an agent's identity ultimately requires thoughtful consideration of its core characteristics—and, like any good hire, you can grow and adapt it based on real-world feedback.

As an example, let's follow the possible evolution of **Hexon**, our technical support specialist. In its first iteration, we might simply define it as "a technical support agent who helps users solve technical problems professionally and efficiently." After observing some interactions, we might notice that it comes across as too mechanical, failing to build trust with users.

So we refine its identity:

> "A technical support specialist who combines deep technical knowledge with patient explanation. You take pride in making complex concepts accessible without oversimplifying them. While you're always professional, you communicate with a warm, approachable tone. You believe that every technical issue is an opportunity to help users better understand their tools. When users are frustrated, you remain calm and empathetic, acknowledging their challenges while focusing on solutions."

As we observe more interactions, we might further refine this general identity. Perhaps we notice users respond better when Hexon shows more personality, or maybe we find certain technical discussions need more gravitas. The identity can evolve with these insights.

The key is to start with an identity that gives the agent its basic orientation, but remain open to refinement based on real interactions. Watch how users respond to the agent's mannerisms. Gather feedback from stakeholders. Adjust the identity accordingly.


### A Single Agent or Multiple Agents?
There's a frequent debate on whether to model user-facing agents as a single agent or multi-agent system. Parlant's position is a mix of both.

Generally speaking, managing complexity is easier when our solutions model the real world, because it makes us naturally have much more data with which to reason about design decisions, rather than trying to come up with something contrived. So instead of asking a very fundamental, "How should users interact with this agent?" we can instead ask something much more fruitful, like, "What would user expect based on their real-life experience?"

In practice, when we interact with human service representatives, there are certain expectations we've come to have from such experiences:
- If we're talking to an agent, they have the full context of our conversation. They're coherent. They don't suddenly just forget or unexpectedly change their interpretation of the situation.
- The agent we're talking to may not always be able to help us with everything. We may need to be transferred to another agent who specializes in some topic.
- We expect to be notified of such transfers. If they happen suddently or without our awareness, we take that as a careless customer experience.

You can see how insights from familiar, real-world usage patterns help us arrive at informed design decisions. By modeling agent interactions on real-world patterns, we not only better understand what outcomes to strive for, but it turns out that managing our agents' configuration becomes easier to reason about, too.

This is why Parlant's formal recommendation is to model AI agents after how human agents work. In other words, if you can see it being a single personality in a real-life use case, that means it should be represented as a single AI agent in Parlant. Incidentally, Parlant's filtration of relevant elements of the agent's conversation model allow you to manage quite a lot of complexity in a single agent, so you don't need to adopt a multi-agent approach if that was your concern.

> **Tip: The Failures of Multi-Agent Systems**
>
> There's an interesting paper on the [failures of multi-agent systems](https://arxiv.org/abs/2503.13657#:~:text=We%20present%20MAST%20%28Multi-Agent%20System%20Failure%20Taxonomy%29%2C%20the,over%20200%20tasks%2C%20involving%20six%20expert%20human%20annotators), despite their promise of modularity and specialization. It highlights how multi-agent systems often struggle with coordination, communication, and consistency, leading to unexpected behaviors and failures. This aligns with Parlant's approach of using a single agent to maintain coherence and context in conversations.

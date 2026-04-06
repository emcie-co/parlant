# User-Input Moderation

Adding content filtering to your AI agents helps achieve a more professional level of customer interactions. Here's why it matters.

### Understanding the Challenges

AI agents, being based on LLMs, are statistical pattern matchers that can be influenced by the nature of inputs they receive. Think of them like customer service representatives who benefit from clear boundaries about what conversations they should and shouldn't engage in.

#### Sensitive Topics

Some topics, like mental health or illicit activities, require professional human handling. While your agent might technically handle these topics, in practical use cases it's often better for it to avoid such conversations, or even redirect them to appropriate human resources.

#### Protection from Harassment

Customer interactions should remain professional, but some users might attempt to harass or abuse the agent (or others). This isn't just about maintaining decorum: LLMs (like humans) can in some cases be influenced by aggressive or inappropriate language, potentially affecting their responses.

To address such cases, Parlant integrates with moderation APIs, such as [OpenAI's Omni Moderation](https://openai.com/index/upgrading-the-moderation-api-with-our-new-multimodal-moderation-model/), to filter such interactions before they reach your agent.

### Enabling Input Moderation
To enable moderation, all you need to do is set a query parameter when creating events.

#### Python
```python
from parlant.client import ParlantClient

client = ParlantClient(base_url=SERVER_ADDRESS)

client.sessions.create_event(
    SESSION_ID,
    kind="message",
    source="customer",
    message=MESSAGE,
    moderation="auto",
)
```

#### TypeScript
```typescript
import { ParlantClient } from 'parlant-client';

const client = new ParlantClient({ environment: SERVER_ADDRESS });

await client.sessions.createEvent(SESSION_ID, {
     kind: "message",
     source: "customer",
     message: MESSAGE,
     moderation: "auto",
});
```

When customers send inappropriate messages, Parlant ensures that their content is not even visible to the agent; rather, all the agent sees is that a customer sent a message which has been "censored" for a some specific reason (e.g. harrassment, illicit behavior, etc.).

This integrates well with guidelines. For example, you may install a guideline such as:

> * **Condition:** the customer's last message is censored
> * **Action:** inform them that you can't help them with this query, and suggest they contact human support

From a UX perspective, this approach is superior to just "erroring out" when encountering such messages. Instead of seeing an error, the customer gets a polite and informative response. Better yet, the response can be controlled with guidelines and tools just as in any other situation.

## Jailbreak Protection

While your agent's guidelines aren't strictly security measures (as that's handled more robustly by backend permissions), maintaining presentable behavior is important even when some users might try to trick the agent into revealing its instructions or acting outside its intended boundaries.

Parlant's moderation system supports a special `paranoid` mode, which integrates with [Lakera Guard](https://www.lakera.ai/lakera-guard) (from the creators of the [Gandalf Challenge](https://gandalf.lakera.ai/baseline)) to prevent such manipulation attempts.

#### Python
```python
from parlant.client import ParlantClient

client = ParlantClient(base_url=SERVER_ADDRESS)

client.sessions.create_event(
    SESSION_ID,
    kind="message",
    source="customer",
    message=MESSAGE,
    moderation="paranoid",
)
```

#### TypeScript
```typescript
import { ParlantClient } from 'parlant-client';

const client = new ParlantClient({ environment: SERVER_ADDRESS });

await client.sessions.createEvent(SESSION_ID, {
     kind: "message",
     source: "customer",
     message: MESSAGE,
     moderation: "paranoid",
});
```

Note that to activate `paranoid` mode, you need to get an API key from Lakera and assign it to the environment variable `LAKERA_API_KEY` before starting the server.

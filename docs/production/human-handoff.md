# Human Handoff

Human handoff is a crucial aspect of customer service automation, especially when using AI agents. It allows for a smooth transition from automated responses to human expertise when necessary. This guide will walk you through the process of implementing human handoff in Parlant.

### The Call Center Model: Understanding Tier Structure

Modern call centers operate on a tiered system designed for efficiency and cost optimization. **Tier 1** representatives handle the majority of calls—typically 80% of customer inquiries—dealing with common issues like account questions, basic troubleshooting, and standard requests. These representatives are trained on frequently asked questions and standard procedures.

**Tier 2** representatives are more experienced and handle complex cases that require specialized knowledge, escalated issues, or nuanced problem-solving. While Tier 2 agents are more skilled and capable, they're also significantly more expensive to employ and maintain.

## Parlant's Approach

Our belief at Parlant, having worked deeply with generative AI, is that AI agents today can effectively automate most **Tier 1 work**, potentially reducing 80% of the customer service workforce while handling inquiries with:

- Professional communication standards
- High efficiency and 24/7 availability
- Graceful conversation management
- Full compliance with business rules

While Parlant's mission includes eventually automating Tier 2 use cases, we honestly believe the technology isn't quite there yet for the most complex scenarios. However, **automating Tier 1 is where the most significant cost savings and efficiency improvements can be achieved**.

## Integrated Human Handoff

Since we're automating Tier 1 requests, it makes sense to support human handoff (essentially to Tier 2) in an integrated way. Parlant allows you to seamlessly integrate with whatever external system you use—such as HubSpot, Zendesk, or custom support platforms.

Here's how to implement human handoff in Parlant:

## Setting Session to Manual Mode

The first step in human handoff is to stop the AI agent from automatically responding to new messages. You can accomplish this by setting the session to manual mode using a tool whenever the right conditions are met (e.g., when the AI agent cannot adequately assist the customer).:

### Using a Tool to Trigger Manual Mode

```python
@p.tool
async def initiate_human_handoff(context: p.ToolContext, reason: str) -> p.ToolResult:
    """Initiate handoff to a human agent when the AI cannot adequately help the customer."""

    # Set session to manual mode to stop automatic AI responses
    return p.ToolResult(
        data=f"Human handoff initiated because: {reason}",
        control={
            "mode": "manual"  # This stops automatic agent responses
        }
    )
```
```python
# Associate the tool with a guideline
await agent.create_guideline(
    condition="Customer requests human assistance",
    action="Initiate human handoff and explain the transition professionally",
    tools=[initiate_human_handoff]
)
```

### Manually Taking Over a Session
You can also manually set a session to manual mode from an external system using the Parlant client SDKs:

```typescript
// Using the Parlant client to manually set session mode
import { ParlantClient } from 'parlant-client';

const client = new ParlantClient({
    environment: "http://localhost:8800" // Your Parlant server URL
});

async function setSessionToManual(sessionId: string) {
    await client.sessions.update(sessionId, {
        mode: "manual" // Stops automatic AI responses
    });

    console.log(`Session ${sessionId} set to manual mode`);
}
```

## Managing Session Events Manually

Once a session is in manual mode, you can manage events manually using Parlant's REST API and client SDKs:

### Adding Human Operator Messages

#### Python

```python
from parlant.client import AsyncParlantClient

client = AsyncParlantClient(base_url="http://localhost:8800")

# Add message from human operator
async def send_human_message(session_id: str, message: str, operator_name: str):
    event = await client.sessions.create_event(
        session_id=session_id,
        kind="message",
        source="human_agent",  # Message from human operator
        message=message,
        participant={
            "id": OPTIONAL_ID_FOR_EXTERNAL_SYSTEM_REFERENCE,
            "display_name": operator_name
        }
    )
    return event
```

#### TypeScript

```typescript
import { ParlantClient } from 'parlant-client';

const client = new ParlantClient({
    environment: "http://localhost:8800"
});

// Add message from human operator
async function sendHumanMessage(sessionId: string, message: string, operatorName: string) {
    const event = await client.sessions.createEvent(sessionId, {
        kind: "message",
        source: "human_agent", // Message from human operator
        message: message,
        participant: {
            id: OPTIONAL_ID_FOR_EXTERNAL_SYSTEM_REFERENCE,
            display_name: operatorName
        }
    });

    return event;
}
```

### Adding Messages on Behalf of AI Agent

Sometimes human operators may want to send messages that appear to come from the AI agent:

#### Python

```python
# Send message on behalf of AI agent
async def send_message_as_ai(session_id: str, message: str):
    event = await client.sessions.create_event(
        session_id=session_id,
        kind="message",
        source="human_agent_on_behalf_of_ai_agent",  # Human sending as AI
        message=message
    )
    return event
```

#### TypeScript

```typescript
// Send message on behalf of AI agent
async function sendMessageAsAI(sessionId: string, message: string) {
    const event = await client.sessions.createEvent(sessionId, {
        kind: "message",
        source: "human_agent_on_behalf_of_ai_agent", // Human sending as AI
        message: message
    });

    return event;
}
```

## Receiving Events from Parlant

To integrate with external systems, you need to monitor Parlant sessions for new events:

### Event Polling Pattern - Parlant as Single Source of Truth

The key to proper integration is treating Parlant sessions as the **single source of truth** for conversation state. Only read events from Parlant and update your external system accordingly. Even when you send events to Parlant, wait for them to be returned from `list_events()` before showing them in your external system UI.

#### Python

```python
async def monitor_session_events(session_id: str, last_offset: int = 0):
    """
    Poll for new events in a Parlant session.
    Parlant session is the single source of truth - all message display
    should be based on events returned from list_events().
    """

    while True:
        try:
            # Wait for new events with timeout
            events = await client.sessions.list_events(
                session_id=session_id,
                kinds="message",
                min_offset=last_offset,
                wait_for_data=30  # Wait up to 30 seconds for new events
            )

            for event in events:
                # Process ALL message events from Parlant
                await process_event_for_display(event)
                last_offset = max(last_offset, event.offset + 1)

        except Exception as e:
            # Try again after timeout from list_events()
            continue

async def process_event_for_display(event):
    """
    Process incoming events from Parlant for display in external system.
    """
    # Display all message events in external system chat UI
    await update_external_chat_display(
        message=event.data.get('message'),
        source=event.source,
        participant_name=event.data.get('participant', {}).get('display_name', 'Unknown'),
        timestamp=event.created_at,
        event_id=event.id
    )

async def update_external_chat_display(message: str, source: str, participant_name: str,
                                     timestamp: str, event_id: str):
    """Update the chat UI in your external system (HubSpot, Zendesk, etc.)"""

    # Map Parlant sources to your UI display logic
    if source == "customer":
        await add_customer_message_to_ui(message, timestamp, event_id)
    elif source == "ai_agent":
        await add_ai_message_to_ui(message, timestamp, event_id)
    elif source == "human_agent":
        await add_human_agent_message_to_ui(message, participant_name, timestamp, event_id)
    elif source == "human_agent_on_behalf_of_ai_agent":
        # Display as AI message but track that human sent it
        await add_ai_message_to_ui(message, timestamp, event_id, sent_by_human=True)
```

#### TypeScript

```typescript
async function monitorSessionEvents(sessionId: string, lastOffset: number = 0): Promise<void> {
    // Parlant session is the single source of truth for conversation state
    while (true) {
        try {
            // Poll for new events from Parlant
            const events = await client.sessions.listEvents(sessionId, {
                minOffset: lastOffset,
                kinds: "message",
                waitForData: 30 // Wait up to 30 seconds
            });

            for (const event of events) {
                // Process ALL message events for display
                await processEventForDisplay(event);
                lastOffset = Math.max(lastOffset, event.offset + 1);
            }

        } catch (error) {
            // Try again after timeout from listEvents()
        }
    }
}

async function processEventForDisplay(event: any): Promise<void> {
    /**
     * Process events from Parlant for display in external system.
     */
    await updateExternalChatDisplay({
        message: event.data.message,
        source: event.source,
        participantName: event.data.participant?.display_name,
        timestamp: event.createdAt,
        eventId: event.id
    });
}

interface DisplayMessageParams {
    message: string;
    source: string;
    participantName: string;
    timestamp: string;
    eventId: string;
}

async function updateExternalChatDisplay(params: DisplayMessageParams): Promise<void> {
    /**
     * Update the chat UI in your external system (HubSpot, Zendesk, etc.)
     * based on what Parlant shows as the authoritative conversation state.
     */
    const { message, source, participantName, timestamp, eventId } = params;

    switch (source) {
        case "customer":
            await addCustomerMessageToUI(message, timestamp, eventId);
            break;
        case "ai_agent":
            await addAIMessageToUI(message, timestamp, eventId);
            break;
        case "human_agent":
            await addHumanAgentMessageToUI(message, participantName, timestamp, eventId);
            break;
        case "human_agent_on_behalf_of_ai_agent":
            // Display as AI message but track that human sent it
            await addAIMessageToUI(message, timestamp, eventId, true);
            break;
    }
}

async function addCustomerMessageToUI(message: string, timestamp: string, eventId: string): Promise<void> {
    // Implement your UI update logic for customer messages
}

async function addAIMessageToUI(message: string, timestamp: string, eventId: string, sentByHuman: boolean = false): Promise<void> {
    // Implement your UI update logic for AI messages
}

async function addHumanAgentMessageToUI(message: string, agentName: string, timestamp: string, eventId: string): Promise<void> {
    // Implement your UI update logic for human agent messages
}
```

## Best Practices for Human Handoff

1. **Clear Transition Messages**: Always inform customers when they're being transferred to a human agent and explain why. You can achieve this using guidelines.

2. **Context Preservation**: Ensure human agents have access to the full conversation history from the AI interaction. Do this by synchronizing the session's events with your external system.

3. **Seamless Experience**: Use `human_agent_on_behalf_of_ai_agent` source when you want to maintain the illusion of a single agent experience. Customers don't always need to know they're interacting with a human.

4. **Monitoring and Analytics**: Track handoff rates, reasons, and resolution outcomes to improve your AI agent over time. Implement lessons learned from human interactions to refine agent responses and guidelines.

5. **Return to AI**: Consider implementing logic to return sessions to automatic mode when appropriate.


This integration approach allows you to leverage AI agents for efficient Tier 1 support while ensuring complex issues can be seamlessly escalated to human experts, providing the best of both worlds for customer service.

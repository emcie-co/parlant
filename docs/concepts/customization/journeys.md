# Journeys

There are many use cases where you want your agent to follow a specific flow of conversation, such as booking a trip, troubleshooting an issue, or otherwise guiding a user through a conversational process in the intended manner.

In Parlant, this can be achieved easily and reliably using **Journeys**.

#### Journey Structure

Journeys have 4 important components:
1. **Title:** A short, descriptive name for the journey, to differentiate it from other journeys.
1. **Conditions:** These contextual queries determine when a journey should be active.
1. **Description:** This lets you describe the nature of the journey, including motivating or orientating notes, if needed.
1. **States & Transitions**: A state diagram that communicates to the agent what the ideal flow is.

> **Balancing Rigidity vs. Flexibility**
>
> In traditional conversational frameworks, flows are rigidly defined, with each state and transition being strictly followed to the letter.
>
> While this type of approach is easy to reason about and implement, it very often leads to frustrating user experiences when the agent is unable to adapt to the customer's interaction patterns. This "one-size-fits-all" approach doesn't account for the nuances of human conversation—leading to user disengagement and dissatisfaction, and ultimately resulting in an unused agent.
>
> Parlant implements a "lessons learned" approach, allowing agents to traverse the journey's states in a more natural way. They can choose to skip states, revisit previous states, or even jump ahead to later states in an adaptive manner.
>
> As such, journeys are not meant to be followed rigidly, but rather to serve as a guiding framework for the agent. The agent will strive to follow the flow as strictly as it can, while still maintaining an adaptive approach toward the customer's interaction patterns.

## A Worked Example

Consider the following example for a journey in a travel agent:
> * **Title:** Book Flight
> * **Conditions:** The customer requested to book a flight
> * **Description:** This journey guides the customer through the flight booking process.

```mermaid
%%{init: { "theme": "forest" }}%%
stateDiagram-v2
    direction LR
    state "Where to?" as A
    state "Dates?" as B
    state "Load destinations" as Ca
    state "Suggest" as Cb
    state "Confirm" as E
    state "Book" as F
    state "Provide ticket" as G
    [*] --> A
    A --> Ca: Don't know
    Ca --> Cb
    A --> B: Destination provided
    Cb --> B: Destination selected
    B --> E
    E --> F: Yes
    E --> [*]: No
    F --> G
    G --> [*]

    style Ca fill:#ffeecc,stroke:#333,stroke-width:1px
    style F fill:#ffeecc,stroke:#333,stroke-width:1px
```

This journey will be activated when the customer asks to book a flight. The agent will then strive to follow the flow while maintaining an adaptive approach at the pace of the customer, yet ensuring that all necessary information is collected.

#### Implementing the Journey
Before we learn more about how journeys work, let's look at how we would implement the journey above:

```python
async def create_book_flight_journey(agent: p.Agent):
    journey = await agent.create_journey(
        title="Book Flight",
        conditions=["The customer requested to book a flight"],
        description="This journey guides the customer through the flight booking process.",
    )

    t1 = await journey.initial_state.transition_to(chat_state="Ask if they have a destination in mind")

    #  Branch out based on the customer's response
    t2 = await t1.target.transition_to(condition="They do", chat_state="Get dates of travel")

    t3a = await t1.target.transition_to(condition="They don't", tool_state=load_popular_destinations)
    t3b = await t3a.target.transition_to(chat_state="Recommend a destination")

    # Merge back to the main path after choosing a destination.
    # This is done by transitioning into an existing state node.
    await t3b.target.transition_to(state=t2.target, condition="Destination selected")

    t4 = await t2.target.transition_to(chat_state="Confirm details")

    t5a = await t4.target.transition_to(tool_state=book_flight)
    t5b = await t5a.target.transition_to(chat_state="Provide ticket details")
```

## States and Transitions
A journey is modeled after a state diagram, which is a directed graph in which each node represents a **state** and each edge represents a **transition** (which may be associated with a condition).

```mermaid
stateDiagram-v2
    direction LR
    state "CHAT STATE" as A
    state "TOOL STATE" as B
    state "CHAT STATE" as C
    state "CHAT STATE" as D
    state "FORK STATE" as E
    state "CHAT STATE" as F
    state "CHAT STATE" as G

    [*] --> A: INITIAL
    A --> B: CONDITIONAL
    A --> C: CONDITIONAL
    B --> D: DIRECT
    C --> E: DIRECT
    D --> E: DIRECT
    E --> F: CONDITIONAL
    E --> G: CONDITIONAL
    F --> [*]: END
    G --> [*]: END

    style B fill:#ffeecc,stroke:#333,stroke-width:1px
```

#### States
1. **Chat States:** While in this state, the agent will chat with the customer while being guided by the state's action. The agent may spend multiple turns in this state, until it decides to transition to another state.
```python
t = await state.transition_to(chat_state=CONVERSATIONAL_INSTRUCTION)
```
2. **Tool States:** In this state, the agent will call an external tool to perform an action and load its result into the context. A tool state must be followed by a chat state, which will usually be used to present the tool's result to the customer.
```python
t = await state.transition_to(tool_state=TOOL)
```
```python
t = await state.transition_to(tool_state=TOOL, tool_instruction=OPTIONAL_HINT_ON_HOW_TO_USE_TOOL)
```

> **Transitioning from Tool to Chat**
>
> When transitioning from a tool state to a chat state, the agent will automatically load the tool's result into the context, so you can use it in the chat state's action. Note that a tool state cannot transition to another tool state; it must always be followed by a chat state.
>
> This is by design, as tool usage can incur noticeable latency in agentic applications. Instead of using sequential tool states, you should use a single tool state to perform all necessary actions, and then follow it with a chat state to present the results to the customer.

#### Transitions
1. **Direct Transitions:** These transitions should always be taken. They move the conversation forward without branching.
2. **Conditional Transitions:** These transitions are only taken if/when their associated condition is met.
```python
t = await state.transition_to(chat_state=CONVERSATIONAL_INSTRUCTION, condition=CONDITION)
```
```python
t = await state.transition_to(tool_state=TOOL, condition=CONDITION)
```

In most cases, you'd be using the `transition_to()` overload that takes a `chat_state` or `tool_state` argument, which will automatically create the transition's target state for you. However, you can also use the `transition_to()` overload that takes a `state` argument, which allows you to transition to an existing state node in the journey.

```python
t = await state.transition_to(state=EXISTING_STATE)
```
```python
t = await state.transition_to(state=EXISTING_STATE, condition=CONDITION)
```

> **Combining Conditional and Direct Transitions**
>
> If a state has a conditional transition to another state, it cannot also have a direct transition coming out of it. This is because the engine would not be able to logically determine which transition to take when the condition is met. The SDK enforces this rule.

#### Fork States
Journeys also support a special kind of state, called a **fork state**.

In this state, the agent will evaluate conditions and branch the conversation flow accordingly. While, strictly speaking, such branching can be modeled without fork states, they are sometimes a useful modeling tool for keeping the conversation flow clear, explicit, and organized.

```python
fork = await state.fork()

t1 = await fork.transition_to(chat_state=CONVERSATIONAL_INSTRUCTION, condition=CONDITION_1)
t2 = await fork.transition_to(chat_state=CONVERSATIONAL_INSTRUCTION, condition=CONDITION_2)
t3 = await fork.transition_to(tool_state=TOOL, condition=CONDITION_3)
```

> **Visualizing Your Journey**
>
> Building a state diagram in code can sometimes be a bit confusing. It's often useful to visualize the journey as you build it, to ensure that the flow is clear, logical, and as you intend. Here's how it's done:
>
> 1. Visit `http://localhost:8800/journeys` in your browser.
> 2. Copy the ID of the journey you want to visualize.
> 3. Visit `http://localhost:8800/journeys/<JOURNEY_ID>/mermaid` in your browser, replacing `<JOURNEY_ID>` with the ID you copied.
> 4. Copy the generated Mermaid diagram code.
> 5. Paste it into a [Mermaid live editor](https://mermaid.live/) to visualize the journey.

## Journey vs. Task Automation
If you look at how the engine works with journeys, it means that journeys should not be used to guide the model on how to automate tasks. Instead, journeys are used by the agent to self-orientate and guide the conversation flow according to your preferences.

This is a good time to recall the importance of separating **business logic** from **conversation logic**. The former is best handled by custom, [dedicated tools](https://parlant.io/docs/concepts/customization/tools) (which may or may not use LLMs internally), while the latter is best handled by the conversational engine.

#### Do's and Don'ts

**DON'T**

The following is ***not*** a valid journey, as it does not represent a conversation flow but rather a task automation flow.

```mermaid
stateDiagram-v2
    direction LR
    state "Find user ID" as A
    state "Load personal preferences" as B
    state "Send email" as C
    [*] --> A
    A --> B
    B --> C: Email notifications enabled
    B --> [*]: Email notifications disabled

    style A fill:#ffeecc,stroke:#333,stroke-width:1px
    style B fill:#ffeecc,stroke:#333,stroke-width:1px
    style C fill:#ffeecc,stroke:#333,stroke-width:1px

```

**DO**

The following is a valid journey, as it represents a conversation protocol that guides the customer through a process.

```mermaid
stateDiagram-v2
    direction LR
    state "Ask for order number" as A
    state "Get order details" as B
    state "Process refund" as C
    state "Transfer to human" as D
    [*] --> A
    A --> B
    B --> C: Eligible for refund
    B --> D: Not eligible for refund
    C --> [*]
    D --> [*]

    style B fill:#ffeecc,stroke:#333,stroke-width:1px
```

## Context Management

LLMs are a magnificent creation, built on the principle of [statistical attention](https://arxiv.org/abs/1706.03762) in text; yet, their attention span is painfully finite. When it comes to following instructions, they need help.

Behind the scenes, Parlant ensures that agent responses are aligned with expectations by dynamically managing the LLM's context to only include the relevant journeys at each point.

It does this using the `GuidelineMatcher`, essentially matching the current conversation context with the relevant journeys' conditions—which, behind the scenes are basically observational (non-actionable) guidelines.

```mermaid
%%{init: {'sequence': {'mirrorActors': false}}}%%
sequenceDiagram
    participant Engine
    participant GuidelineMatcher
    participant JourneyStore
    participant MessageComposer

    Engine ->> GuidelineMatcher: match guidelines
    GuidelineMatcher -->> Engine: <guidelines>
    Engine ->> JourneyStore: get journeys for matched conditions
    JourneyStore -->> Engine: <journeys>
    Engine ->> GuidelineMatcher: match journey states
    GuidelineMatcher -->> Engine: <journey states>
    Engine ->> MessageComposer: <journey states, guidelines>
    MessageComposer -->> Engine: <well-guided message>
```

Before each response, Parlant only loads the guidelines and journeys that are relevant to the conversation's current state. This dynamic management keeps the LLM's "cognitive load" minimal, maximizing its attention and, consequently, the alignment of each response with expected behavior.

> **Latency Optimizations**
>
> This back and forth approach is implemented in an optimized algorithm that minimizes response latency.
>
> The engine first tries to predict which journeys will be activated based on the current conversation context. Given this prediction, it attempts to match the relevant journeys' states in parallel to guideline matching, shaving seconds off the response latency.
>
> Only when this prediction fails (i.e., when other journeys were activated) does it incur the extra step to match their states, as well.

## Journey-Scoped Guidelines

You can add journey-scoped [guidelines](https://parlant.io/docs/concepts/customization/guidelines) that can only be activated when their dependent journeys are also active. At all other times, these guidelines would be ignored.

Using journey-scoped guidelines is the recommended way to handle digressions from the journey's main flow in deliberate ways. It also helps you maintain a clean and organized conversation model, ensuring that certain guidelines are only evaluated and activated in their intended context.

> **Instruction Precedence**
>
> Please note that, in general, Parlant agents give more weight to guidelines than to journey states, as guidelines are treated as more specific behavioral overrides. This means that, if a guideline is matched, it will tend to take precedence over the active journey states.

```python
@p.tool
async def transfer_to_human_agent(context: p.ToolContext) -> p.ToolResult:
    ...

guideline = await journey.create_guideline(
    condition="the customer says they're unable to pay"
    action="connect them with a human agent",
    tools=[transfer_to_human_agent],
),
```

> **Learn More**
>
> To learn more about guidelines, check out the [Guidelines](https://parlant.io/docs/concepts/customization/guidelines) page.

## Journey-Scoped Canned Responses
You can also attach canned responses to journeys, scoping them to those journeys such that they will only be considered when their dependent journeys are active.

```python
await journey.create_canned_response(
    template="What destination are you interested in?",
)

await journey.create_canned_response(
    template="I'm sorry, but I can't assist with that right now. Shall we go on with booking your flight?",
)
```

#### State-Scoped Canned Responses
You can also associate specific canned responses with specific states within a journey.

There are two modes for state-scoped canned responses: **Explicit Consideration** and **Exclusive Consideration**.

1. **Explicit Consideration:** In this mode, the agent will ensure that the associated responses are always considered for selection when in that state. This is done by creating the canned response under the `journey` or `agent` objects.
```python
await state.transition_to(
    chat_state="Ask if they have a destination in mind",
    canned_responses=[
        await journey.create_canned_response(
            template="What destination are you interested in?",
        ),
    ],
)
```
2. **Exclusive Consideration:** In this mode, the agent will only consider the associated responses when in that state. It won't use these responses at any other time. This is done by creating the canned response under the `server` object.

```python
await state.transition_to(
    chat_state="Ask if they have a destination in mind",
    canned_responses=[
        await server.create_canned_response(
            template="What destination are you interested in?",
        ),
    ],
)
```

> **Learn More**
>
> To learn more about canned responses, check out the [Canned Responses](https://parlant.io/docs/concepts/customization/canned-responses) page.

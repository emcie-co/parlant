# Tools

Parlant provides a guided approach to tool usage, tightly integrated with its guidance system.

Parlant's tool-calling approach is built from the ground up. It's probably more comprehensive than the tool-calling mechanisms you may be familiar with from most LLM APIs (including [MCP](https://modelcontextprotocol.io/introduction)). This is because it's built to enable a deep, seamless integration with its guidance-based behavior control.

### Understanding Tool Usage
In Parlant, tools are always associated with specific guidelines. A tool only executes when its associated guideline is matched to the conversation. This design creates a clear chain of intent: guidelines determine when and why tools are used, rather than leaving it to the LLM's judgment (which is rife with errors).

In Parlant, **business logic** (encoded in tools) is consciously separated from **presentation** (user interface) concerns, or the customer-facing behavior that is controlled by guidelines.

This allows you to have developers work out API logic in code, with full control—offering these tools in the "tool shed" of an agent. Then, business experts can independently define natural-language guidelines that determine when and how these tools are used, without needing to get involved with the underlying code. This separation of concerns is a key design principle in Parlant, allowing for cleaner, more maintainable systems.

#### Conversational UI vs. Graphical UI
As an analogy, you can think of Guidelines and Tools like Widgets and Event Handlers in Graphical UI frameworks. A GUI Button has an `onClick` event which we can associate with some API function to say, _"When this button is clicked, run this function."._ In the same way, in Parlant (which is essentally a Conversational UI framework) the Guideline is like the Button, the Tool is like the API function, and the association connects the two (like registering an event handler) to say, _"When this guideline is applied, run this tool."_

Here's a concrete example to illustrate these concepts:
> * **Condition:** The user asks about service features
> * **Action:** Understand their intent and consult the docs to answer
> * **Tools Associations:** `[query_docs(user_query)]`

Here, the documentation query tool only runs after the guideline instructs that we should be consulting documentation for this user interaction.


## Writing Tools

To write a tool, you need to define a function that takes a `ToolContext` as its first argument, followed by any other parameters you want to pass to the tool. The function should return a `ToolResult`.

Here's the basic structure of any tool in a Parlant agent:

```python
import parlant.sdk as p

@p.tool
async def tool_name(context: p.ToolContext, param1: str, param2: int) -> p.ToolResult:
  """Multi-line tool description.
  This is readable by the agent and helps it decide if/when to run this tool.
  """
  ...
```

To illustrate that more concretely, here's a simple example of a tool that fetches products into the agent's context:

```python
import parlant.sdk as p

@p.tool
async def find_products(context: p.ToolContext, query: str) -> p.ToolResult:
    """Fetch products based on a natural-language search query."""

    # Simulate fetching the balance from a database or API
    products = await MY_DB.get_products(query=query)

    return p.ToolResult(balance)
```

#### Optional Parameters
You can also define optional parameters in your tool by using the `Optional` type from the `typing` module. This allows the tool to be called without providing a value for that parameter.

```python
from typing import Optional
import parlant.sdk as p

@p.tool
async def find_products(
  context: p.ToolContext,
  query: str,
  limit: Optional[int]
) -> p.ToolResult:
    default_limit = 10
    products = await MY_DB.get_products(query=query, limit=limit or default_limit)

    return p.ToolResult(products)
```

## Connecting Tools to Your Agent
To allow your agent to run a tool, you need to specify the conditions under which it may be evaluated and called. As stated above, this approach helps to eliminate false-positive tool calls, which is a common problem with LLMs.

```python
@p.tool
async def my_tool(context: p.ToolContext) -> p.ToolResult:
  ...

await agent.create_guideline(
    condition=CONDITION,
    action=ACTION,
    tools=[my_tool],
)
```

If you the tool itself implies the action, you can skip specifying it manually by letting Parlant figure it out from the tool's description. Here's how that would look:

```python
await agent.attach_tool(condition=CONDITION, tool=my_tool)
```

## Tool Result
The `ToolResult` is a special object that encapsulates the result of a tool call.

It has five properties you can use. While most of the time you will only use the `data` property, it's worth knowing about the others, as they enable interesting use cases: `data`, `metadata`, `control`, `canned_responses`, and `canned_response_fields`.

> **Tool Result Lifespan**
>
> Unlike many general-purpose agent frameworks, Parlant is specifically and deliberately optimized for building conversational agents. As such, its architecture optimizes the default behavior of tool-calling for a conversational application.
>
> To this end, tool results are saved in the session by default, and are therefore available for the agent reference's throughout the entire interaction session. This means that subsequent guideline matching and tool calls are automatically informed by the previous results of tools. This is useful for results that need to be referenced later, such as account balances, item IDs, or other product information.
>
> For example, if you call a tool that returns product names and IDs, and—after the customer responds by selecting a specific product—you then call another tool that takes a `product_id` parameter, the agent will be able to use the previous tool's result to fill in that parameter automatically from context. For example:
>
> ```python
> @p.tool
> async def get_products(context: p.ToolContext, query: str) -> p.ToolResult:
>     products = await MY_DB.get_products(query=query)
>     return p.ToolResult(data=products)
>
> @p.tool
> async def get_product_details(context: p.ToolContext, product_id: str) -> p.ToolResult:
>     # The agent will be able to parameterize the right `product_id`
>     # if the previous tool call was already made in the session.
>     product = await MY_DB.get_product(product_id=product_id)
>     return p.ToolResult(data=product)
> ```


### Tool Result Properties
Let's look at each of these properties, what they're used for, and how to use them:


#### Data
The `data` property contains the main output of the tool. This can be any JSON-serializable type, such as a string, list, or dictionary.

This is the only property of the `ToolResult` that is _always_ required, as it is the only one that the agent uses to understand the history of interaction events. Meaning, if you don't return anything in the `data` property, the agent will not be informed about your result, and it will not be able to it to navigate the interaction.

```python
# Example 1
return p.ToolResult(data="This is the result of my tool call")

# Example 2
return p.ToolResult(data={"appointments": [
  { "id": "123", "date": "2023-10-01 10:00" },
  { "id": "456", "date": "2023-10-02 11:00" },
]})
```

#### Metadata
The `metadata` property is an optional dictionary that can be used to store additional information about the tool call.

The agent is not aware of this metadata at all, but you can fetch it from the response using the REST API client. This makes it useful for sending back additional information about the response that can add value in your frontend.

A classic use case here is to return RAG information sources (e.g., URLs, document IDs, etc.) that can be used to display the source of the information in the frontend. Another one is to return image links to generated charts or other visualizations that can be displayed in the frontend.

```python
return p.ToolResult(
  data=ANSWER,
  metadata={ "sources": [{"url": s.url, "title": s.title} for s in ANSWER_SOURCES]},
)
```

```python
return p.ToolResult(
  data="The profit margin is 20%",
  metadata={ "generated_chart_url": "https://example.com/chart.png" },
)
```

> **Mind the Lifespan**
>
> Since metadata is primarily useful when accessing session events, it generally only makes sense to use it with `lifespan: "session"` (the default). If you use `lifespan: "response"`, the metadata will not be available in the session events, hence not accessible to the frontend.

#### Control
The `control` property lets you specify control directives for the agent and the engine.

There are currently two control directives you can use:
- `"mode": p.SessionMode`: This allows you to put the session in manual mode, which means the agent will not automatically generate responses. This is particularly useful for human-handoff scenarios, where you want to pause the agent's automatic responses and let a human operator take over.
- `"lifespan": p.Lifespan`: This controls how long the `ToolResult` should live. There are two options:
  - `"session"`: The result will be saved and made available to the agent for the entire session. This is the default.
  - `"response"`: The result will only be available for the current response. This is useful for temporary results that are not needed beyond the current response, such as reporting errors, or providing very transient information.

```python
return p.ToolResult(
  data="Transferring to a human agent",
  # Once this tool result is returned, the agent will not generate any more responses
  # until the session is put back on automatic mode using an API call.
  control={ "mode": "manual" },
)
```

```python
return p.ToolResult(
  data="Encountered an error while fetching data",
  # This tool result will not be saved in the session.
  # The agent will only be aware of it during the current response.
  control={ "lifespan": "response" },
)
```

#### Canned Responses
Tools can also return complete canned responses for consideration, as well as fields to be substituted during canned response rendering. For more information about canned response properties, refer to the [Canned Responses](https://parlant.io/docs/concepts/customization/canned-responses) section.

## Guideline Reevaluation Based on Tool Results

In some cases, tool results can influence which guidelines become relevant. Here's an example:

Consider a banking agent handling transfers. When a user requests a transfer, a guideline with the condition the user wants to make a transfer activates the `get_user_account_balance()` tool to check available funds. This tool returns the current balance, which can then trigger additional guideline matches based on its return value.

For instance, if the balance is below $500, we might have a low-balance guideline activate, instructing the agent to say something like: _"I see your current balance is low. Are you sure you want to proceed with this transfer? This transaction might put you at risk of overdraft fees."_

In Parlant, you can mark certain guidelines for reevaluation after a tool call. This means that once the tool is called, the guideline matcher will re-evaluate the session to see if any new guidelines should be activated based on the tool's results—*after* running the tool but *before* generating the response.

Here's how you can do that:

```python
# The agent will ensure to reevaluate this guideline after running this tool
await guideline.reevaluate_after(my_tool)
```


## Best Practices

#### A Note on Natural Language Programming
While LLMs excel at conversational tasks, they struggle with complex logical operations and multi-step planning. Recent research in LLM architectures shows that even advanced models have difficulty with consistent logical reasoning and sequential decision-making. The "planning problem" in LLMs—breaking down complex tasks into ordered steps and synthesizing conclusions—remains a significant unsolved problem when consistently at scale is required.

Given these limitations, Parlant takes a pragmatic approach: Separate logic in code from behavior modeling. Instead of embedding business logic in guidelines, Parlant encourages a clean separation between conversational behavior and underlying business operations.

Consider your tools as your place for deterministic, programmatic business logic, and guidelines as your conversational interface design. This separation creates cleaner, more maintainable, and more reliable systems.

> **Agentic API Design**
>
> To learn more about best-practices for designing your agent's tools, we recommend reading our blog post on [Agentic Backends](https://parlant.io/blog/what-no-one-tells-you-about-agentic-api-design).

#### Examples

**1. E-commerce Product Recommendations:**

DON'T

Complex business logic in guideline, over-relying on the LLM
> * **Guideline Action:**
> If user mentions sports, check their purchase history.
> If they bought running gear, recommend premium shoes.
> If they're new, suggest starter kit.
> * **Tool Associations:** `[get_product_catalog]`

DO

Logic goes in the coded recommendation engine, where it belongs

> * **Guideline Action:** Offer personalized recommendations
> * **Tool Associations:** `[get_personalized_recommendations]`

**2. Financial Advisory:**

DON'T

Financial analysis logic relies on unreliable LLM numeric comprehension
> * **Guideline Action:** Check account balance and recent transactions.
           If spending exceeds 80% of usual pattern, suggest budget review.
           If investment returns are down, recommend portfolio adjustment.
> * **Tool Associations:** `[get_account_data]`

DO

Financial analysis logic is handled reliably in code

> * **Guideline Action:** Get personalized financial insights
> * **Tool Associations:** `[get_financial_insights]`


## Tool Context

The `ToolContext` parameter is a special object that provides the tool with contextual information and utilities.

Let's look at some of the most useful attributes and methods available in the `ToolContext`:

1. `agent_id`: The unique identifier of the agent that is calling the tool.
2. `customer_id`: The unique identifier of the customer interacting with the agent.
3. `session_id`: The unique identifier of the current session.
4. `emit_message(message: str)`: A method to send a message back to the customer. This can be used to report progress during a long-running tool call.
5. `emit_status(status: p.SessionStatus)`: A method to update the [session status](https://parlant.io/docs/concepts/sessions#status-event).

#### Accessing Server Objects (Agent, Customer, etc.)

You can also access the server object from the `ToolContext`, giving you access to your agents, guidelines, and other server-level resources.

To access the server object, use `p.ToolContextAccessor` as follows:

```python
import parlant.sdk as p

@p.tool
async def my_tool(context: p.ToolContext) -> p.ToolResult:
    server = p.ToolContextAccessor(context).server

    # Access the current agent using the agent_id from the context
    agent = await server.get_agent(id=context.agent_id)

    # Access the current customer
    customer = await server.get_customer(id=context.customer_id)

    # ...

    return p.ToolResult(...)
```

#### Secure Data Access

Suppose you need to build a tool that retrieves or displays data private to different customers.

A naive approach would be to ask the customer to identify themselves and use that as an access token into the right data. But this approach is highly insecure, as it relies on the LLM for identifying the user. The LLM can get it wrong or, worse yet, be manipulated by malicious users.

A better and more reliable way to do this is to [register your customers](https://parlant.io/docs/concepts/entities/customers#registering-customers) with Parlant and use the information available programmatically, which is contained in the `ToolContext` parameter of your tool.

Here’s how that would look in practice:

```python
@p.tool
async def get_transactions(context: p.ToolContext) -> p.ToolResult:
    transactions = await DB.get_transactions(context.customer_id)
    return p.ToolResult(transactions)
```

## Tool Insights and Parameter Options

Because Parlant's architecture is radically modular, components like guideline matching, tool calling and message composition operate independently. While this non-monolithic approach offers many advantages in managing its complex semantic logic, it also requires it to communicate contextual awareness across these components.

**Tool Insights** is a bridging component between tool calling and message composition, ensuring the composition component is informed when a tool couldn't be called for some reason—for example, due to missing required parameters.


This allows the agent to respond more intelligently. For example, if it had no knowledge of when an appropriate tool couldn't be called, it might generate a misleading response. But with tool insights, the agent recognizes missing information and, if needed, can prompt the customer for the required tool arguments automatically.

#### Tool Parameter Options

To allow you to enhance the baseline behavior of Tool Insights, you can make use of **ToolParameterOptions**, a special parameter annotation which adds more control over how tool parameters are handled and communicated.

While Tool Insights helps the agent recognize when and why a tool call fails, **ToolParameterOptions** goes a step further by guiding the agent on when and how to explain specific missing parameters.

```python
from typing import Annotated
import parlant.sdk as p

@p.tool
async def transfer_money(
  context: p.ToolContext,
  amount: Annotated[float, p.ToolParameterOptions(
    source="customer",  # Only the customer can provide this value - the agent cannot infer it
  )],
  recipient: Annotated[str, p.ToolParameterOptions(
    source="customer",
  )]
) -> ToolResult:
    # ...
```

The **ToolParameterOptions** consists of several optional arguments, each refining the agent’s understanding and application of the parameter:

- `hidden` If set to `True`, this parameter will not be exposed to message composition. This means the agent won't notify the customer if it’s missing. It's commonly used for internal parameters like opaque product IDs or any other information that should remain behind the scenes.

- `precedence` When a tool has multiple required parameters, the tool insights communicated to the customer can be overwhelming (e.g., asking for 5 different items in a single message). Precedence lets you create groups (which share the same value) such that the customer would only learn about a few (the ones sharing a precedence value) at a time—in the order you choose.

- `source` Defines the source of the argument. Should the agent request the value directly from the customer ("customer"), or should it be inferred from the surrounding context ("context")?
If not specified, the default is "any", meaning the agent can retrieve it from anywhere.

- `description` This helps the agent interpret the parameter correctly when extracting its argument from the context. Fill this if the parameter name is ambigious or unclear.

- `significance` A customer-facing description of why this parameter is required. This helps customers understand and relate to what information they need to provide and why.

- `examples` A list of sample values illustrating how the argument should be extracted. This is useful for enforcing formats (e.g., a date format like "YYYY-MM-DD").

- `adapter` A function that converts the inferred value into the correct type before passing it to the tool. If provided, the agent will run the extracted argument through this function to ensure it matches the expected format. Use when the parameter type is a custom type in your codebase.

- `choice_provider` A function that provides valid choices for the parameter's argument. Use this to constrain the agent to dynamically choose a value from a specific set returned by this function.


## Parameter Value Constraints

In cases where you need a tool's argument to fall into a specific set of choices, Parlant can help you ensure that the tool-call is parameterized according to those choices. There are three ways to go about it:

1. Use enums when you are able to provide hard-coded choices
1. Use `choice_provider` when the choices are dynamic (e.g., customer-specific)
1. Use a Pydantic model when the parameter follows a more complex structure

#### Enum Parameters
Specify a fixed set of choices that are known ahead of time, using an `Enum` class.

```python
import enum

class ProductCategory(enum.Enum):
  LAPTOPS = "laptops"
  PERIPHERALS = "peripherals"
  MONITORS = "monitors"

@p.tool
async def get_products(
  context: p.ToolContext,
  category: ProductCategory,
) -> p.ToolResult:
  # your code here
  return p.ToolResult(returned_data)
```

#### Choice Provider
Dynamically offer a set of choices based on the current execution context.

```python
async def get_last_order_ids(context: p.ToolContext) -> list[str]:
  return await load_last_order_ids_from_db(customer_id=context.customer_id)

@p.tool
async def load_order(
  context: p.ToolContext,
  order_id: Annotated[Optional[str], p.ToolParameterOptions(
    choice_provider=get_last_order_ids,
  )],
) -> p.ToolResult:
  # your code here
  return p.ToolResult({...})
```

#### Pydantic Model
Use a Pydantic model to define a complex structure for the parameter, which can include validation and constraints.
```python
from pydantic import BaseModel

class ProductSearchQuery(BaseModel):
  category: str
  price_range: tuple[float, float]

@p.tool
async def search_products(
  context: p.ToolContext,
  query: ProductSearchQuery,
) -> p.ToolResult:
  # your code here

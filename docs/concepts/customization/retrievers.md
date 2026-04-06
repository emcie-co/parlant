# Retrievers

For pragmatic reasons, Parlant distinguishes between two modes of data access; namely, tools and **retrievers**.

When developing customer-facing agents, there are practically two different use cases for fetching data:
1. **Tools**: Fetching data from specific services in response to specific events, such as user requests.
2. **Retrievers**: Fetching contextual information to ground, orientate, and align the agent's knowledge with respect to the current state of the conversation. This is traditionally referred to as RAG (Retrieval-Augmented Generation).

A rule of thumb is to use **retrievers** for data that you would typically expect the agent "to know"—compared to tools, which are used for data that the agent needs to "load" or "do something with".

**Use cases for retrievers include:**
- Fetching answers to common questions
- Fetching relevant documents or information based on the current conversation context
- Fetching user-specific data to personalize the agent's responses (see also [Variables](https://parlant.io/docs/concepts/customization/variables))

> **Tip: The Response Latency Trade-Off**
> 
> Because retrievers are only used to ground the agent's knowledge within the current conversation's context, they can typically be executed in parallel with the agent's other tasks (such as guideline matching, tool calling, etc.).
> 
> Hence, using retrievers allows you to ground your agent's response without the added latency of guideline matching or tool calls.


## Creating a Retriever
A retriever is a function that takes a `p.RetrieverContext` and returns a `p.RetrieverResult`. The `p.RetrieverContext` contains the current conversation context, and the `p.RetrieverResult` contains the data that the retriever has fetched.

```python
async def my_retriever(context: p.RetrieverContext) -> p.RetrieverResult:
  ...
```

#### Simple RAG Example
Here's a simple example of a retriever that fetches documents from a DB based on the customer's last message:

```python
async def answer_retriever(context: p.RetrieverContext) -> p.RetrieverResult:
    # Get the last message from the conversation
    if last_message := context.interaction.last_customer_message:
        # Use an embedder to convert the message into a vector
        message_vector = my_embedder.embed(last_message.content)
        # Fetch documents from the database based on the message vector
        documents = await fetch_documents_from_db(message_vector)

        return p.RetrieverResult(documents)

    return p.RetrieverResult(None)
```

Alternatively, you could use an LLM to generate a query based on the entire interaction history:

```python
async def answer_retriever(context: p.RetrieverContext) -> p.RetrieverResult:
    if context.interaction.messages:
        # Join all messages in the conversation to create a neat context
        conversation_text = "\n".join(str(msg) for msg in context.interaction.messages)

        # Use an LLM to extract a user query from the conversation
        if query := await my_llm.extract_user_query_from_conversation(conversation_text):
          # Use an embedder to convert the query into a vector
          message_vector = my_embedder.embed(query)
          # Fetch documents from the database based on the query vector
          documents = await fetch_documents_from_db(message_vector)

          return p.RetrieverResult(documents)

    return p.RetrieverResult(None)
```

#### Attaching a Retriever

To actually get an agent to use your retriever, you need to attach it in the following manner:

```python
await agent.attach_retriever(my_retriever)
```

You can also specify the retriever's ID, which is useful for debugging and logging purposes:

```python
await agent.attach_retriever(my_retriever, id="my_retriever")
```


## Retriever Result Lifespan

The lifespan of retriever results is limited to the current response; in other words, it does not persist throughout the conversation. This also helps you keep the conversation context clean and focused, while also reducing average input tokens, throughout the conversation.

## Retriever Context

Using the retriever context, you can access a number of useful properties that can help you build more sophisticated retrievers:
- `server`: The server that is currently processing the retriever request, which can be useful for accessing server-specific resources or configurations.
- `container`: The dependency-injection container that is currently being used, which allows you to access services and resources registered in the container.
- `logger`: The logger that is currently being used, which can be useful for logging debug information or errors during the retriever's execution.
- `trace_id`: A unique identifier for the agent's current response, which can be used for tracking and debugging purposes.
- `interaction`: The current interaction, which contains the conversation history and other relevant information.
- `agent`: The agent that is currently processing the interaction.
- `customer`: The customer that is currently interacting with the agent.
- `variables`: The variables that are currently set for the interaction.

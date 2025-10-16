# Custom NLP Models

Once you've understood the basic of setting up [engine extensions](https://parlant.io/docs/advanced/engine-extensions), you can integrate other NLP models into Parlant.

> **A Note on Custom Models**
>
> Parlant was optimized to work with the built-in LLMs, so using other models may require additional configuration and testing.
>
> In particular, please note that Parlant uses some complex output JSON schemas in its operation. This means that you either need a powerful model that can handle complex outputs, or, alternatively, that you use a smaller model (SLM) that has been fine-tuned on Parlant data specifically, using a larger model as a teacher.
>
> Using smaller models is actually a great way to reduce costs, latency—and sometimes even accuracy—in production environments.

## Understanding `NLPService`
Whether you want to use a different model from a supported built-in provider, or an entirely different provider, you can do so by creating a custom `NLPService` implementation.

An `NLPService` has 3 key components:
1. **Schematic Generators**: These are used to generate structured content based on prompts.
2. **Embedders**: These are used to create vector representations of text for semantic retrieval.
3. **Moderation Service**: This is used to filter out harmful or inappropriate user input in conversations.

> **Reference Example**
>
> You can take a look at the official [`OpenAIService`](https://github.com/emcie-co/parlant/blob/main/src/parlant/adapters/nlp/openai_service.py) for a production-ready reference implementation of an `NLPService`.

### Schematic Generation
Throughout the Parlant engine, you'll find references to `SchematicGenerator[T]` objects. These are objects that generate [Pydantic](https://docs.pydantic.dev/latest/) models using instructions in a provided prompt. Behind the scenes, they always use LLMs to generate JSON schemas that in turn are converted to Pydantic models.

All LLM requests in Parlant are actually made using these schematic generators, which means that, whatever model you use, it must be able to generate valid JSON schemas consistently. This is the only requirement for a model in Parlant.

Let's now look at a few important interfaces that you need to implement in your custom NLP service.

#### Estimating Tokenizers
The `EstimatingTokenizer` interface is used to estimate the number of tokens in a prompt. This is important for managing costs and rate limits when using LLM APIs. It's also used in embedding models, where Parlant needs to chunk the input text into smaller parts to fit within the model's context window.

The reason it's called "estimating" is because not all model APIs provide exact token counts.

```python
class EstimatingTokenizer(ABC):
    """An interface for estimating the token count of a prompt."""

    @abstractmethod
    async def estimate_token_count(self, prompt: str) -> int:
        """Estimate the number of tokens in the given prompt."""
        ...
```

For example, with `OpenAI`, you can implement this to use the `tiktoken` library to get accurate token counts for GPT models, or estimated token counts for other popular models.

#### Schematic Generators
Now let's look at the `SchematicGenerator[T]` interface itself, which is used to generate structured content based on a prompt.

Each generation result from a `SchematicGenerator[T]` contains not just the generated object, but also additional metadata about the generation process. Here's what it looks like:
```python
@dataclass(frozen=True)
class SchematicGenerationResult(Generic[T]):
    content: T  # The generated schematic content (a Pydantic model instance)
    info: GenerationInfo  # Metadata about the generation process


@dataclass(frozen=True)
class GenerationInfo:
    schema_name: str  # The name of the Pydantic schema used for the generated content
    model: str  # The name of the model used for generation
    duration: float  # Time taken for the generation in seconds
    usage: UsageInfo  # Token usage information


@dataclass(frozen=True)
class UsageInfo:
    input_tokens: int
    output_tokens: int
    extra: Optional[Mapping[str, int]] = None  # May contain metrics like cached input tokens
```

Now let's look at the `SchematicGenerator[T]` interface itself, which you'd need to implement for your custom model:

```python
class SchematicGenerator(ABC, Generic[T]):
    """An interface for generating structured content based on a prompt."""

    @abstractmethod
    async def generate(
        self,
        # The prompt (or PromptBuilder) containing instructions for the generation.
        prompt: str | PromptBuilder,
        # Hints are a good way to provide additional context or parameters for the generation,
        # such as temperature, top P, logit bias, and things of that nature.
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[T]:
        """Generate content based on the provided prompt and hints."""
        # Implement this method to generate content using your own model.
        ...

    @property
    @abstractmethod
    def id(self) -> str:
        """Return a unique identifier for the generator."""
        # Normally, this would be the model name or ID used in the LLM API.
        ...

    @property
    @abstractmethod
    def max_tokens(self) -> int:
        """Return the maximum number of tokens in the underlying model's context window."""
        # Return the maximum number of tokens that can be processed by your model.
        ...

    @property
    @abstractmethod
    def tokenizer(self) -> EstimatingTokenizer:
        """Return a tokenizer that approximates that of the underlying model."""
        # This tokenizer should be able to estimate token counts for prompts for this model.
        ...

    @cached_property
    def schema(self) -> type[T]:
        """Return the schema type for the generated content.

        This is useful for derived classes, allowing them to access the concrete
        schema type for the current instance without needing to know the type parameter.
        """
        # You don't need to implement this method - it's an inherited convenience method.
        orig_class = getattr(self, "__orig_class__")
        generic_args = get_args(orig_class)
        return cast(type[T], generic_args[0])
```

> **Reference Example**
>
> You can take a look at the official [`OpenAIService`](https://github.com/emcie-co/parlant/blob/main/src/parlant/adapters/nlp/openai_service.py) for a production-ready reference implementation of an `SchematicGenerator[T]`.

### Embedding
In addition to generating structured content, Parlant also uses embedders to create vector representations of text. These are used for semantic retrieval where applicable throughout the response lifecycle.

#### Embedding Results
Every embedding operation returns an `EmbeddingResult`, which contains the vectors generated by the embedder:

```python
@dataclass(frozen=True)
class EmbeddingResult:
    vectors: Sequence[Sequence[float]]
```

#### Embedders
Now let's look at the `Embedder` interface and how to implement it:

```python
class Embedder(ABC):
    @abstractmethod
    async def embed(
        self,
        texts: list[str],
        hints: Mapping[str, Any] = {},
    ) -> EmbeddingResult:
        # Generate embeddings for the given texts.
        ...

    @property
    @abstractmethod
    def id(self) -> str:
        # Return a unique identifier for the embedder - usually the model name or ID.
        ...

    @property
    @abstractmethod
    def max_tokens(self) -> int:
        # Return the maximum number of tokens in the model's context window.
        ...

    @property
    @abstractmethod
    def tokenizer(self) -> EstimatingTokenizer:
        # Return a tokenizer that approximates the model's token count for prompts.
        ...

    @property
    @abstractmethod
    def dimensions(self) -> int:
        # Return the dimensionality of the embedding space.
        ...
```

> **Reference Example**
>
> You can take a look at the official [`OpenAIService`](https://github.com/emcie-co/parlant/blob/main/src/parlant/adapters/nlp/openai_service.py) for a production-ready reference implementation of an `Embedder`.

### Moderation Services

Parlant includes a comprehensive content moderation system to filter harmful or inappropriate user input. The moderation service is the third key component of an `NLPService`, alongside schematic generators and embedders.

#### Understanding Moderation in Parlant

Parlant's moderation system provides content filtering capabilities that can detect and flag various types of harmful content before it reaches your AI agents. The engine can integrate with all stsandard moderation providers and can be configured with different levels of strictness.

#### Moderation Interface

All moderation services implement the `ModerationService` abstract base class:

```python
@dataclass(frozen=True)
class CustomerModerationContext:
    """Context for moderation check"""
    session: Session    # Session context for the message being checked
    message: str    # The content of the message to check

@dataclass(frozen=True)
class ModerationCheck:
    """Result of a moderation check."""
    flagged: bool  # Whether the content was flagged as inappropriate
    tags: list[str]  # Specific categories that were flagged

class ModerationService(ABC):
    """Abstract base class for content moderation services."""

    @abstractmethod
    async def moderate_customer(self, context: CustomerModerationContext) -> ModerationCheck:
        """Check content for policy violations and return moderation result."""
        ...
```

#### Moderation Tags

Parlant uses standardized moderation tags that map to common content policy categories:

```python
ModerationTag: TypeAlias = Literal[
    "jailbreak",      # Prompt injection attempts
    "harassment",     # Harassment or bullying content
    "hate",          # Hate speech or discrimination
    "illicit",       # Illegal activities or substances
    "self-harm",     # Self-harm or suicide content
    "sexual",        # Sexual or adult content
    "violence",      # Violence or graphic content
]
```

#### Implementing Custom Moderation Services

Here's how to create your own moderation service:

```python
import httpx
import parlant.sdk as p

class MyModerationService(p.ModerationService):
    def __init__(self, api_key: str, logger: p.Logger):
        self._api_key = api_key
        self._logger = logger
        self._client = httpx.AsyncClient()

    async def moderate_customer(self, context: p.CustomerModerationContext) -> p.ModerationCheck:
        """Implement your moderation logic here."""
        try:
            # Example: Call your moderation API
            response = await self._client.post(
                "https://api.your-moderation-service.com/moderate",
                json={"text": context.message},
                headers={"Authorization": f"Bearer {self._api_key}"}
            )
            response.raise_for_status()

            result = response.json()

            # Map your service's response to Parlant's format
            flagged = result.get("flagged", False)
            categories = result.get("categories", [])

            # Convert your categories to Parlant's standardized tags
            tags = []
            category_mapping = {
                "toxic": "harassment",
                "hate_speech": "hate",
                "violence": "violence",
                "sexual_content": "sexual",
                "self_harm": "self-harm",
                "illegal": "illicit",
                "prompt_injection": "jailbreak",
            }

            for category in categories:
                if category in category_mapping:
                    tags.append(category_mapping[category])

            return p.ModerationCheck(
                flagged=flagged,
                tags=tags,
            )

        except Exception as e:
            self._logger.error(f"Moderation check failed: {e}")
            # Fail closed: return unflagged to allow content through
            # Or fail open: return flagged to block content
            return p.ModerationCheck(flagged=False, tags=[])
```

## Customizing Prompts
When you implement your own `SchematicGenerator[T]`, you can also customize the prompts it actually uses.

This is achieved via the `PromptBuilder` class. It's the same class used throughout the Parlant engine to build prompts for LLMs using consistent rules and formats, and it allows you to access and modify prompt templates.

One of the cool things you can do with it is to edit specific prompt sections right before you build the final prompt.

Let's look at an example of how we'd override the draft creation prompt of the `CannedResponseGenerator`.

```python
class MySchematicGenerator(p.SchematicGenerator[p.T]):
    async def generate(
        self,
        prompt: str | p.PromptBuilder,
        hints: Mapping[str, Any] = {},
    ) -> p.SchematicGenerationResult[T]:
        def edit_draft_instructions(section: p.PromptSection) -> p.PromptSection:
            # You can inspect the section's dynamically-passed properties
            # to see what you can make use of in your modified template.
            section.props

            section.template = f"""
            Write your custom instructions here ...
            Pass in dynamic props where needed: {section.props}
            """

            return section

        prompt.edit_section(
            name="canned-response-generator-draft-general-instructions",
            editor_func=edit_draft_instructions,
        )

        # Call the parent class's generate method with the modified prompt
        return await super().generate(prompt, hints)
```

You can modify any section used anywhere within Parlant. You can find these sections by looking at references to `PromptBuilder.add_section()` in the Parlant codebase.

## Implementing an `NLPService`
Now that you understand the key interfaces, you can implement your own `NLPService`. This is the easy part.

Here's what that would look like:

```python
class MyNLPService(p.NLPService):
    def __init__(self, logger: p.Logger):
        self.logger = logger

    async def get_schematic_generator(self, t: type[p.T]) -> p.SchematicGenerator[p.T]:
        # Return your custom schematic generator for the given type.
        return MySchematicGenerator[p.T](
            logger=self.logger,  # Assuming you use a logger
        )

    async def get_embedder(self) -> p.Embedder:
        return MyEmbedder(
            logger=self.logger,  # Assuming you use a logger
        )

    async def get_moderation_service(self) -> p.ModerationService:
        # Return your custom moderation service implementation.
        # If you don't need moderation, return NoModeration().
        return MyModerationService(logger=self.logger)
```

## Injecting a Custom `NLPService`
Once you've implemented your custom `NLPService`, you can easily register it with your Parlant server.

You also get a reference to the dependency-injection container, from which you can access the system's logger and other services, as needed.

```python
def load_custom_nlp_service(container: p.Container) -> p.NLPService:
    return MyNLPService(
        logger=container[p.Logger]
    )
```

Then, when you start your Parlant server, pass your loader function to the `nlp_service` parameter:

```python
async with p.Server(
    nlp_service=load_custom_nlp_service,
) as server:
    # Your code here
```

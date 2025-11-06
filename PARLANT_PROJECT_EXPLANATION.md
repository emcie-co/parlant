# Parlant Project: Comprehensive Explanation

**Generated:** 2025-11-06
**Framework:** Parlant - Enterprise AI Agent Framework
**Repository:** https://parlant.io

---

## Table of Contents

1. [Why Parlant Was Created - Pros & Cons vs Other Frameworks](#1-why-parlant-was-created---pros--cons-vs-other-frameworks)
2. [Does It Follow Correct & Proven Methodology?](#2-does-it-follow-correct--proven-methodology)
3. [Core Ideas of the Project](#3-core-ideas-of-the-project)
4. [How the Project is Organized](#4-how-the-project-is-organized)
5. [How to Evaluate Alignment Modeling](#5-how-to-evaluate-alignment-modeling)
6. [How to Port to Golang](#6-how-to-port-to-golang)

---

## 1. Why Parlant Was Created - Pros & Cons vs Other Frameworks

### The Core Problem Parlant Solves

Parlant addresses a fundamental issue that **n8n, LangChain, Rasa, and similar frameworks don't adequately solve**: **ensuring LLM agents actually follow instructions reliably**.

- **LangChain/LlamaIndex**: Focus on chaining LLM calls and RAG, but rely on prompt engineering (probabilistic compliance)
- **n8n**: Workflow automation, but lacks conversational AI sophistication and LLM reasoning enforcement
- **Rasa**: Rule-based dialogue management, but doesn't integrate modern LLMs deeply or enforce complex guidelines

### Parlant's Unique Value Proposition

**What makes it different:**

1. **Agentic Behavior Modeling (ABM)** - Move from "hoping your LLM follows instructions" to **guaranteeing** it through:
   - Structured guidelines with condition→action patterns
   - **Attentive Reasoning Queries (ARQs)** that force explicit reasoning before decisions
   - Relational constraints between guidelines (priorities, dependencies, entailments)

2. **Enterprise-First Design** - Built for compliance-heavy industries:
   - Financial services (built-in risk management)
   - Healthcare (HIPAA-ready)
   - Legal tech (precise guidance, auditable decisions)

3. **Hallucination Elimination** via Canned Responses:
   ```python
   # Three modes: fluid, composited, strict
   # Strict mode = ONLY approved templates, zero hallucinations
   template="Your balance is {{account_balance}}"
   ```

### Pros vs Other Frameworks

| Advantage | vs LangChain | vs Rasa | vs n8n |
|-----------|--------------|---------|--------|
| **Enforced compliance** | ✅ (ARQs + guidelines) | ~ (rigid rules) | ❌ |
| **LLM flexibility** | ✅ (20+ providers) | ❌ | ~ |
| **Conversational state** | ✅ (journeys) | ✅ | ❌ |
| **Tool reliability** | ✅ (tools only fire when conditions met) | N/A | ~ |
| **Explainability** | ✅ (ARQs trace every decision) | ❌ | ❌ |
| **Zero hallucinations** | ✅ (strict canned mode) | ✅ | N/A |
| **Hexagonal architecture** | ❌ | ❌ | ❌ |

### Cons vs Other Frameworks

1. **Learning curve**: More concepts to master (guidelines, journeys, ARQs, relationships) vs LangChain's simpler chains
2. **Newer ecosystem**: Smaller community than LangChain/LlamaIndex
3. **Opinionated**: Strong architectural opinions (hexagonal pattern) may not suit all teams
4. **Overhead for simple tasks**: If you just need basic RAG, LangChain might be faster to prototype

---

## 2. Does It Follow Correct & Proven Methodology?

**Yes, Parlant follows multiple proven methodologies:**

### A. Hexagonal Architecture (Ports & Adapters) ✅

- **core/**: Pure domain logic, zero external dependencies
- **adapters/**: Implementations (OpenAI, MongoDB, Anthropic, etc.)
- **api/**: REST layer using FastAPI

**Benefits**: Testability, swappable implementations, clear separation of concerns

### B. Test-Driven Development (TDD) ✅

From `CLAUDE.md`:
```
1. Create failing test first
2. Implement just enough to pass
3. Refactor
```

**Test infrastructure**: `pytest`, `SDKTest` base class, mirrors source structure

### C. Type Safety ✅

- MyPy on **strict mode**
- Every parameter and return value type-annotated
- Example from `src/parlant/core/guidelines.py:56-80`:
  ```python
  @dataclass(frozen=True, kw_only=True)
  class GuidelineContent:
      condition: str
      action: str
      coherence_check: bool
      coherence_check_question: str | None
  ```

### D. Domain-Driven Design Patterns ✅

- Rich domain models (agents, guidelines, sessions, journeys)
- Repository pattern for persistence (`src/parlant/core/persistence/`)
- Service layer for NLP operations

### E. Research-Backed Techniques ✅

The framework implements peer-reviewed research:
- **ARQs**: https://arxiv.org/abs/2503.03669 - Attentive Reasoning Queries
- Supports findings from https://arxiv.org/abs/2503.13657 on multi-agent system failures

### F. Async-First Design ✅

Fully async using Python's `asyncio` for scalability

---

## 3. Core Ideas of the Project

### Big Idea #1: Structured Behavior Modeling > Prompt Engineering

**Instead of this (traditional approach):**
```python
system_prompt = """
You are a helpful assistant. When customer wants to return:
1. Get order number
2. Check eligibility
3. Process return
...
"""
```

**Parlant does this:**
```python
await agent.create_guideline(
    condition="Customer wants to return an item",
    action="Get order number and item name, then help them return it",
    tools=[check_return_eligibility, process_return]
)
```

**Why better?**
- Guideline evaluated **before** message generation
- Tools only fire when condition is met
- Auditable via ARQs (see explainability below)

### Big Idea #2: Attentive Reasoning Queries (ARQs)

Forces LLM to explicitly reason about every decision:

```json
{
  "guideline_id": "xyz",
  "condition": "customer wants to return item",
  "condition_application_rationale": "Customer said 'I want to return my shoes'",
  "condition_applies": true,
  "action_application_rationale": [
    {
      "action_segment": "Get order number",
      "rationale": "I haven't collected the order number yet"
    }
  ],
  "applies_score": 9
}
```

From `docs/advanced/explainability.md:15-45` - every decision is traceable.

### Big Idea #3: Journeys = Adaptive State Machines

Unlike rigid chatbot flows, journeys are **suggestive not prescriptive**:

```python
journey = Journey(states=[
    ChatState(guideline=greet_customer),
    ToolState(tool=check_account_status),
    ConditionalTransition(
        condition="account is premium",
        target=offer_premium_service
    )
])
```

Agent can skip, revisit, or jump states based on context while maintaining flow intent.

### Big Idea #4: Relationship Modeling

Guidelines can have relationships (`src/parlant/core/guidelines.py:218-281`):
- **Priority**: "If customer is upset, hand off to human (ignore other guidelines)"
- **Entailment**: "If offering refund, always check fraud first"
- **Dependency**: "Only offer discount if loyalty program active"
- **Disambiguation**: "If multiple shipping options apply, ask customer to clarify"

### Big Idea #5: Context as First-Class Citizen

Three context mechanisms:

1. **Variables** (`src/parlant/core/context_variables.py`): Customer-specific data with freshness rules
   ```python
   var = await agent.create_variable(
       name="account_balance",
       tool=get_balance,
       freshness_rules="*/30 * * * *"  # Refresh every 30 min
   )
   ```

2. **Glossary** (`src/parlant/core/glossary.py`): Domain terminology
   ```python
   await agent.create_term(
       name="VIP Customer",
       description="Customer with >$10k lifetime value",
       synonyms=["premium customer", "top tier"]
   )
   ```

3. **Tool Results**: Automatically loaded into context for subsequent turns

### Big Idea #6: Zero Hallucinations via Canned Responses

Three composition modes (`docs/concepts/customization/canned-responses.md:30-60`):
- **Fluid**: Prefer canned, fall back to generation
- **Composited**: Use canned to style generated responses
- **Strict**: **Only** output approved templates (100% hallucination-free)

---

## 4. How the Project is Organized

### High-Level Structure

```
src/parlant/
├── core/                       # Domain logic (framework rules)
│   ├── engines/alpha/          # Main processing engine
│   │   ├── engine.py           # Orchestrates entire request→response flow
│   │   ├── guideline_matching/ # ARQ-based guideline evaluation
│   │   ├── tool_calling/       # Tool invocation & parameterization
│   │   ├── message_generator.py # LLM message generation
│   │   └── canned_response_generator.py
│   ├── nlp/                    # NLP abstractions (generation, embedding, moderation)
│   ├── persistence/            # Abstract DB interfaces
│   └── [agents, guidelines, sessions, journeys, tools, etc.]
│
├── adapters/                   # External implementations
│   ├── nlp/                    # 20+ LLM providers (OpenAI, Anthropic, Vertex, etc.)
│   ├── db/                     # MongoDB, JSON file, in-memory
│   ├── vector_db/              # Chroma, in-memory
│   └── [loggers, tracing, meter]
│
└── api/                        # REST API (FastAPI)
    ├── app.py                  # App factory
    └── [agents.py, sessions.py, guidelines.py, ...]
```

### Key Modules Explained

| Module | Purpose | Key Files |
|--------|---------|-----------|
| **Engines** | Request processing pipeline | `core/engines/alpha/engine.py:123+` |
| **Guidelines** | Behavior modeling core | `core/guidelines.py`, `core/app_modules/guidelines.py` |
| **Journeys** | Conversational flows | `core/journeys.py` |
| **Tools** | External action integration | `core/tools.py` |
| **Sessions** | Conversation lifecycle | `core/sessions.py` |
| **NLP** | LLM provider abstraction | `core/nlp/generation.py` (interface), `adapters/nlp/*` (implementations) |
| **Persistence** | Data storage | `core/persistence/*` (interfaces), `adapters/db/*` (implementations) |

### Processing Pipeline (from `core/engines/alpha/engine.py`)

```
1. Session State Loading → EngineContext
2. Preparation Iterations (hooks, variable loading)
3. Guideline Matching (ARQs evaluate all guidelines)
4. Relational Resolution (apply priorities, dependencies, etc.)
5. Journey State Selection (if journeys active)
6. Tool Calling (for matched guidelines)
7. Message Generation (with full context)
8. Canned Response Generation (retrieve & render templates)
9. Message Event Composition (final message + metadata)
10. Emission & Persistence
```

### Core Technologies & Dependencies

**Core Dependencies** (from `pyproject.toml`):

| Category | Key Dependencies |
|----------|-----------------|
| **Framework** | FastAPI 0.120+, Starlette 0.49+ |
| **LLM Integration** | OpenAI, Anthropic, Azure, Google Vertex, Ollama, Together, Mistral, Cerebras, Deepseek, Fireworks, LiteLLM |
| **Data Layer** | MongoDB, Chroma (vector DB), PyMongo |
| **NLP/ML** | Transformers, Torch, tiktoken, tokenizers, nano-vectordb |
| **Observability** | OpenTelemetry (tracing, metrics, logging) |
| **Async** | aiofiles, aiorwlock, asyncio |
| **Tool Integration** | fastmcp (Model Context Protocol), aiopenapi3 |
| **Utilities** | Jinja2 (templating), croniter (scheduling), networkx (graph operations), structlog |

---

## 5. How to Evaluate Alignment Modeling

Parlant has a **built-in evaluation system** (`src/parlant/core/evaluations.py:56-140`):

### A. Evaluation System API

```python
# Create evaluation for a guideline
evaluation = await agent.create_evaluation(
    kind=EvaluationKind.GUIDELINE,
    target_id=guideline.id
)

# System generates test "invoices" (scenarios)
# You can inspect them, run tests, check if behavior changed

# Run evaluation
result = await agent.run_evaluation(evaluation.id)

# Check results
assert result.status == EvaluationStatus.COMPLETED
```

### B. Manual Alignment Testing Strategies

Based on `docs/production/agentic-design.md:80-120`, test:

1. **Guideline Coverage**
   - Do guidelines cover all expected scenarios?
   - Test edge cases: "What if customer asks X?"

2. **Relationship Correctness**
   - Do priorities work as expected?
   - Are entailments firing correctly?

3. **Tool Reliability**
   - Do tools only fire when conditions are met?
   - Are parameters extracted correctly?

4. **Canned Response Accuracy**
   - In strict mode, are all responses approved?
   - Are templates rendering correctly with tool data?

5. **Journey Flow**
   - Can agent navigate states appropriately?
   - Does it handle user deviations gracefully?

### C. Explainability for Alignment Auditing

From `docs/advanced/explainability.md:15-80`:

Every agent response includes ARQ traces:
```json
{
  "message": "Let me help you return that item...",
  "arqs": [
    {
      "guideline_id": "xyz",
      "condition_applies": true,
      "condition_application_rationale": "Customer said 'return my shoes'",
      "action_application_rationale": [...]
    }
  ]
}
```

**Audit questions:**
- Are ARQ rationales sensible?
- Are condition_applies scores accurate?
- Are action segments being followed?

### D. Compliance Verification

For regulated industries:

1. **Input Moderation** (`docs/production/input-moderation.md:20-50`):
   - Test jailbreak attempts
   - Verify censorship works
   - Check escalation to human when needed

2. **Output Control**:
   - In strict canned mode, verify zero hallucinations
   - Audit that no PII/sensitive data leaks

3. **Tool Insights** (`docs/concepts/customization/tools.md:150-180`):
   - Track missing parameters
   - Log problematic results
   - Verify agent handles missing data gracefully

### E. Metrics to Track

From `src/parlant/adapters/meter/opentelemetry.py`:
- Guideline match rates
- Tool invocation success/failure rates
- Moderation flag frequencies
- Journey state transitions
- LLM call latencies

---

## 6. How to Port to Golang

### Step 1: Understand Core Architecture Principles

The key to porting is **preserving the hexagonal architecture**:

```
core/          → Pure Go domain logic (no external deps)
adapters/      → Go implementations (OpenAI SDK, MongoDB driver, etc.)
api/           → Go HTTP server (use Gin, Echo, or net/http)
```

### Step 2: Map Python Concepts to Go

| Python Concept | Go Equivalent | Notes |
|----------------|---------------|-------|
| **FastAPI** | Gin, Echo, or Chi | Fast HTTP framework |
| **Pydantic** | go-validator, struct tags | Validation |
| **asyncio** | goroutines + channels | Go's concurrency model |
| **@dataclass** | structs with tags | `type Guideline struct {...}` |
| **Protocol/ABC** | interfaces | `type SchematicGenerator interface {...}` |
| **Type hints** | Go types | Static typing native |

### Step 3: Port Core Domain Models First

Start with foundational types from `src/parlant/core/`:

```go
// From core/guidelines.py:56-80
type GuidelineContent struct {
    Condition              string  `json:"condition"`
    Action                 string  `json:"action"`
    CoherenceCheck         bool    `json:"coherence_check"`
    CoherenceCheckQuestion *string `json:"coherence_check_question,omitempty"`
}

type Guideline struct {
    ID        string           `json:"id"`
    AgentID   string           `json:"agent_id"`
    Content   GuidelineContent `json:"content"`
    CreatedAt time.Time        `json:"creation_utc"`
    UpdatedAt time.Time        `json:"update_utc"`
}

// From core/sessions.py:40-60
type EventSource string
const (
    EventSourceCustomer EventSource = "customer"
    EventSourceAgent    EventSource = "agent"
    EventSourceContext  EventSource = "context"
)

type Event struct {
    ID        string                 `json:"id"`
    Source    EventSource            `json:"source"`
    Kind      string                 `json:"kind"`
    Data      map[string]interface{} `json:"data"`
    CreatedAt time.Time              `json:"creation_utc"`
}
```

### Step 4: Define Persistence Interfaces

From `src/parlant/core/persistence/document_database.py:30-100`:

```go
// persistence/document_database.go
type DocumentDatabase interface {
    ReadAgent(ctx context.Context, agentID string) (*Agent, error)
    CreateAgent(ctx context.Context, agent *Agent) error
    UpdateAgent(ctx context.Context, agent *Agent) error
    DeleteAgent(ctx context.Context, agentID string) error

    ListGuidelines(ctx context.Context, agentID string) ([]*Guideline, error)
    CreateGuideline(ctx context.Context, guideline *Guideline) error
    UpdateGuideline(ctx context.Context, guideline *Guideline) error
    DeleteGuideline(ctx context.Context, guidelineID string) error

    // ... sessions, journeys, tools, etc.
}

// persistence/vector_database.go
type VectorDatabase interface {
    AddDocuments(ctx context.Context, collection string, docs []Document) error
    Search(ctx context.Context, collection string, query string, limit int) ([]SearchResult, error)
    DeleteCollection(ctx context.Context, collection string) error
}
```

### Step 5: Port NLP Abstractions

From `src/parlant/core/nlp/generation.py:50-120`:

```go
// nlp/generation.go
type SchematicGenerator[T any] interface {
    Generate(ctx context.Context, params GenerationParams) (*T, error)
}

type GenerationParams struct {
    SchemaName string
    Context    string
    MaxTokens  int
    Temperature float64
}

// Example: ARQ generator
type ARQGenerator = SchematicGenerator[AttentiveReasoningQuery]

// Implement for OpenAI
type OpenAIGenerator[T any] struct {
    client *openai.Client
    model  string
}

func (g *OpenAIGenerator[T]) Generate(ctx context.Context, params GenerationParams) (*T, error) {
    // Use OpenAI's structured outputs
    // Map JSON schema from T
    // Call client.CreateChatCompletion with response_format
    // Parse and return *T
}
```

### Step 6: Implement Core Engine

From `src/parlant/core/engines/alpha/engine.py:123+`:

```go
// engines/alpha/engine.go
type AlphaEngine struct {
    guidelineMatcher     *GuidelineMatcher
    toolEventGenerator   *ToolEventGenerator
    messageGenerator     *MessageGenerator
    cannedResponseGen    *CannedResponseGenerator
    relationalResolver   *RelationalGuidelineResolver

    db                   persistence.DocumentDatabase
    vectorDB             persistence.VectorDatabase
    nlpService           nlp.Service
}

func (e *AlphaEngine) React(ctx context.Context, sessionID string, message string) (*Event, error) {
    // 1. Load session context
    engineCtx, err := e.loadEngineContext(ctx, sessionID)
    if err != nil {
        return nil, err
    }

    // 2. Preparation iterations
    if err := e.runPreparationIterations(ctx, engineCtx); err != nil {
        return nil, err
    }

    // 3. Match guidelines (ARQs)
    matches, err := e.guidelineMatcher.Match(ctx, engineCtx)
    if err != nil {
        return nil, err
    }

    // 4. Resolve relationships
    resolved, err := e.relationalResolver.Resolve(ctx, matches)
    if err != nil {
        return nil, err
    }

    // 5. Call tools
    toolEvents, err := e.toolEventGenerator.Generate(ctx, engineCtx, resolved)
    if err != nil {
        return nil, err
    }

    // 6. Generate message
    messageEvent, err := e.messageGenerator.Generate(ctx, engineCtx, resolved, toolEvents)
    if err != nil {
        return nil, err
    }

    // 7. Apply canned responses
    finalEvent, err := e.cannedResponseGen.Apply(ctx, engineCtx, messageEvent)
    if err != nil {
        return nil, err
    }

    // 8. Persist and return
    if err := e.db.AddEventToSession(ctx, sessionID, finalEvent); err != nil {
        return nil, err
    }

    return finalEvent, nil
}
```

### Step 7: Implement Adapters

```go
// adapters/nlp/openai.go
type OpenAIService struct {
    client *openai.Client
    config OpenAIConfig
}

func (s *OpenAIService) GenerateARQ(ctx context.Context, params nlp.GenerationParams) (*ARQ, error) {
    // Implementation using openai-go SDK
}

// adapters/db/mongodb.go
type MongoDBAdapter struct {
    client *mongo.Client
    dbName string
}

func (m *MongoDBAdapter) ReadAgent(ctx context.Context, agentID string) (*core.Agent, error) {
    collection := m.client.Database(m.dbName).Collection("agents")
    var agent core.Agent
    err := collection.FindOne(ctx, bson.M{"id": agentID}).Decode(&agent)
    return &agent, err
}
```

### Step 8: Build HTTP API

```go
// api/server.go
func NewServer(app *core.Application) *http.Server {
    r := gin.Default()

    // Agent routes
    r.POST("/agents", handlers.CreateAgent(app))
    r.GET("/agents/:agent_id", handlers.GetAgent(app))
    r.PATCH("/agents/:agent_id", handlers.UpdateAgent(app))

    // Guideline routes
    r.POST("/agents/:agent_id/guidelines", handlers.CreateGuideline(app))
    r.GET("/agents/:agent_id/guidelines", handlers.ListGuidelines(app))

    // Session routes
    r.POST("/agents/:agent_id/sessions", handlers.CreateSession(app))
    r.POST("/agents/:agent_id/sessions/:session_id/events", handlers.ReactToMessage(app))

    // ... etc

    return &http.Server{
        Addr:    ":8000",
        Handler: r,
    }
}
```

### Step 9: Handle Go-Specific Challenges

| Challenge | Solution |
|-----------|----------|
| **No async/await** | Use goroutines + channels, context.Context for cancellation |
| **Generic types** | Go 1.18+ supports generics for SchematicGenerator[T] |
| **JSON schema validation** | Use go-jsonschema or gojsonschema |
| **Jinja2 templating** | Port to text/template or use pongo2 (Jinja2-like for Go) |
| **Datetime handling** | time.Time, be careful with timezones |
| **Dataclass → struct** | Use struct tags: `json:"field" bson:"field"` |

### Step 10: Testing Strategy

```go
// tests/core/guidelines_test.go
func TestGuidelineMatching(t *testing.T) {
    // Setup
    ctx := context.Background()
    db := adapters.NewTransientDB()
    nlp := adapters.NewMockNLPService()

    engine := alpha.NewEngine(db, nlp)

    // Create test data
    agent := &core.Agent{ID: "test-agent"}
    db.CreateAgent(ctx, agent)

    guideline := &core.Guideline{
        Content: core.GuidelineContent{
            Condition: "customer wants to return item",
            Action: "get order number",
        },
    }
    db.CreateGuideline(ctx, guideline)

    // Test
    session, _ := db.CreateSession(ctx, agent.ID)
    event, err := engine.React(ctx, session.ID, "I want to return my shoes")

    // Assert
    assert.NoError(t, err)
    assert.Contains(t, event.Data["message"], "order number")
}
```

### Step 11: Recommended Go Libraries

| Category | Library | Purpose |
|----------|---------|---------|
| **HTTP** | Gin, Echo, Chi | Fast web frameworks |
| **OpenAI** | sashabaranov/go-openai | OpenAI SDK |
| **Anthropic** | anthropics/anthropic-sdk-go | Claude SDK |
| **MongoDB** | mongo-go-driver | Official MongoDB driver |
| **Vector DB** | chroma-go-client | Chroma client |
| **Validation** | go-playground/validator | Struct validation |
| **Logging** | zerolog, zap | Structured logging |
| **Tracing** | OpenTelemetry Go | Observability |
| **Testing** | testify | Assertions & mocking |
| **Templates** | pongo2 | Jinja2-like templating |

### Step 12: Migration Path

Don't port everything at once. Recommend phased approach:

| Phase | Components | Duration |
|-------|------------|----------|
| **Phase 1** | Core models + persistence interfaces | 2-3 weeks |
| **Phase 2** | NLP abstractions + OpenAI adapter | 2 weeks |
| **Phase 3** | Engine pipeline (guideline matching, tool calling) | 3-4 weeks |
| **Phase 4** | Message generation + canned responses | 2 weeks |
| **Phase 5** | API layer + HTTP server | 2 weeks |
| **Phase 6** | Additional adapters (Anthropic, MongoDB, etc.) | 2-3 weeks |
| **Phase 7** | Testing infrastructure | Ongoing |

**Total estimate**: **3-4 months** for feature parity with 2-3 engineers

---

## Summary

**Parlant is a production-grade AI agent framework** that solves the fundamental problem of **ensuring LLM compliance** through:

1. **Structural enforcement** (guidelines, ARQs, relationships)
2. **Hallucination elimination** (strict canned responses)
3. **Enterprise-ready architecture** (hexagonal pattern, observability)
4. **Research-backed techniques** (ARQs from academic papers)

It follows **proven methodologies** (hexagonal architecture, TDD, strong typing, DDD) and is well-organized for porting to other languages. The Go port would preserve the core abstractions while leveraging Go's strengths (goroutines, static typing, performance).

For alignment evaluation, use the built-in evaluation system + manual auditing via ARQ traces + compliance testing for regulated use cases.

---

## Key References

- **Parlant Website**: https://parlant.io
- **ARQs Research Paper**: https://arxiv.org/abs/2503.03669
- **Multi-Agent Failures**: https://arxiv.org/abs/2503.13657
- **Documentation**: `/docs/` directory in repository
- **Core Concepts**: `/docs/concepts/customization/`
- **Production Guide**: `/docs/production/`

---

**End of Document**

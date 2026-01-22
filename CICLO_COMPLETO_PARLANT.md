# 🔄 CICLO COMPLETO PARLANT - Guida di Riferimento

**Data creazione**: 2026-01-08
**Versione repo**: Latest
**Scopo**: Documentazione completa del flusso end-to-end di una query in Parlant

---

## 📑 Indice

1. [Panoramica Architettura](#panoramica-architettura)
2. [Componenti Principali](#componenti-principali)
3. [Ciclo Completo Query](#ciclo-completo-query)
4. [Agenti: Definizione e Funzionamento](#agenti-definizione-e-funzionamento)
5. [Sistema di Hooks](#sistema-di-hooks)
6. [Guideline Matching](#guideline-matching)
7. [Tool Calling Custom](#tool-calling-custom)
8. [Message Generation](#message-generation)
9. [Prompt Building](#prompt-building)
10. [Provider Integration](#provider-integration)
11. [Riferimenti Rapidi](#riferimenti-rapidi)

---

## 📐 Panoramica Architettura

Parlant segue **Hexagonal Architecture (Ports and Adapters)**:

```
┌─────────────────────────────────────────────────────────────┐
│                       API Layer (FastAPI)                    │
│                  src/parlant/api/                            │
│  - sessions.py: HTTP endpoints                               │
│  - agents.py: Agent CRUD                                     │
│  - guidelines.py: Guideline management                       │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│                  Application Layer (Modules)                 │
│              src/parlant/core/app_modules/                   │
│  - SessionModule: Orchestrazione sessioni                    │
│  - AgentModule: Gestione agenti                              │
│  - GuidelineModule: Gestione guidelines                      │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│                    Core Domain Logic                         │
│                  src/parlant/core/                           │
│  - engines/alpha/engine.py: AlphaEngine (processing)         │
│  - agents.py: Agent entities                                 │
│  - sessions.py: Session entities                             │
│  - guidelines.py: Guideline entities                         │
│  - tools.py: Tool abstractions                               │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│                  Adapters (Implementations)                  │
│                  src/parlant/adapters/                       │
│  - nlp/: Provider implementations (Anthropic, OpenAI, ecc.)  │
│  - db/: Database adapters (MongoDB, JSON, Snowflake)         │
│  - vector_db/: Vector stores (Chroma, Qdrant)               │
└─────────────────────────────────────────────────────────────┘
```

**Riferimenti**:
- `src/parlant/CLAUDE.md`: Linee guida architetturali

---

## 🧩 Componenti Principali

### Agent

**File**: `src/parlant/core/agents.py:62-71`

```python
@dataclass(frozen=True)
class Agent:
    id: AgentId
    name: str
    description: Optional[str]
    creation_utc: datetime
    max_engine_iterations: int  # Default: 3
    tags: Sequence[TagId]
    composition_mode: CompositionMode
```

**Composition Modes** (`agents.py:48-52`):
- `FLUID`: Totalmente AI-generated
- `CANNED_FLUID`: Template riempiti dall'AI
- `CANNED_COMPOSITED`: AI assembla parti predefinite
- `CANNED_STRICT`: Solo risposte predefinite

**Store**: `src/parlant/core/agents.py:144-464` (AgentDocumentStore)

---

### Session

**File**: `src/parlant/core/sessions.py`

```python
@dataclass(frozen=True)
class Session:
    id: SessionId
    customer_id: CustomerId
    agent_id: AgentId
    mode: SessionMode  # "auto" | "manual"
    status: SessionStatus
    agent_states: Sequence[AgentState]
    # agent_states contiene:
    # - applied_guideline_ids: Guidelines già applicate
    # - journey_paths: Percorsi nei journey attivi
```

**Eventi**: Immutabili, event sourcing pattern
- `EventKind`: MESSAGE, TOOL, STATUS, CUSTOM
- `EventSource`: CUSTOMER, AI_AGENT, HUMAN_AGENT, SYSTEM

---

### Guideline

**File**: `src/parlant/core/guidelines.py`

```python
@dataclass(frozen=True)
class GuidelineContent:
    condition: str  # "When X happens"
    action: Optional[str]  # "Then do Y" (None = observational)

@dataclass(frozen=True)
class Guideline:
    id: GuidelineId
    content: GuidelineContent
    criticality: Criticality  # LOW, MEDIUM, HIGH
    enabled: bool
    tools: Sequence[ToolId]  # Tool associati (opzionali)
```

**Relationships** (`src/parlant/core/relationships.py`):
- `ENTAILMENT`: Guideline A implica B
- `PRIORITY`: A ha priorità su B
- `DEPENDENCY`: A richiede B
- `DISAMBIGUATION`: A disambigua tra B e C
- `OVERLAP`: A e B si sovrappongono

---

### Journey

**File**: `src/parlant/core/journeys.py`

```python
@dataclass(frozen=True)
class Journey:
    id: JourneyId
    title: str
    description: str
    conditions: Sequence[GuidelineId]  # Trigger conditions
    root_id: str  # Root node
    # Journey è un grafo di nodi con transizioni
```

**Nodi**: Possono essere:
- `chat_state`: Conversazione
- `tool_state`: Esecuzione tool

---

### Tool

**File**: `src/parlant/core/tools.py:143-177`

```python
@dataclass(frozen=True)
class ToolResult:
    data: Any  # Visibile all'agent
    metadata: Mapping[str, Any]  # Solo per frontend
    control: ControlOptions  # Controllo sessione (mode, lifespan)
    canned_responses: Sequence[str]
    canned_response_fields: Mapping[str, Any]
```

**Tool Decorator** (SDK):
```python
@p.tool
async def get_weather(context: p.ToolContext, city: str) -> p.ToolResult:
    return p.ToolResult(data={"temp": 20, "condition": "sunny"})
```

---

## 🔄 Ciclo Completo Query

### FASE 1: Arrivo HTTP Request

**File**: `src/parlant/api/sessions.py:1637-1763`

```
POST /sessions/{session_id}/events
Body: {
  "kind": "message",
  "message": "Voglio prenotare un appuntamento",
  "source": "customer"
}
```

**Flow**:
```python
@router.post("/sessions/{session_id}/events")
async def create_event(session_id, params, moderation):
    # 1. Authorization check (sessions.py:1649-1651)
    await authorization_policy.authorize(
        request, Operation.CREATE_CUSTOMER_EVENT
    )

    # 2. Delega a SessionModule
    return await _add_customer_message(session_id, params, moderation)
```

**Riferimento**: `src/parlant/api/sessions.py:1740-1763`

---

### FASE 2: Session Module - Moderazione e Persistenza

**File**: `src/parlant/core/app_modules/sessions.py:254-310`

```python
async def create_customer_message(
    session_id, message, source, trigger_processing, ...
):
    # 1. MODERAZIONE (se attiva) - Line 269-283
    if moderation in [AUTO, PARANOID]:
        moderation_service = await nlp_service.get_moderation_service()
        check = await moderation_service.moderate_customer(context)
        if check.flagged:
            flagged = True
            tags = check.tags  # ["offensive", "jailbreak"]

    # 2. CREA EVENTO nel DB - Line 296-303
    message_data: MessageEventData = {
        "message": message,
        "participant": {"id": customer_id, "display_name": name},
        "flagged": flagged,
        "tags": tags,
    }

    event = await session_store.create_event(
        session_id=session_id,
        kind=EventKind.MESSAGE,
        data=message_data,
    )

    # 3. TRIGGER PROCESSING - Line 303-306
    if trigger_processing:
        await self.dispatch_processing_task(session)

    return event
```

---

### FASE 3: Background Task Dispatch

**File**: `src/parlant/core/app_modules/sessions.py:368-388`

```python
async def dispatch_processing_task(session: Session):
    # Spawn background task (non-blocking per API response)
    await background_task_service.restart(
        self._process_session(session),
        tag=f"process-session({session.id})"
    )

async def _process_session(session: Session):
    # Crea EventEmitter per emettere nuovi eventi
    event_emitter = await event_emitter_factory.create_event_emitter(
        emitting_agent_id=session.agent_id,
        session_id=session.id,
    )

    # ⚠️ CHIAMA L'ENGINE - Line 382-388
    await self._engine.process(
        Context(session_id=session.id, agent_id=session.agent_id),
        event_emitter=event_emitter,
    )
```

**API Response**: Ritorna subito al client, processing continua in background

---

### FASE 4: AlphaEngine Processing

**File**: `src/parlant/core/engines/alpha/engine.py:176-367`

#### 4.1 Entry Point

```python
async def process(context: Context, event_emitter: EventEmitter) -> bool:
    # 1. LOAD CONTEXT COMPLETO - Line 184
    loaded_context = await self._load_context(context, event_emitter)
    # Carica:
    # - Agent (da AgentStore)
    # - Session (da SessionStore)
    # - Customer (da CustomerStore)
    # - Interaction history (eventi precedenti)

    # 2. CHECK SESSION MODE - Line 186-187
    if loaded_context.session.mode == "manual":
        return True  # Human agent in controllo

    # 3. DO PROCESSING - Line 190-192
    await self._do_process(loaded_context)
```

**Riferimenti**:
- Load context: `engine.py:399-449`
- Context structure: `src/parlant/core/engines/alpha/engine_context.py`

#### 4.2 Main Processing Loop

**File**: `src/parlant/core/engines/alpha/engine.py:261-367`

```python
async def _do_process(context: EngineContext):
    # 1. HOOK: on_acknowledging - Line 265
    if not await hooks.call_on_acknowledging(context):
        return

    # 2. EMIT: Acknowledgement event - Line 269
    await emit_acknowledgement_event(context)
    # Status: "acknowledged"

    # 3. HOOK: on_acknowledged - Line 271
    if not await hooks.call_on_acknowledged(context):
        return

    # 4. HOOK: on_preparing - Line 275
    if not await hooks.call_on_preparing(context):
        return

    # 5. INITIALIZE RESPONSE STATE - Line 278
    await initialize_response_state(context)
    # Carica:
    # - Context Variables (from store)
    # - Glossary Terms (via embedding similarity)
    # - Capabilities (via embedding similarity)

    # 6. PREPARATION ITERATIONS LOOP - Line 280-302
    while not context.state.prepared_to_respond:
        # HOOK: preparation_iteration_start - Line 285

        # Get preamble task - Line 283
        preamble_task = await get_preamble_task(context)

        # RUN PREPARATION ITERATION - Line 292
        iteration_result = await run_preparation_iteration(
            context, preamble_task
        )

        if iteration_result.resolution == BAIL:
            return

        # Update session mode (se tools richiedono handoff) - Line 299
        await update_session_mode(context)

        # HOOK: preparation_iteration_end - Line 301

        # Check se pronti (Line 491-503)
        # - No new tool data OR
        # - Reached max_engine_iterations

    # 7. GENERATE MESSAGES - Line 318-329
    if not await hooks.call_on_generating_messages(context):
        return

    await generate_messages(context, latch)

    # 8. EMIT: Ready event - Line 332
    await emit_ready_event(context)

    # 9. UPDATE AGENT STATE - Line 334-343
    await add_agent_state(
        context=context,
        guideline_matches=matched_guidelines
    )
    # Salva in session:
    # - applied_guideline_ids
    # - journey_paths

    # 10. HOOK: on_messages_emitted - Line 345
```

---

### FASE 5: Preparation Iteration

**File**: `src/parlant/core/engines/alpha/engine.py:468-724`

#### 5.1 Initial Iteration

```python
async def _run_initial_preparation_iteration(
    context, preamble_task
) -> _PreparationIterationResult:

    # 1. CAPTURE TOOL PREEXECUTION STATE - Line 554
    tool_preexecution_state = await capture_tool_preexecution_state(context)

    # 2. ⚠️ MATCH GUIDELINES AND JOURNEYS - Line 559-561
    result = await load_matched_guidelines_and_journeys(context)
    # Returns:
    # - matches_guidelines: List[GuidelineMatch]
    # - resolved_guidelines: List[GuidelineMatch] (con relationships)
    # - journeys: List[Journey] (attivati)

    # 3. UPDATE GLOSSARY TERMS - Line 584
    context.state.glossary_terms.update(
        await load_glossary_terms(context)
    )

    # 4. DISTINGUISH TOOL-ENABLED vs ORDINARY - Line 588-598
    context.state.tool_enabled_guideline_matches = \
        await find_tool_enabled_guideline_matches(resolved_guidelines)

    context.state.ordinary_guideline_matches = list(
        set(resolved_guidelines) - set(tool_enabled_matches.keys())
    )

    # 5. ⚠️ CALL TOOLS - Line 602-612
    if tool_calling_result := await call_tools(context, tool_preexecution_state):
        (tool_events, new_tool_events, tool_insights) = tool_calling_result

        context.state.tool_events += new_tool_events
        context.state.tool_insights = tool_insights

    # 6. UPDATE GLOSSARY AGAIN (tool results may introduce new terms) - Line 619

    return _PreparationIterationResult(
        state=IterationState(
            matched_guidelines=...,
            resolved_guidelines=...,
            tool_insights=...,
            executed_tools=...
        ),
        resolution=COMPLETED
    )
```

#### 5.2 Additional Iterations

**File**: `engine.py:636-724`

Simile all'initial, ma:
- Reevalua solo guidelines che dipendono da tool results precedenti
- Usa `find_guidelines_that_need_reevaluation()` per filtrare

---

## 🎯 Guideline Matching

### Overview

**File**: `src/parlant/core/engines/alpha/engine.py:1042-1139`

Il processo di guideline matching avviene in **8 steps**:

```python
async def _load_matched_guidelines_and_journeys(context):
    # STEP 1: Retrieve available journeys - Line 1047-1049
    available_journeys = await entity_queries.finds_journeys_for_context(
        agent_id=context.agent.id
    )

    # STEP 2: Retrieve all guidelines - Line 1052-1059
    all_stored_guidelines = {
        g.id: g
        for g in await entity_queries.find_guidelines_for_context(
            agent_id=context.agent.id,
            journeys=available_journeys,
        )
        if g.enabled
    }

    # STEP 3: Prune low-probability guidelines - Line 1062-1074
    # Usa journey_paths per identificare journeys attivi
    # Filtra solo top_k=1 journey più rilevanti (via embedding)
    relevant_guidelines, high_prob_journeys = \
        await prune_low_prob_guidelines_and_all_graph(
            context, available_journeys, all_stored_guidelines, top_k=1
        )

    # STEP 4: ⚠️ MATCH GUIDELINES con LLM - Line 1077-1082
    matching_result = await guideline_matcher.match_guidelines(
        context=context,
        active_journeys=high_prob_journeys,
        guidelines=relevant_guidelines,
    )
    # Returns: List[GuidelineMatch] con rationale e score

    # STEP 5: Filter activated journeys - Line 1087-1089
    activated_journeys = filter_activated_journeys(
        context, matching_result.matches, available_journeys
    )

    # STEP 6: Additional matching for low-prob journeys - Line 1094-1116
    if second_match := await process_activated_low_probability_journey_guidelines(...):
        matching_result = merge(matching_result, second_match)

    # STEP 7: Build matched guidelines - Line 1119-1124
    matched_guidelines = await build_matched_guidelines(
        context, relevant_guidelines, matching_result.matches, activated_journeys
    )

    # STEP 8: ⚠️ RESOLVE RELATIONSHIPS - Line 1128-1132
    all_relevant_guidelines = await relational_guideline_resolver.resolve(
        usable_guidelines=list(all_stored_guidelines.values()),
        matches=matched_guidelines,
        journeys=activated_journeys,
    )
    # Gestisce: PRIORITY, ENTAILMENT, DEPENDENCY, DISAMBIGUATION

    return GuidelineAndJourneyMatchingResult(
        matching_result=matching_result,
        matches_guidelines=matching_result.matches,
        resolved_guidelines=all_relevant_guidelines,
        journeys=activated_journeys,
    )
```

### GuidelineMatcher

**File**: `src/parlant/core/engines/alpha/guideline_matching/guideline_matcher.py:197-250`

```python
async def match_guidelines(
    context: EngineContext,
    active_journeys: Sequence[Journey],
    guidelines: Sequence[Guideline],
) -> GuidelineMatchingResult:

    # 1. CREATE MATCHING CONTEXT
    matching_context = GuidelineMatchingContext(
        agent=context.agent,
        customer=context.customer,
        session=context.session,
        context_variables=context.state.context_variables,
        interaction_history=context.interaction.events,
        terms=context.state.glossary_terms,
        journeys=active_journeys,
    )

    # 2. GROUP GUIDELINES BY STRATEGY
    # Diverse strategie:
    # - Generic (ordinary guidelines)
    # - Journey node selection
    # - Journey backtrack
    # - Custom strategies
    strategy_groups = await group_guidelines_by_strategy(guidelines)

    # 3. CREATE BATCHES per ogni strategy
    all_batches = []
    for strategy, strategy_guidelines in strategy_groups.items():
        batches = await strategy.create_matching_batches(
            strategy_guidelines,
            matching_context
        )
        all_batches.extend(batches)

    # 4. ⚠️ PROCESS BATCHES in PARALLEL (chiamate LLM!)
    batch_results = await async_utils.safe_gather(*[
        process_guideline_matching_batch_with_retry(batch)
        for batch in all_batches
    ])

    # 5. AGGREGATE RESULTS
    all_matches = list(chain.from_iterable(
        result.matches for result in batch_results
    ))

    return GuidelineMatchingResult(
        total_duration=sum(b.generation_info.duration for b in batch_results),
        batch_count=len(batch_results),
        batches=[b.matches for b in batch_results],
        matches=all_matches,
    )
```

**Riferimenti**:
- Batch strategies: `src/parlant/core/engines/alpha/guideline_matching/generic/`
- Journey matching: `src/parlant/core/engines/alpha/guideline_matching/generic/journey/`

### GuidelineMatch Structure

```python
@dataclass(frozen=True)
class GuidelineMatch:
    guideline: Guideline
    rationale: str  # Perché questa guideline è matched
    score: int  # Relevance score
    metadata: dict  # Journey info, ecc.
```

---

## 🔧 Tool Calling Custom

### Panoramica

Parlant **NON usa native tool calling** dei provider (Anthropic tool use, OpenAI function calling). Usa invece:
- **Prompt testuali custom** con descrizioni tool
- **JSON structured output** con schema Pydantic
- **Validazione manuale** post-inference

**File principale**: `src/parlant/core/engines/alpha/tool_calling/`

### Perché Custom?

**✅ Vantaggi**:
1. **Provider-Agnostic**: Funziona con qualsiasi LLM (Ollama, modelli locali, ecc.)
2. **Rich Reasoning**: LLM spiega perché chiama il tool
3. **Multi-Iteration**: Feedback loop (tool results → reevaluate → call more tools)
4. **Missing Data Handling**: Sistema sofisticato per parametri mancanti
5. **Audit Trail**: Rationale espliciti per compliance

**❌ Svantaggi**:
1. **Latency**: Prompt lunghi, multiple LLM calls
2. **Cost**: Token usage elevato
3. **Complexity**: Codebase grande, prompt engineering
4. **No Streaming**: Deve aspettare response completa

### ToolCaller Flow

**File**: `src/parlant/core/engines/alpha/tool_calling/tool_caller.py:173-249`

```python
async def infer_tool_calls(context: ToolCallContext) -> ToolCallInferenceResult:
    # 1. GROUP TOOLS by guideline matches
    tools: dict[tuple[ToolId, Tool], list[GuidelineMatch]] = defaultdict(list)

    for guideline_match, tool_ids in context.tool_enabled_guideline_matches.items():
        for tool_id in tool_ids:
            service = await service_registry.read_tool_service(tool_id.service_name)
            tool = await service.resolve_tool(tool_id.tool_name, tool_context)
            tools[(tool_id, tool)].append(guideline_match)

    # 2. CREATE BATCHES via Batcher
    batches = await batcher.create_batches(tools=tools, context=context)

    # 3. PROCESS BATCHES IN PARALLEL
    batch_results = await async_utils.safe_gather(*[
        batch.process() for batch in batches
    ])

    # 4. AGGREGATE RESULTS
    return ToolCallInferenceResult(
        total_duration=t_end - t_start,
        batches=[result.tool_calls for result in batch_results],
        insights=ToolInsights(
            evaluations=aggregated_evaluations,
            missing_data=aggregated_missing_data,
            invalid_data=aggregated_invalid_data,
        ),
    )
```

### SingleToolBatch

**File**: `src/parlant/core/engines/alpha/tool_calling/single_tool_batch.py`

```python
async def process(self) -> ToolCallBatchResult:
    tool_id, tool, _ = self._candidate_tool

    # OPTIMIZATION 1: Auto-approve non-consequential tools no params - Line 201-230
    if not tool.consequential and not tool.parameters:
        return ToolCallBatchResult(
            tool_calls=[ToolCall(id=..., tool_id=tool_id, arguments={})],
            insights=ToolInsights(
                evaluations=[(tool_id, ToolCallEvaluation.NEEDS_TO_RUN)]
            )
        )

    # OPTIMIZATION 2: Simplified mode for non-consequential WITH params - Line 232-242
    if not tool.consequential:
        return await infer_calls_for_non_consequential_tool(...)

    # FULL INFERENCE PATH (consequential tools) - Line 245-272
    (generation_info, inference_output, execution_status,
     missing_data, invalid_data) = \
        await infer_calls_for_consequential_tool(...)

    return ToolCallBatchResult(
        generation_info=generation_info,
        tool_calls=inference_output,
        insights=ToolInsights(
            evaluations=execution_status,
            missing_data=missing_data,
            invalid_data=invalid_data,
        ),
    )
```

### Tool Inference Schema

**File**: `single_tool_batch.py:84-125`

```python
class SingleToolBatchArgumentEvaluation(DefaultBaseModel):
    parameter_name: str
    acceptable_source_for_this_argument_according_to_its_tool_definition: str
    evaluate_is_it_provided_by_an_acceptable_source: str
    evaluate_was_it_already_provided_and_should_it_be_provided_again: str
    evaluate_is_it_potentially_problematic_to_guess_what_the_value_is: str
    is_optional: Optional[bool] = False
    valid_invalid_or_missing: ValidationStatus  # VALID | INVALID | MISSING
    value_as_string: Optional[str] = None

class SingleToolBatchToolCallEvaluation(DefaultBaseModel):
    applicability_rationale: str
    is_applicable: bool
    argument_evaluations: Optional[list[SingleToolBatchArgumentEvaluation]]
    same_call_is_already_staged: bool
    relevant_subtleties: str
    # Edge case handling
    are_optional_arguments_missing: Optional[bool]
    are_non_optional_arguments_missing: Optional[bool]

class SingleToolBatchSchema(DefaultBaseModel):
    last_customer_message: Optional[str]
    most_recent_customer_inquiry_or_need: Optional[str]
    name: str
    subtleties_to_be_aware_of: str
    tool_calls_for_candidate_tool: list[SingleToolBatchToolCallEvaluation]
```

**Nota**: Schema estremamente verbose per guidare reasoning del LLM!

### Tool Execution

**File**: `tool_caller.py:251-300`

```python
async def _run_tool(
    context: ToolContext,
    tool_call: ToolCall,
    tool_id: ToolId,
) -> ToolCallResult:
    # 1. RESOLVE SERVICE
    service = await service_registry.read_tool_service(tool_id.service_name)

    # 2. CALL TOOL
    result = await service.call_tool(
        tool_id.tool_name,
        context,
        tool_call.arguments,
    )

    # 3. RETURN RESULT
    return ToolCallResult(
        id=ToolResultId(generate_id()),
        tool_call=tool_call,
        result={
            "data": result.data,
            "metadata": result.metadata,
            "control": result.control,
            "canned_responses": result.canned_responses,
            "canned_response_fields": result.canned_response_fields,
        },
    )
```

### ToolInsights

**File**: `tool_caller.py:77-113`

```python
@dataclass(frozen=True)
class MissingToolData(ProblematicToolData):
    parameter: str
    significance: Optional[str]
    description: Optional[str]
    examples: Optional[Sequence[str]]
    precedence: Optional[int]  # Per priorità in message generation

@dataclass(frozen=True)
class InvalidToolData(ProblematicToolData):
    parameter: str
    invalid_value: str
    # ... altri campi

@dataclass(frozen=True)
class ToolInsights:
    evaluations: Sequence[tuple[ToolId, ToolCallEvaluation]]
    missing_data: Sequence[MissingToolData]
    invalid_data: Sequence[InvalidToolData]
```

**Uso**: Gli insights vengono passati a message generator per informare il cliente sui dati mancanti!

---

## 💬 Message Generation

### Overview

**File**: `src/parlant/core/engines/alpha/message_generator.py`

Message generation è il processo finale che produce la risposta al cliente.

### Flow Principale

**File**: `message_generator.py:166-300`

```python
async def generate_response(
    context: EngineContext,
    latch: Optional[CancellationSuppressionLatch],
) -> Sequence[MessageEventComposition]:

    return await _do_generate_events(
        agent=context.agent,
        customer=context.customer,
        session=context.session,
        context_variables=context.state.context_variables,
        interaction_history=context.interaction.events,
        terms=list(context.state.glossary_terms),
        capabilities=context.state.capabilities,
        ordinary_guideline_matches=context.state.ordinary_guideline_matches,
        tool_enabled_guideline_matches=context.state.tool_enabled_guideline_matches,
        tool_insights=context.state.tool_insights,
        staged_tool_events=context.state.tool_events,
        latch=latch,
    )
```

### Do Generate Events

**File**: `message_generator.py:208-300`

```python
async def _do_generate_events(...) -> Sequence[MessageEventComposition]:

    # 1. SKIP SE NESSUN TRIGGER - Line 227-235
    if (not interaction_history and
        not ordinary_guideline_matches and
        not tool_enabled_guideline_matches):
        return []

    # 2. BUILD PROMPT - Line 237-251
    prompt = build_prompt(
        agent=agent,
        context_variables=context_variables,
        customer=customer,
        session=session,
        interaction_history=interaction_history,
        terms=terms,
        ordinary_guideline_matches=ordinary_guideline_matches,
        tool_enabled_guideline_matches=tool_enabled_guideline_matches,
        capabilities=capabilities,
        staged_tool_events=staged_tool_events,
        tool_insights=tool_insights,
        shots=await self.shots(),  # Few-shot examples
    )

    # 3. EMIT: Typing status - Line 253-259
    await event_emitter.emit_status_event(
        data={"status": "typing", "data": {}}
    )

    # 4. RETRY LOOP con temperature diverse - Line 261-299
    generation_attempt_temperatures = [0.5, 0.7, 0.9]

    for generation_attempt in range(3):
        try:
            generation_info, response_message = \
                await generate_response_message(
                    prompt,
                    temperature=generation_attempt_temperatures[generation_attempt],
                    final_attempt=(generation_attempt + 1 == 3),
                )

            if latch:
                latch.enable()  # Cancellation protection

            # 5. EMIT: Message event - Line 279-282
            if response_message is not None:
                handle = await event_emitter.emit_message_event(
                    trace_id=tracer.trace_id,
                    data=response_message,
                )

                return [MessageEventComposition(
                    {"message_generation": generation_info},
                    [handle.event]
                )]
            else:
                # LLM decided no response needed
                return []

        except Exception as exc:
            last_generation_exception = exc
            continue

    # All attempts failed
    raise MessageCompositionError(last_generation_exception)
```

### MessageSchema

Schema Pydantic per structured output:

```python
class ContextEvaluation(DefaultBaseModel):
    most_recent_customer_inquiries_or_needs: Optional[str]
    parts_of_the_context_with_specific_information: Optional[str]
    topics_i_have_sufficient_information_for: Optional[str]
    what_i_do_not_have_enough_information_for: Optional[str]

class Revision(DefaultBaseModel):
    revision_number: int
    content: str
    factual_information_provided: Optional[list[FactualInformationEvaluation]]
    instructions_followed: Optional[list[str]]
    instructions_broken: Optional[list[str]]
    followed_all_instructions: Optional[bool]
    further_revisions_required: Optional[bool]

class MessageSchema(DefaultBaseModel):
    last_message_of_customer: Optional[str]
    produced_reply: Optional[bool]
    produced_reply_rationale: Optional[str]
    guidelines: Optional[list[str]]
    context_evaluation: Optional[ContextEvaluation]
    revisions: Optional[list[Revision]]
```

---

## 📋 Prompt Building

### PromptBuilder

**File**: `src/parlant/core/engines/alpha/prompt_builder.py:92-250`

```python
class PromptBuilder:
    def __init__(self):
        self.sections: dict[str | BuiltInSection, PromptSection] = {}

    def add_section(
        self,
        name: str | BuiltInSection,
        template: str,
        props: dict[str, Any],
        status: Optional[SectionStatus] = None,
    ) -> PromptBuilder:
        self.sections[name] = PromptSection(template, props, status)
        return self

    def build(self) -> str:
        buffer = StringIO()

        for section_name, section in self.sections.items():
            buffer.write(section.template.format(**section.props))
            buffer.write("\n\n")

        return buffer.getvalue().strip()
```

### Built-in Sections

**File**: `prompt_builder.py:56-71`

```python
class BuiltInSection(str, Enum):
    AGENT_IDENTITY = auto()
    CUSTOMER_IDENTITY = auto()
    INTERACTION_HISTORY = auto()
    CONTEXT_VARIABLES = auto()
    GLOSSARY = auto()
    GUIDELINE_DESCRIPTIONS = auto()
    GUIDELINES = auto()
    STAGED_EVENTS = auto()
    JOURNEYS = auto()
    OBSERVATIONS = auto()
    CAPABILITIES = auto()
```

### Esempio Costruzione Prompt

```python
prompt_builder = PromptBuilder()

# Agent Identity
prompt_builder.add_section(
    BuiltInSection.AGENT_IDENTITY,
    template="You are {agent_name}. {agent_description}",
    props={
        "agent_name": agent.name,
        "agent_description": agent.description
    }
)

# Interaction History
prompt_builder.add_section(
    BuiltInSection.INTERACTION_HISTORY,
    template="# Conversation History\n{history}",
    props={
        "history": json.dumps([
            PromptBuilder.adapt_event(e)
            for e in interaction_history
        ])
    }
)

# Context Variables
prompt_builder.add_section(
    BuiltInSection.CONTEXT_VARIABLES,
    template="# Context Information\n{variables}",
    props={
        "variables": context_variables_to_json(context_variables)
    }
)

# Guidelines
prompt_builder.add_section(
    BuiltInSection.GUIDELINES,
    template="# Guidelines to Follow\n{guidelines}",
    props={
        "guidelines": [
            {
                "id": match.guideline.id,
                "condition": match.guideline.content.condition,
                "action": match.guideline.content.action,
                "criticality": match.guideline.criticality.value,
                "rationale": match.rationale,
            }
            for match in ordinary_guideline_matches
        ]
    }
)

# Glossary Terms
prompt_builder.add_section(
    BuiltInSection.GLOSSARY,
    template="# Important Terms\n{terms}",
    props={
        "terms": [
            {
                "name": term.name,
                "synonyms": term.synonyms,
                "description": term.description
            }
            for term in terms
        ]
    }
)

final_prompt = prompt_builder.build()
```

### Event Adaptation

**File**: `prompt_builder.py:202-250`

```python
@staticmethod
def adapt_event(e: Event | EmittedEvent) -> str:
    """Converte eventi in formato JSON per prompt"""

    if e.kind == EventKind.MESSAGE:
        message_data = cast(MessageEventData, e.data)

        if message_data.get("flagged"):
            data = {
                "participant": message_data["participant"]["display_name"],
                "message": "<N/A>",  # Censored
                "censored": True,
                "reasons": message_data["tags"],
            }
        else:
            data = {
                "participant": message_data["participant"]["display_name"],
                "message": message_data["message"],
            }

    if e.kind == EventKind.TOOL:
        tool_data = cast(ToolEventData, e.data)
        data = {
            "tool_calls": [
                {
                    "tool_id": tc["tool_id"],
                    "arguments": tc["arguments"],
                    "result": tc["result"]["data"],
                }
                for tc in tool_data["tool_calls"]
            ]
        }

    source_map = {
        EventSource.CUSTOMER: "user",
        EventSource.AI_AGENT: "ai_agent",
        EventSource.SYSTEM: "system-provided",
    }

    return json.dumps({
        "event_kind": e.kind.value,
        "event_source": source_map[e.source],
        "data": data,
    })
```

---

## 🌐 Provider Integration

### NLPService Interface

**File**: `src/parlant/core/nlp/service.py:46-57`

```python
class NLPService(ABC):
    @abstractmethod
    async def get_schematic_generator(
        self, t: type[T], hints: SchematicGeneratorHints = {}
    ) -> SchematicGenerator[T]: ...

    @abstractmethod
    async def get_embedder(
        self, hints: EmbedderHints = {}
    ) -> Embedder: ...

    @abstractmethod
    async def get_moderation_service(self) -> ModerationService: ...
```

### SchematicGenerator

**File**: `src/parlant/core/nlp/generation.py:41-77`

```python
class SchematicGenerator(ABC, Generic[T]):
    """Interface for generating structured content"""

    @cached_property
    def schema(self) -> type[T]:
        """Return the Pydantic schema type"""
        ...

    @abstractmethod
    async def generate(
        self,
        prompt: str | PromptBuilder,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[T]:
        """Generate content based on prompt"""
        ...

    @property
    @abstractmethod
    def max_tokens(self) -> int:
        """Return max context window size"""
        ...

    @property
    @abstractmethod
    def tokenizer(self) -> EstimatingTokenizer:
        """Return tokenizer for counting tokens"""
        ...
```

### Anthropic Implementation

**File**: `src/parlant/adapters/nlp/anthropic_service.py:72-200`

```python
class AnthropicAISchematicGenerator(BaseSchematicGenerator[T]):
    def __init__(self, model_name: str, logger, tracer, meter):
        super().__init__(logger, tracer, meter, model_name)
        self._client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self._estimating_tokenizer = AnthropicEstimatingTokenizer(...)

    @override
    async def do_generate(
        self,
        prompt: str | PromptBuilder,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[T]:

        if isinstance(prompt, PromptBuilder):
            prompt = prompt.build()

        # Extract supported hints (temperature, ecc.)
        anthropic_args = {
            k: v for k, v in hints.items()
            if k in self.supported_hints
        }

        # ⚠️ CHIAMATA API ANTHROPIC
        response = await self._client.messages.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model_name,  # "claude-sonnet-4-5"
            max_tokens=4096,
            **anthropic_args,
        )

        raw_content = response.content[0].text

        # Extract JSON
        json_content = normalize_json_output(raw_content)
        json_object = jsonfinder.only_json(json_content)[2]

        # Validate with Pydantic
        model_content = self.schema.model_validate(json_object)

        # Record metrics
        await record_llm_metrics(
            meter, model_name, schema_name,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

        return SchematicGenerationResult(
            content=model_content,
            info=GenerationInfo(
                schema_name=self.schema.__name__,
                model=self.id,
                duration=(t_end - t_start),
                usage=UsageInfo(
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                ),
            ),
        )
```

### Altri Provider

**Directory**: `src/parlant/adapters/nlp/`

Provider disponibili:
- `anthropic_service.py`: Claude (Anthropic)
- `openai_service.py`: GPT-4, GPT-3.5
- `gemini_service.py`: Google Gemini
- `azure_service.py`: Azure OpenAI
- `aws_service.py`: AWS Bedrock
- `vertex_service.py`: Google Vertex AI
- `ollama_service.py`: Ollama (local models)
- `litellm_service.py`: LiteLLM (proxy a 100+ providers)
- `deepseek_service.py`: DeepSeek
- `mistral_service.py`: Mistral
- `cerebras_service.py`: Cerebras
- Provider cinesi: `zhipu_service.py`, `qwen_service.py`, `glm_service.py`

Tutti implementano `NLPService` interface → **provider-agnostic**!

---

## 🪝 Sistema di Hooks

### Overview

Gli hooks permettono di intercettare e modificare il comportamento dell'engine in punti specifici del processing.

**File**: `src/parlant/core/engines/alpha/hooks.py`

### EngineHooks Structure

**File**: `hooks.py:52-105`

```python
@dataclass(frozen=False)
class EngineHooks:
    # Error handling
    on_error: list[EngineHook] = field(default_factory=list)

    # Acknowledgement
    on_acknowledging: list[EngineHook] = field(default_factory=list)
    on_acknowledged: list[EngineHook] = field(default_factory=list)

    # Preamble
    on_generating_preamble: list[EngineHook] = field(default_factory=list)
    on_preamble_generated: list[EngineHook] = field(default_factory=list)
    on_preamble_emitted: list[EngineHook] = field(default_factory=list)

    # Preparation
    on_preparing: list[EngineHook] = field(default_factory=list)
    on_preparation_iteration_start: list[EngineHook] = field(default_factory=list)
    on_preparation_iteration_end: list[EngineHook] = field(default_factory=list)

    # Message generation
    on_generating_messages: list[EngineHook] = field(default_factory=list)
    on_draft_generated: list[EngineHook] = field(default_factory=list)
    on_message_generated: list[EngineHook] = field(default_factory=list)
    on_message_emitted: list[EngineHook] = field(default_factory=list)
    on_messages_emitted: list[EngineHook] = field(default_factory=list)

    # Guideline-specific handlers
    on_guideline_match_handlers: dict[
        GuidelineId,
        list[Callable[[EngineContext, GuidelineMatch], Awaitable[None]]]
    ] = field(default_factory=lambda: defaultdict(list))

    on_guideline_message_handlers: dict[
        GuidelineId,
        list[Callable[[EngineContext, GuidelineMatch], Awaitable[None]]]
    ] = field(default_factory=lambda: defaultdict(list))
```

### Hook Results

**File**: `hooks.py:26-41`

```python
class EngineHookResult(Enum):
    CALL_NEXT = auto()
    """Runs the next hook in the chain"""

    RESOLVE = auto()
    """Returns without running next hooks (continue processing)"""

    BAIL = auto()
    """Interrupts processing completely, drops response"""
```

### Hook Signature

```python
EngineHook: TypeAlias = Callable[
    [EngineContext, Any, Optional[Exception]],
    Awaitable[EngineHookResult]
]
```

### Esempio Hook Custom

```python
async def my_custom_hook(
    context: EngineContext,
    payload: Any,
    exc: Optional[Exception]
) -> EngineHookResult:
    # Accesso a tutto il contesto
    agent = context.agent
    session = context.session
    customer = context.customer
    interaction_history = context.interaction.events
    state = context.state

    # Logica custom
    if some_condition:
        logger.info("Blocking processing due to X")
        return EngineHookResult.BAIL

    # Continue normalmente
    return EngineHookResult.CALL_NEXT

# Registrazione hook
engine_hooks.on_preparing.append(my_custom_hook)
```

### Hook Execution

**File**: `hooks.py:145-161`

```python
async def call_hooks(
    hooks: Sequence[EngineHook],
    context: EngineContext,
    payload: Any,
    exc: Optional[Exception] = None,
) -> bool:
    """Returns True se processing può continuare"""

    for callable in hooks:
        match await callable(context, payload, exc):
            case EngineHookResult.CALL_NEXT:
                continue
            case EngineHookResult.RESOLVE:
                return True
            case EngineHookResult.BAIL:
                return False

    return True
```

### Guideline-Specific Hooks

**File**: `engine.py:870-898`

```python
async def _call_guideline_handlers(
    context: EngineContext,
    handlers: dict[GuidelineId, list[Callable]],
):
    """Call handlers for matched guidelines"""

    all_guideline_matches = list(chain(
        context.state.ordinary_guideline_matches,
        context.state.tool_enabled_guideline_matches,
    ))

    handler_tasks = [
        handler(context, match)
        for match in all_guideline_matches
        if match.guideline.id in handlers
        for handler in handlers[match.guideline.id]
    ]

    if handler_tasks:
        await async_utils.safe_gather(*handler_tasks)
```

**Esempio Usage**:
```python
# Hook quando specifica guideline è matched
async def on_insurance_guideline_matched(
    ctx: EngineContext,
    match: GuidelineMatch
):
    logger.info(f"Insurance guideline matched: {match.rationale}")
    # Custom logic...

engine_hooks.on_guideline_match_handlers[guideline_id].append(
    on_insurance_guideline_matched
)

# Hook dopo che messaggi per guideline sono emessi
async def on_insurance_message_sent(
    ctx: EngineContext,
    match: GuidelineMatch
):
    # Analytics, logging, ecc.
    await analytics.track("insurance_inquiry_handled")

engine_hooks.on_guideline_message_handlers[guideline_id].append(
    on_insurance_message_sent
)
```

---

## 🎯 Agenti: Definizione e Funzionamento

### Costruzione di un Agente

**Via SDK** (`examples/healthcare.py`):

```python
import parlant.sdk as p

async with p.Server() as server:
    # 1. CREATE AGENT
    agent = await server.create_agent(
        name="Healthcare Agent",
        description="Empathetic and calming to patients",
        max_engine_iterations=3,  # Default
        composition_mode=p.CompositionMode.FLUID,
    )

    # 2. ADD GLOSSARY TERMS
    await agent.create_term(
        name="Office Hours",
        description="Monday to Friday, 9 AM to 5 PM",
    )

    await agent.create_term(
        name="Charles Xavier",
        synonyms=["Professor X"],
        description="Neurologist, available Mondays and Tuesdays",
    )

    # 3. CREATE TOOLS
    @p.tool
    async def get_upcoming_slots(context: p.ToolContext) -> p.ToolResult:
        # Simulate fetching from DB
        return p.ToolResult(data=["Monday 10 AM", "Tuesday 2 PM"])

    @p.tool
    async def schedule_appointment(
        context: p.ToolContext,
        datetime: datetime
    ) -> p.ToolResult:
        # Simulate scheduling
        return p.ToolResult(data=f"Scheduled for {datetime}")

    # 4. CREATE GUIDELINES
    await agent.create_guideline(
        condition="Patient asks about insurance",
        action="List providers and suggest calling office",
        tools=[get_insurance_providers],
    )

    await agent.create_guideline(
        condition="Patient says visit is urgent",
        action="Tell them to call office immediately",
        criticality=p.Criticality.HIGH,
    )

    # 5. CREATE JOURNEYS
    journey = await agent.create_journey(
        title="Schedule Appointment",
        description="Helps patient find appointment time",
        conditions=["Patient wants to schedule appointment"],
    )

    # Define journey graph
    t0 = await journey.initial_state.transition_to(
        chat_state="Determine reason for visit"
    )

    t1 = await t0.target.transition_to(
        tool_state=get_upcoming_slots
    )

    t2 = await t1.target.transition_to(
        chat_state="List available times and ask which works"
    )

    t3 = await t2.target.transition_to(
        chat_state="Confirm details before scheduling",
        condition="Patient picks a time",
    )

    t4 = await t3.target.transition_to(
        tool_state=schedule_appointment,
        condition="Patient confirms details",
    )

    await t4.target.transition_to(state=p.END_JOURNEY)

    # 6. CREATE OBSERVATIONS (for disambiguation)
    status_inquiry = await agent.create_observation(
        "Patient asks to follow up, but unclear how"
    )
    await status_inquiry.disambiguate([journey1, journey2])
```

### Agent Lifecycle

```
┌─────────────────────────────────────────────────────────┐
│                    AGENT CREATION                        │
│  - Name, description                                     │
│  - max_engine_iterations (quanto iterare)                │
│  - composition_mode (come generare messaggi)             │
└──────────────────┬──────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────────┐
│                  CONFIGURATION                           │
│  - Add Glossary Terms (semantic search)                  │
│  - Add Capabilities (vector-matched)                     │
│  - Add Context Variables (dynamic data)                  │
│  - Define Tools (@p.tool decorator)                      │
└──────────────────┬──────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────────┐
│                  BEHAVIOR DEFINITION                     │
│  - Create Guidelines (condition → action)                │
│  - Create Journeys (multi-step workflows)                │
│  - Create Observations (disambiguation)                  │
│  - Define Relationships (priority, entailment, ecc.)     │
└──────────────────┬──────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────────┐
│                  RUNTIME (per session)                   │
│  1. Customer message arrives                             │
│  2. Engine loads agent configuration                     │
│  3. Match guidelines based on context                    │
│  4. Navigate journeys if applicable                      │
│  5. Call tools if needed                                 │
│  6. Generate response following guidelines               │
│  7. Update agent_state (applied guidelines, paths)       │
└─────────────────────────────────────────────────────────┘
```

### Agent State Management

**File**: `src/parlant/core/sessions.py`

```python
@dataclass(frozen=True)
class AgentState:
    trace_id: str
    applied_guideline_ids: Sequence[GuidelineId]
    journey_paths: Mapping[JourneyId, Sequence[Optional[str]]]
```

**Aggiornamento** (`engine.py:1804-1859`):
```python
async def _add_agent_state(
    context: EngineContext,
    session: Session,
    guideline_matches: Sequence[GuidelineMatch],
):
    # Get previously applied guidelines
    applied_guideline_ids = (
        list(session.agent_states[-1].applied_guideline_ids)
        if session.agent_states
        else []
    )

    # Analyze response to determine which guidelines were actually used
    result = await guideline_matcher.analyze_response(
        agent=context.agent,
        session=session,
        customer=context.customer,
        interaction_history=context.interaction.events,
        staged_message_events=context.state.message_events,
        guideline_matches=matches_to_analyze,
    )

    # Add newly applied guideline IDs
    new_applied_guideline_ids = [
        a.guideline.id
        for a in result.analyzed_guidelines
        if a.is_previously_applied
    ]

    applied_guideline_ids.extend(new_applied_guideline_ids)

    # Update session with new agent state
    await entity_commands.update_session(
        session_id=session.id,
        params=SessionUpdateParamsModel(
            agent_states=list(session.agent_states) + [
                AgentState(
                    trace_id=tracer.trace_id,
                    applied_guideline_ids=applied_guideline_ids,
                    journey_paths=context.state.journey_paths,
                )
            ]
        ),
    )
```

---

## 📊 Riferimenti Rapidi

### File Chiave per Area

**API Layer**:
- `src/parlant/api/sessions.py`: Endpoints HTTP per sessioni/eventi
- `src/parlant/api/agents.py`: CRUD agenti
- `src/parlant/api/guidelines.py`: Gestione guidelines
- `src/parlant/api/journeys.py`: Gestione journeys

**Application Layer**:
- `src/parlant/core/app_modules/sessions.py`: SessionModule orchestration
- `src/parlant/core/application.py`: Application container (DI)

**Core Engine**:
- `src/parlant/core/engines/alpha/engine.py`: AlphaEngine (processing principale)
- `src/parlant/core/engines/alpha/engine_context.py`: EngineContext structure
- `src/parlant/core/engines/alpha/hooks.py`: Hook system

**Guideline Matching**:
- `src/parlant/core/engines/alpha/guideline_matching/guideline_matcher.py`
- `src/parlant/core/engines/alpha/guideline_matching/generic/`: Generic matching
- `src/parlant/core/engines/alpha/guideline_matching/generic/journey/`: Journey matching
- `src/parlant/core/engines/alpha/relational_guideline_resolver.py`: Relationships

**Tool Calling**:
- `src/parlant/core/engines/alpha/tool_calling/tool_caller.py`: Main orchestrator
- `src/parlant/core/engines/alpha/tool_calling/single_tool_batch.py`: Single tool inference
- `src/parlant/core/engines/alpha/tool_calling/overlapping_tools_batch.py`: Multiple tools
- `src/parlant/core/engines/alpha/tool_calling/default_tool_call_batcher.py`: Batching logic

**Message Generation**:
- `src/parlant/core/engines/alpha/message_generator.py`: Fluid message generation
- `src/parlant/core/engines/alpha/canned_response_generator.py`: Canned responses
- `src/parlant/core/engines/alpha/message_event_composer.py`: Abstract interface
- `src/parlant/core/engines/alpha/prompt_builder.py`: Prompt construction

**NLP Integration**:
- `src/parlant/core/nlp/service.py`: NLPService interface
- `src/parlant/core/nlp/generation.py`: SchematicGenerator interface
- `src/parlant/adapters/nlp/anthropic_service.py`: Anthropic implementation
- `src/parlant/adapters/nlp/openai_service.py`: OpenAI implementation
- `src/parlant/adapters/nlp/`: Altri 20+ provider

**Domain Entities**:
- `src/parlant/core/agents.py`: Agent entity & store
- `src/parlant/core/sessions.py`: Session, Event entities
- `src/parlant/core/guidelines.py`: Guideline entity & store
- `src/parlant/core/journeys.py`: Journey entity & graph
- `src/parlant/core/tools.py`: Tool abstractions
- `src/parlant/core/context_variables.py`: Context variables
- `src/parlant/core/glossary.py`: Glossary terms
- `src/parlant/core/capabilities.py`: Capabilities (vector-matched)

**Persistence**:
- `src/parlant/core/persistence/document_database.py`: Abstract DB interface
- `src/parlant/core/persistence/vector_database.py`: Vector store interface
- `src/parlant/adapters/db/`: DB implementations (MongoDB, JSON, ecc.)
- `src/parlant/adapters/vector_db/`: Vector DB implementations (Chroma, Qdrant)

**SDK**:
- `src/parlant/sdk.py`: Public SDK interface

**Examples**:
- `examples/healthcare.py`: Healthcare agent completo
- `examples/travel_voice_agent.py`: Travel agent

---

## 🔍 Pattern e Best Practices

### Hexagonal Architecture

```
Core Domain (ports) ← Adapters (implementations)
```

**Regole**:
- Core dipende solo da abstractions (ABC, Protocol)
- Adapters implementano le interfaces del core
- Dependency Injection via Lagom container

### Event Sourcing

- Tutti gli eventi sono **immutabili**
- History completa della sessione
- Offset-based per sincronizzazione client

### Store Pattern

Ogni entità ha:
- Abstract `Store` class (interface)
- `DocumentStore` implementation (con versioning)
- Migration support per schema evolution
- Reader-writer locks per concurrency

### Async-First

- Tutto è `async def`
- `async_utils.safe_gather()` per parallel execution
- Timeout management
- Cancellation handling con `CancellationSuppressionLatch`

### Type Safety

- MyPy strict mode
- Full type annotations
- NewType per domain IDs
- TypedDict per schemas
- Pydantic per validation

---

## 🎓 Tips per Studiare la Repo

### 1. Parti dal Flusso

Segui un messaggio end-to-end:
1. `api/sessions.py:1637` → HTTP endpoint
2. `app_modules/sessions.py:254` → SessionModule
3. `engines/alpha/engine.py:176` → AlphaEngine.process()
4. `engines/alpha/engine.py:261` → _do_process()

### 2. Cerca Pattern Ricorrenti

- `Store` pattern per tutte le entities
- `SchematicGenerator[T]` per LLM calls
- `PromptBuilder` per costruzione prompt
- `ToolCallBatch` per tool inference

### 3. Usa gli Examples

- `examples/healthcare.py`: Esempio completo e commentato
- Mostra SDK usage in pratica
- Guidelines, journeys, tools integrati

### 4. Debugging con Logs

Attiva logging verbose:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Molti file hanno trace logging con dettagli prompt/response.

### 5. Tests come Documentazione

`tests/` mirror della struttura `src/`:
- Test names spiegano behavior: `test_that_...`
- Setup mostra come usare i componenti
- Fixtures mostrano dati validi

### 6. Key Questions da Farsi

Quando leggi un modulo:
1. **Qual è il suo ruolo?** (layer, responsabilità)
2. **Quali abstractions usa?** (ABC, Protocol)
3. **Chi lo chiama?** (caller)
4. **Cosa chiama?** (dependencies)
5. **Come è testato?** (test file corrispondente)

---

## 📈 Metriche e Osservability

### Tracing

**File**: `src/parlant/core/tracer.py`

```python
with tracer.span("operation_name", attributes={...}):
    # Code
    tracer.add_event("event_name", attributes={...})
```

**Span names importanti**:
- `process`: AlphaEngine.process()
- `preparation_iteration_{n}`: Iteration n
- `guideline_matcher`: Guideline matching
- `tool_caller`: Tool calling
- `message_generation`: Message generation

### Metrics

**File**: `src/parlant/core/meter.py`

```python
# Duration histograms
hist = meter.create_duration_histogram(
    name="eng.process",
    description="Engine processing duration"
)

async with hist.measure({"session_id": session_id}):
    # Code

# Record value
await hist.record(value_ms, attributes={...})
```

**Key metrics**:
- `eng.process`: Total engine processing
- `eng.utter`: Utterance generation
- `gm.match`: Guideline matching
- `gm.analysis`: Response analysis
- `gen`: LLM generation requests
- `ttfm`: Time to first message

### Logging

**File**: `src/parlant/core/loggers.py`

```python
logger.trace("Detailed info")
logger.debug("Debug info")
logger.info("General info")
logger.warning("Warning")
logger.error("Error")
logger.critical("Critical error")

# Scoped logging
with logger.scope("ScopeName"):
    logger.info("Inside scope")
```

---

## 🚀 Next Steps

### Per Approfondire

**Guideline Matching**:
- `guideline_matching/generic/disambiguation_batch.py`
- `guideline_matching/generic/actionable_batch.py`
- `guideline_matching/generic/journey/journey_step_selection.py`

**Journey Navigation**:
- `journey_guideline_projection.py`
- `journeys.py`: Graph structure

**Relationships**:
- `relational_guideline_resolver.py`
- `relationships.py`: RelationshipKind

**Context Variables**:
- `context_variables.py`: Tool-enabled variables
- Freshness rules (cron expressions)

**Canned Responses**:
- `canned_response_generator.py`
- Selection strategies

**Evaluations**:
- `evaluations.py`: Testing framework
- Behavioral testing

---

## 📚 Risorse Aggiuntive

- **README.md**: Setup e quick start
- **CLAUDE.md**: Coding guidelines per contributors
- **CONTRIBUTING.md**: Come contribuire
- **docs/**: Documentazione aggiuntiva
- **llms.txt**: Context per LLM assistants

---

## 🤝 Conclusione

Questa guida copre il 90% del flusso core di Parlant. Per casi specifici:

1. **Leggi il file specifico** con riferimenti qui
2. **Traccia il flusso** con debugger o logs
3. **Guarda i test** per esempi d'uso
4. **Chiedi nella repo** (GitHub Issues)

Buono studio! 🎓

# OCI Generative AI Adapter - Reverse Engineering & Design Notes

Questo documento riassume cosa ho scoperto nell'architettura NLP di Parlant, quali moduli sono rilevanti, come si collegano, e una proposta di pseudocodice per un adapter OCI che gestisca le due API (Generic vs Cohere). Include anche le open question da chiarire prima di implementare.

## Autenticazione OCI (doc + proposta)
OCI usa un file di configurazione standard e supporta più metodi di autenticazione nel Python SDK. Dobbiamo prevedere **sia** il config file di default **sia** un override via variabili d'ambiente.

**Riferimenti ufficiali**
- Metodi di autenticazione supportati (API key/config file, session token, instance principal, resource principal):  
  https://docs.oracle.com/en-us/iaas/Content/API/Concepts/sdk_authentication_methods.htm
- Config file OCI (formato e path di default `~/.oci/config`):  
  https://docs.oracle.com/iaas/Content/API/Concepts/sdkconfig.htm
- Python SDK `oci.config.from_file` con override via env:  
  https://docs.oracle.com/en-us/iaas/tools/python/latest/api/config.html

**Cosa fare nel nostro adapter**
- **Supportare il config file di default**: `~/.oci/config` con profilo `DEFAULT`.
- **Supportare override via env**:
  - `OCI_CONFIG_FILE` per path custom del file config
  - `OCI_CONFIG_PROFILE` per profilo custom
- **(Opzionale)** Supportare env “per-campo” *a livello adapter* (non garantito dall’SDK):
  - `OCI_USER`, `OCI_TENANCY`, `OCI_FINGERPRINT`, `OCI_KEY_FILE`, `OCI_PASSPHRASE`, `OCI_REGION`

**Ordine di precedenza consigliato**
1) Env per-campo (se tutti i campi richiesti sono presenti)  
2) Env per file/profilo (OCI_CONFIG_FILE/OCI_CONFIG_PROFILE)  
3) Default (`~/.oci/config`, profilo `DEFAULT`)

**Pseudocodice**
```python
def load_oci_config() -> dict:
    # 1) Override completo via env (adapter-level)
    if all(os.environ.get(k) for k in [
        "OCI_USER", "OCI_TENANCY", "OCI_FINGERPRINT",
        "OCI_KEY_FILE", "OCI_REGION"
    ]):
        return {
            "user": os.environ["OCI_USER"],
            "tenancy": os.environ["OCI_TENANCY"],
            "fingerprint": os.environ["OCI_FINGERPRINT"],
            "key_file": os.environ["OCI_KEY_FILE"],
            "pass_phrase": os.environ.get("OCI_PASSPHRASE"),
            "region": os.environ["OCI_REGION"],
        }

    # 2) Config file (con override path/profilo)
    return oci.config.from_file(
        file_location=os.getenv("OCI_CONFIG_FILE", "~/.oci/config"),
        profile_name=os.getenv("OCI_CONFIG_PROFILE", "DEFAULT"),
    )
```

## Obiettivo
- Integrare OCI Generative AI in Parlant con un adapter NLP coerente con gli altri provider.
- Gestire la doppia API OCI:
  - **Generic API format** per modelli non-Cohere (meta.*, google.*, xai.*, openai.*).
  - **Cohere API format** per modelli cohere.*.
- Mantenere l'architettura standard: generator JSON, embedder, moderation.

## Moduli chiave in Parlant (contesto)
**Interfacce core**
- `parlant/src/parlant/core/nlp/service.py`
  - Definisce `NLPService` con i 3 elementi obbligatori: `get_schematic_generator`, `get_embedder`, `get_moderation_service`.
- `parlant/src/parlant/core/nlp/generation.py`
  - `SchematicGenerator` e `BaseSchematicGenerator`: contratto per generazione JSON con schema Pydantic.
- `parlant/src/parlant/core/nlp/embedding.py`
  - `Embedder` e `BaseEmbedder` per embeddings.
- `parlant/src/parlant/core/nlp/moderation.py`
  - `ModerationService`, `NoModeration`, e servizi specifici per provider.

**Hook di runtime / wiring**
- `parlant/src/parlant/bin/server.py`
  - CLI flag provider, `NLPServiceName`, e `NLP_SERVICE_INITIALIZERS`.
  - `require_env_keys` valida env a runtime.
- `parlant/src/parlant/sdk.py`
  - `NLPServices` factory per SDK (uso programmatico).

**Pattern adapter esistenti**
- `parlant/src/parlant/adapters/nlp/openai_service.py`
  - Template di adapter completo (generator, embedder, moderation).
- `parlant/src/parlant/adapters/nlp/gemini_service.py`
  - Modello multi‑size / fallback generator.
- `parlant/src/parlant/adapters/nlp/vertex_service.py`
  - Esempio di routing interno verso API diverse in base al modello (Anthropic vs Google).
- `parlant/src/parlant/adapters/nlp/common.py`
  - `record_llm_metrics` + `normalize_json_output`.

**Uso nel core**
- `parlant/src/parlant/core/app_modules/sessions.py`
  - Usa `get_moderation_service` per check safety.
- `parlant/src/parlant/core/services/tools/service_registry.py`
  - Registry per servizi NLP (non riguarda tool-calling del provider).
- `parlant/src/parlant/core/persistence/vector_database.py`
  - Usa `Embedder` per embeddings.

## Collegamento dei pezzi (flow operativo)
1) **Avvio server**
   - CLI (`parlant/bin/server.py`) sceglie NLP provider e valida env.
2) **Creazione NLPService**
   - `NLP_SERVICE_INITIALIZERS` crea istanza adapter (es. OpenAIService).
3) **Generazione strutturata**
   - Il core chiede a `NLPService.get_schematic_generator()` un generator.
   - Il generator invia prompt e pretende JSON valido per schema Pydantic.
4) **Embeddings**
   - Il core usa `NLPService.get_embedder()` in vector DB e indexing.
5) **Moderation**
   - `NLPService.get_moderation_service()` viene chiamato su sessioni.

Quindi un adapter OCI deve coprire questi tre blocchi, senza introdurre dipendenze di tool-calling esterne al modello (Parlant non usa function calling del provider nelle richieste LLM).

## Strategie (OCI Generic vs Cohere)

- **Strategia Generic**: usa `GenericChatRequest` con `messages` (lista) e `response_format` standard.
- **Strategia Cohere**: usa `CohereChatRequest` con `message` + opzionale `chat_history` e `response_format` Cohere.
- **Routing**: scelta strategia via prefix del modello (`cohere.` vs altri).

## Proposta di design per OCI in Parlant
### 1) Configurazione via env
Suggerita (allineata al pattern degli altri adapter):
- `OCI_COMPARTMENT_ID` (obbligatorio)
- `OCI_CONFIG_FILE` (opzionale, default `~/.oci/config`)
- `OCI_CONFIG_PROFILE` (opzionale, default `DEFAULT`)
- `OCI_MODEL_ID` (default: `meta.llama-3.3-70b-instruct`)
- `OCI_MAX_TOKENS`, `OCI_TEMPERATURE` (opzionali)
- `OCI_MAX_CONTEXT_TOKENS` (opzionale: usato solo come metadata per max_tokens del generator)
- `OCI_EMBEDDING_MODEL_ID` (default: `cohere.embed-multilingual-v3.0`)
- `OCI_EMBEDDING_DIMS` (opzionale, override dimensioni embedding)

### 2) Routing API format
Regola base:
- Se `model_id.startswith("cohere.")` → usare `CohereChatRequest`.
- Altrimenti → usare `GenericChatRequest`.

Il routing è gestito internamente in un unico `OCISchematicGenerator` con un flag `_is_cohere` (non due classi separate).

**Parametri (temperature/top_p/top_k/max_tokens)**
- Pattern Parlant: `hints` → allow‑list → mapping nel request (come OpenAI/Azure/Vertex/Ollama).
- Per OCI usare una lista `supported_hints` e passare solo i campi supportati dal request scelto.
  - Esempio: `supported_hints = ["temperature", "max_tokens", "top_p", "top_k"]`
  - Se un parametro non è supportato dal request OCI, lo si ignora (come fa Gemini).
  - Questa logica vive dentro `OCISchematicGenerator._build_*_request()`.

**Parametri supportati per formato (OCI docs)**
- **GenericChatRequest** supporta: `temperature`, `top_k`, `top_p`, `max_tokens`, `max_completion_tokens`, `presence_penalty`, `logit_bias`, `stop`, `num_generations`, `seed`, `response_format`, `tools`, `tool_choice`, `reasoning_effort`, `service_tier`, `verbosity`, `stream_options`, `web_search_options`, `metadata`, `prediction`. citeturn0search3turn0search2turn0search6
- **CohereChatRequest** supporta: `temperature`, `top_k`, `top_p`, `max_tokens`, `max_input_tokens`, `presence_penalty`, `frequency_penalty`, `stop_sequences`, `seed`, `prompt_truncation`, `response_format`, `tools`, `tool_results`, `safety_mode`, `stream_options`, `preamble_override`, `is_search_queries_only`, `chat_history`, `documents`. citeturn0search4turn0search0turn0search1
- **Differenze principali**:
  - Generic ha `logit_bias`, `num_generations`, `stop` (array di stringhe), `tool_choice`, `reasoning_effort`, `service_tier`, `verbosity`, `web_search_options`. citeturn0search3turn0search2turn0search6
  - Cohere ha `max_input_tokens`, `prompt_truncation`, `frequency_penalty`, `stop_sequences` (stringhe), `safety_mode`, `is_search_queries_only`, `preamble_override`. citeturn0search4turn0search0turn0search1

**Mapping hints Parlant → OCI**
Supportiamo solo gli hints già usati in altri adapter:

| Hint Parlant | GenericChatRequest | CohereChatRequest |
| --- | --- | --- |
| `temperature` | `temperature` | `temperature` |
| `top_p` | `top_p` | `top_p` |
| `top_k` | `top_k` | `top_k` |
| `max_tokens` | `max_tokens` (oppure `max_completion_tokens` se richiesto dal modello) | `max_tokens` |
| `stop` | `stop` (list[str]) | `stop_sequences` (list[str]) |

Note:
- `max_completion_tokens` è disponibile nel Generic API; scegliamo `max_tokens` salvo indicazioni specifiche del modello.
- Hints non presenti in tabella vengono ignorati.

### 3) Tokenizer
OCI non espone un endpoint per contare i token prima della chiamata. Usiamo `tiktoken` per stimare quanti token occupa il prompt prima di inviarlo, così Parlant può verificare che il prompt rientri nel context window del modello. Il codice è simile a `aws_service.py`.

```python
class OCIEstimatingTokenizer(EstimatingTokenizer):
    def __init__(self) -> None:
        self.encoding = tiktoken.encoding_for_model("gpt-4o-2024-08-06")

    @override
    async def estimate_token_count(self, prompt: str) -> int:
        tokens = self.encoding.encode(prompt)
        return int(len(tokens) * 1.15)  # margine di sicurezza
```

### 4) Embeddings
OCI espone **embedding models** (Cohere) e un endpoint dedicato di inference.

**Modelli disponibili**
- La pagina dei pretrained models elenca gli **Embed Models** (Cohere Embed 4, Embed English/Multilingual, Image/Light, ecc.). citeturn1view0
- I modelli hanno dimensionalità diverse; ad esempio **Cohere Embed English Light 3** produce vettori da **384** dimensioni. citeturn1view1

**API/SDK da usare**
- Python SDK: `GenerativeAiInferenceClient.embed_text(embed_text_details)` produce embeddings. citeturn5search4
- Payload principale: `EmbedTextDetails` con campi richiesti `compartment_id`, `inputs`, `serving_mode`. Opzionali: `input_type`, `truncate`, `is_echo`. citeturn3view0
- Vincoli input: `inputs` è una lista di stringhe (o una singola immagine base64 con `input_type=IMAGE`); ogni stringa ha max **512 token**. citeturn3view0
- Limiti operativi: per i modelli text-only, la console limita a **96** input e **512 token** per input (e “truncate” è richiesto se si supera). citeturn1view4turn1view1
- Output: `EmbedTextResult` restituisce `embeddings` (list[list[float]]) e metadata come `model_id`/`model_version`. citeturn4view0
- Nota: la doc del client embed_text menziona array da **1024** numeri per embedding (valido per alcuni modelli). La dimensione effettiva va presa dal modello scelto. citeturn5search4turn1view1

**Impatti per Parlant**
- Implementare un `OCIEmbedder` che chiama `embed_text` con `EmbedTextDetails` e usa `serving_mode` ON_DEMAND + `model_id` scelto.
- Gestire `input_type`/`truncate` come hint opzionali.
- Non assumere una dimensione fissa: ricavare le dimensioni dal primo embedding (cache) e permettere override via env (`OCI_EMBEDDING_DIMS`).
  - **Ispirazione 1 (OpenRouter)**: cache dinamica delle dimensioni al primo response se il modello non è noto.  
    Vedi `src/parlant/adapters/nlp/openrouter_service.py` (classe `OpenRouterEmbedder`, `_cached_dimensions` + `len(vectors[0])`).
  - **Ispirazione 2 (Azure)**: override dimensioni via env per embedder custom.  
    Vedi `src/parlant/adapters/nlp/azure_service.py` (usa `AZURE_EMBEDDING_MODEL_DIMS`).
  - **Come fare in OCI**:
    1) `OCIEmbedder.__init__`: legge `OCI_EMBEDDING_DIMS` (se presente) e salva `_cached_dimensions`.
    2) `do_embed`: dopo `embeddings = ...`, se `_cached_dimensions` è `None`, imposta `_cached_dimensions = len(embeddings[0])`.
    3) `dimensions` property: se env/`_cached_dimensions` è settata, usarla; altrimenti usare come fallback **1024** (dimensione di `cohere.embed-multilingual-v3.0`). citeturn0view0
    4) Loggare la dimensione rilevata (come OpenRouter) per trasparenza.

### 5) Moderation
OCI non ha moderation standard per Parlant → `NoModeration`.

### 6) Rate limiting / retry
Policy proposta (allineata agli altri adapter con `@policy(retry(...))`):
- **Retry** su:
  - `oci.exceptions.ServiceError` con `status` **429** (rate limit) o **5xx** (errori transitori)
  - errori di rete `requests.exceptions.Timeout` / `requests.exceptions.ConnectionError`
- **No retry** su altri `4xx` (config/permessi/input errati)

Messaggio di log simile agli altri adapter (es. “OCI API rate limit exceeded...”) e poi raise.

## Pseudocodice (generator OCI)
```python
class OCIService(NLPService):
    verify_environment():
        check OCI SDK install
        check OCI_COMPARTMENT_ID
        check OCI_CONFIG_FILE exists (if provided)

    get_schematic_generator(schema):
        model_id = env("OCI_MODEL_ID", default)
        return OCISchematicGenerator(schema, model_id, config, compartment)

    get_embedder():
        # OCIEmbedder con model_id da env (default) e dimensioni cache/override

    get_moderation_service():
        return NoModeration()


class OCISchematicGenerator(BaseSchematicGenerator):
    def __init__(self, model_id, ...):
        self._is_cohere = model_id.startswith("cohere.")
        self._json_schema = self.schema.model_json_schema()

    def _build_generic_request(self, prompt):
        # GenericChatRequest(
        #   api_format=GENERIC,
        #   messages=[UserMessage(TextContent(prompt))],
        #   response_format=ResponseFormat(type=JSON_SCHEMA, json_schema=self._json_schema)
        # )
        ...

    def _build_cohere_request(self, prompt):
        # CohereChatRequest(
        #   api_format=COHERE,
        #   message=prompt,
        #   response_format=CohereResponseFormat(type=JSON_SCHEMA, json_schema=self._json_schema)
        # )
        ...

    def _extract_text(self, response):
        # Generic: concat TEXT parts from response.choices[0].message.content
        # Cohere: response.chat_response.text
        ...

    def do_generate(prompt):
        request = (
            self._build_cohere_request(prompt)
            if self._is_cohere
            else self._build_generic_request(prompt)
        )
        response = oci_client.chat(request)
        raw_text = self._extract_text(response)
        json_obj = parse JSON
        validate schema
        # Token usage (same shape for Generic/Cohere):
        # usage = response.data.chat_response.usage
        # prompt_tokens = usage.prompt_tokens
        # completion_tokens = usage.completion_tokens
        record metrics from response.data.chat_response.usage
        return SchematicGenerationResult
```

## Mapping degli “pezzi” da toccare (quando implementiamo)
- **Adapter file nuovo**:
  - `parlant/src/parlant/adapters/nlp/oci_service.py`
- **SDK factory**:
  - `parlant/src/parlant/sdk.py` → aggiungere `NLPServices.oci`
- **Server CLI**:
  - `parlant/src/parlant/bin/server.py`
    - `NLPServiceName` + `NLP_SERVICE_INITIALIZERS`
    - `--oci` flag + `require_env_keys`
- **Deps**:
  - `parlant/pyproject.toml` → optional dependency `oci = ["oci>=X"]`
- **Docs**:
  - `parlant/llms.txt` → env vars OCI
  - `parlant/docs/adapters/nlp/oci.md` (questo file)

## Open Questions (da risolvere prima del coding)
1) **Embeddings OCI**: **RISOLTO** → usare `embed_text` e default `cohere.embed-multilingual-v3.0` (documentare dimensioni di questo modello come fallback). citeturn5search4turn1view0turn1view1
2) **Prezzi / costi OCI**: se serve documentare pricing o limiti di costo, servono riferimenti ufficiali.
3) **Formato JSON**: **RISOLTO** → usare sempre `response_format` con `JSON_SCHEMA` (Generic/Cohere) basato su `self.schema.model_json_schema()`. Mantenere `jsonfinder` come fallback se il modello non rispetta lo schema.
4) **Rate limiting / retry**: **RISOLTO** → retry su `ServiceError` 429/5xx e su timeout/connection; no retry su altri 4xx.
5) **Token usage**: **RISOLTO** → presente in `response.data.chat_response.usage.{prompt_tokens, completion_tokens}`.
6) **Model routing**: basta il prefix `cohere.` o serve altro (es. nuovi provider in OCI)?
7) **Auth**: **RISOLTO** → config `~/.oci/config` + override via env sono sufficienti.

## Conclusione
L’implementazione OCI in Parlant è fattibile seguendo i pattern esistenti:
- Routing interno come `vertex_service.py`.
- Generator “thin” per JSON come `openai_service.py`.
- Embeddings e moderation coerenti con altri provider.
La parte critica è la gestione della doppia API (Generic/Cohere) e la chiarezza sull’embedding endpoint OCI.

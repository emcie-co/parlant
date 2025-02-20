# DSPy Integration Roadmap

## Current Status
Our DSPy integration currently provides:
- Basic guideline classification using LLMs
- Response optimization with COPRO parameters
- Support for both OpenAI and Llama models
- Standalone scripts for testing and demonstration

## Integration Phases

### Phase 1: Core Integration (Current PR)
- [x] Implement GuidelineClassifier
- [x] Implement BatchOptimizedGuidelineManager
- [x] Add support for both OpenAI and Llama models
- [x] Create basic tests
- [x] Add demonstration scripts

### Phase 2: Engine Integration
- [ ] Create DSPyGuidelineProposer
  - [ ] Implement async guideline proposal
  - [ ] Add context support
  - [ ] Integrate with engine lifecycle
- [ ] Create DSPyMessageGenerator
  - [ ] Implement response optimization
  - [ ] Add support for multiple iterations
  - [ ] Handle context variables

### Phase 3: Server Integration
- [ ] Add DSPy Configuration
  - [ ] Create DSPyConfig class
  - [ ] Add environment variable support
  - [ ] Integrate with container
- [ ] Create API Endpoints
  - [ ] Add /dspy/optimize endpoint
  - [ ] Add /dspy/classify endpoint
  - [ ] Add metrics endpoints
- [ ] Add Service Registry
  - [ ] Create DSPyToolService
  - [ ] Add classification tool
  - [ ] Add optimization tool

### Phase 4: Storage & Metrics
- [ ] Create DSPyStore
  - [ ] Add metrics collection
  - [ ] Add performance tracking
  - [ ] Add error logging
- [ ] Add Emission Support
  - [ ] Track API calls
  - [ ] Monitor response quality
  - [ ] Log optimization metrics

### Phase 5: Testing & Documentation
- [ ] Add Integration Tests
  - [ ] Test engine integration
  - [ ] Test API endpoints
  - [ ] Test metrics collection
- [ ] Add Documentation
  - [ ] Update API docs
  - [ ] Add configuration guide
  - [ ] Add optimization guide

## Implementation Details

### Key Files to Create/Modify

```
src/parlant/
├── dspy_integration/
│   ├── __init__.py
│   ├── config.py              # DSPy configuration
│   ├── guideline_classifier.py
│   ├── guideline_optimizer.py
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── proposer.py       # DSPyGuidelineProposer
│   │   └── generator.py      # DSPyMessageGenerator
│   ├── api/
│   │   ├── __init__.py
│   │   ├── router.py         # DSPy API endpoints
│   │   └── models.py         # API models
│   ├── services/
│   │   ├── __init__.py
│   │   └── tools.py          # DSPy tools
│   └── storage/
│       ├── __init__.py
│       └── metrics.py        # Metrics storage
└── core/
    └── engines/
        └── dspy.py           # DSPy engine components
```

### Environment Variables
```bash
# DSPy Configuration
DSPY_MODEL=openai/gpt-3.5-turbo
DSPY_OPTIMIZER_BATCH_SIZE=5
DSPY_MAX_TOKENS=2000
DSPY_TEMPERATURE=1.0

# Metrics Configuration
DSPY_METRICS_ENABLED=true
DSPY_METRICS_RETENTION_DAYS=30
```

## Next Steps

1. **Current PR (Phase 1)**
   - Complete the current PR with basic functionality
   - Add comprehensive tests
   - Update documentation

2. **Engine Integration (Phase 2)**
   - Create new branch `feature/dspy-engine-integration`
   - Implement engine components
   - Add context support

3. **Server Integration (Phase 3)**
   - Create new branch `feature/dspy-server-integration`
   - Add configuration system
   - Implement API endpoints

4. **Storage & Metrics (Phase 4)**
   - Create new branch `feature/dspy-metrics`
   - Add metrics storage
   - Implement tracking

5. **Testing & Documentation (Phase 5)**
   - Create new branch `feature/dspy-testing`
   - Add integration tests
   - Complete documentation

## Timeline
- Phase 1: Current PR (1-2 days)
- Phase 2: Engine Integration (3-4 days)
- Phase 3: Server Integration (2-3 days)
- Phase 4: Storage & Metrics (2-3 days)
- Phase 5: Testing & Documentation (2-3 days)

Total estimated time: 2-3 weeks

## Dependencies
- Parlant core engine
- DSPy library
- OpenAI API / Llama models
- MongoDB (for metrics)

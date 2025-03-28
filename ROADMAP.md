# DSPy Integration Roadmap

## Overview
This integration enhances Parlant's guideline proposition capabilities using DSPy's advanced language model techniques. The integration is being developed in phases to ensure robust and maintainable code.

## Project Structure
```
src/parlant/
└── dspy_integration/
    ├── __init__.py
    ├── guideline_classifier.py     # DSPy classifier for guidelines
    ├── guideline_optimizer.py      # DSPy optimizer using COPRO
    └── engine/
        ├── __init__.py
        └── proposer.py            # DSPy guideline proposer implementation
```

## Implementation Phases

### Phase 1: Core Integration ✅
- [x] Implement `GuidelineClassifier` using DSPy
- [x] Implement `BatchOptimizedGuidelineManager` with COPRO
- [x] Add basic metrics tracking
- [x] Implement unit tests for core components

### Phase 2: Engine Integration (Current) 🔄
- [x] Create `DSPyGuidelineProposer` implementing `GuidelineProposer` interface
- [x] Add batch processing support
- [x] Integrate with existing metrics system
- [ ] Add comprehensive tests for the proposer
- [ ] Add integration tests with AlphaEngine

### Phase 3: Server Integration 📋
- [ ] Create configuration system
- [ ] Add environment variable support:
  ```bash
  DSPY_MODEL=openai/gpt-3.5-turbo
  DSPY_OPTIMIZER_BATCH_SIZE=5
  DSPY_MAX_TOKENS=2000
  DSPY_TEMPERATURE=1.0
  ```
- [ ] Implement API endpoints for DSPy-specific operations
- [ ] Add configuration validation

### Phase 4: Storage & Metrics 📊
- [ ] Implement metrics storage
- [ ] Add tracking for:
  - API calls
  - Token usage
  - Response latency
  - Classification accuracy
- [ ] Create dashboard for monitoring

### Phase 5: Testing & Documentation 📚
- [ ] Add end-to-end tests
- [ ] Create benchmark suite
- [ ] Write technical documentation
- [ ] Add usage examples

## Dependencies
- Parlant core engine
- DSPy library
- OpenAI API / Llama models
- MongoDB (for metrics)

## Timeline
- Phase 1: Complete ✅
- Phase 2: In Progress (1-2 days remaining)
- Phase 3: Not Started (2-3 days)
- Phase 4: Not Started (2-3 days)
- Phase 5: Not Started (2-3 days)

Total estimated time: 1.5-2 weeks remaining

## Notes
- The integration maintains Parlant's existing architecture while enhancing guideline proposition capabilities
- DSPy components are designed to be modular and independently testable
- All components follow Parlant's existing patterns and conventions

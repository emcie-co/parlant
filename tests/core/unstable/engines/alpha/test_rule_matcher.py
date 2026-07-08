# ruff: noqa: F401,F403
from tests.core.unstable.engines.alpha.test_guideline_matcher import *
from tests.core.unstable.engines.alpha import test_guideline_matcher as _source
from tests.core.unstable.engines.alpha.test_guideline_matcher import (
    create_guideline as create_rule,
    create_guideline_by_name as create_rule_by_name,
)


class ContextOfTest(_source.ContextOfTest):
    def __init__(self, container, sync_await, rules, logger):
        super().__init__(container, sync_await, guidelines=rules, logger=logger)

    @property
    def rules(self):
        return self.guidelines
